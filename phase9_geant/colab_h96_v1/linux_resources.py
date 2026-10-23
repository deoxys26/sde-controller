"""Versioned Linux/Colab resource policy and v2-runner monitor adapter.

Policy v3 distinguishes a literal unlimited value at every inspected cgroup
level from missing or malformed measurements. Only visible limits are
enforced; parent limits outside the cgroup mount namespace remain explicitly
uncertain. Process high-water and host-available RAM remain mandatory guards.
"""

from __future__ import annotations

import platform
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import psutil

try:  # ``resource`` is absent on Windows, but the module remains importable for tests.
    import resource
except ImportError:  # pragma: no cover - exercised on Windows hosts
    resource = None  # type: ignore[assignment]


MIB = 1024 * 1024
GIB = 1024 * MIB
RESOURCE_POLICY_ID = "colab-linux-process-memory-v3-visible-cgroup"
PROCESS_PEAK_LIMIT_BYTES = 2 * GIB
OS_AVAILABLE_RESERVE_BYTES = 384 * MIB
STARTUP_HEADROOM_BYTES = PROCESS_PEAK_LIMIT_BYTES + OS_AVAILABLE_RESERVE_BYTES
MIN_FREE_DISK_BYTES = 512 * MIB
V1_UNLIMITED_SENTINEL = 1 << 60


class LinuxResourceError(MemoryError):
    """Runtime resources crossed the frozen safety policy."""


def _parse_limit(raw: str, *, version: str) -> tuple[int | None, str]:
    value = raw.strip()
    if version == "cgroup_v2" and value == "max":
        return None, "unlimited"
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed {version} limit value: {raw!r}") from exc
    if number < 0:
        raise ValueError(f"negative {version} limit value: {raw!r}")
    if version == "cgroup_v1" and number >= V1_UNLIMITED_SENTINEL:
        return None, "unlimited_sentinel"
    return number, "finite"


def _read_nonnegative(path: Path) -> int:
    try:
        raw = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read required cgroup measurement {path}: {exc}") from exc
    try:
        result = int(raw)
    except ValueError as exc:
        raise ValueError(f"malformed required cgroup measurement {path}: {raw!r}") from exc
    if result < 0:
        raise ValueError(f"negative required cgroup measurement {path}: {raw!r}")
    return result


def _read_limit_file(path: Path, *, version: str) -> tuple[int | None, str]:
    try:
        raw = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read required cgroup limit {path}: {exc}") from exc
    return _parse_limit(raw, version=version)


def _unescape_mountinfo(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), value)


def _read_memberships(path: str | Path) -> tuple[list[tuple[str, str]], list[str]]:
    try:
        lines = Path(path).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [f"process cgroup membership is unreadable: {exc}"]
    groups: list[tuple[str, str]] = []
    issues: list[str] = []
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].startswith("/"):
            issues.append("malformed /proc/self/cgroup entry")
            continue
        hierarchy, controllers, membership = fields
        if hierarchy == "0" and controllers == "":
            groups.append(("cgroup_v2", membership))
        elif "memory" in controllers.split(","):
            groups.append(("cgroup_v1", membership))
    if not groups:
        issues.append("no cgroup v1 memory or unified v2 membership was discoverable")
    return groups, issues


def _mount_records(path: str | Path) -> tuple[list[dict[str, str]], str | None]:
    try:
        lines = Path(path).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], f"mountinfo is unavailable; only the supplied visible cgroup root can be inspected ({exc})"
    records: list[dict[str, str]] = []
    malformed = False
    for line in lines:
        if " - " not in line:
            malformed = True
            continue
        left, right = line.split(" - ", 1)
        lhs, rhs = left.split(), right.split()
        if len(lhs) < 5 or len(rhs) < 3:
            malformed = True
            continue
        filesystem, source, super_options = rhs[0], rhs[1], rhs[2]
        if filesystem == "cgroup2":
            kind = "cgroup_v2"
        elif filesystem == "cgroup" and "memory" in super_options.split(","):
            kind = "cgroup_v1"
        else:
            continue
        records.append({
            "kind": kind,
            "mount_root": _unescape_mountinfo(lhs[3]),
            "mount_point": _unescape_mountinfo(lhs[4]),
            "source": source,
        })
    if malformed:
        return records, "some mountinfo rows were malformed; mount mapping is not fully verified"
    return records, None


def _posix_components(value: str) -> tuple[str, ...] | None:
    if not value.startswith("/") or "\\" in value or "\x00" in value:
        return None
    raw_parts = value.split("/")[1:]
    if raw_parts and raw_parts[-1] == "":
        raw_parts.pop()
    if any(part in {"", ".", ".."} for part in raw_parts):
        return None
    return tuple(raw_parts)


def _matching_mount(kind: str, membership: str, records: list[dict[str, str]]) -> tuple[dict[str, str] | None, tuple[str, ...] | None]:
    parts = _posix_components(membership)
    if parts is None:
        return None, None
    matches: list[tuple[int, dict[str, str], tuple[str, ...]]] = []
    for record in records:
        if record["kind"] != kind:
            continue
        mount_parts = _posix_components(record["mount_root"])
        if mount_parts is None or parts[:len(mount_parts)] != mount_parts:
            continue
        matches.append((len(mount_parts), record, parts[len(mount_parts):]))
    if not matches:
        return None, parts
    _, record, suffix = max(matches, key=lambda item: item[0])
    return record, suffix


def _visible_chain(mount_root: Path, suffix: tuple[str, ...] | None) -> tuple[list[Path], bool]:
    """Return visible levels root-to-leaf; unsafe membership maps to root only."""
    if suffix is None:
        return [mount_root], False
    levels = [mount_root]
    current = mount_root
    for component in suffix:
        current = current / component
        levels.append(current)
    return levels, True


def _inspect_hierarchy(kind: str, levels: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    finite_headrooms: list[int] = []
    finite_limits: list[int] = []
    for directory in levels:
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"visible cgroup directory is missing or redirected: {directory}")
        if kind == "cgroup_v2":
            usage = _read_nonnegative(directory / "memory.current")
            max_limit, max_kind = _read_limit_file(directory / "memory.max", version=kind)
            high_limit, high_kind = _read_limit_file(directory / "memory.high", version=kind)
            limits = [("memory.max", max_limit, max_kind), ("memory.high", high_limit, high_kind)]
        else:
            usage = _read_nonnegative(directory / "memory.usage_in_bytes")
            hard_limit, hard_kind = _read_limit_file(directory / "memory.limit_in_bytes", version=kind)
            limits = [("memory.limit_in_bytes", hard_limit, hard_kind)]
        row_limits: list[dict[str, Any]] = []
        for name, limit, limit_kind in limits:
            row_limits.append({"name": name, "limit_bytes": limit, "kind": limit_kind})
            if limit is not None:
                finite_limits.append(limit)
                finite_headrooms.append(max(0, limit - usage))
        rows.append({"path": str(directory), "usage_bytes": usage, "limits": row_limits})
    return {
        "rows": rows,
        "leaf_usage_bytes": rows[-1]["usage_bytes"],
        "finite_limit_bytes": min(finite_limits) if finite_limits else None,
        "headroom_bytes": min(finite_headrooms) if finite_headrooms else None,
        "status": "finite" if finite_headrooms else "visible_unlimited",
    }


def read_cgroup_memory(
    cgroup_root: str | Path = "/sys/fs/cgroup", *,
    proc_cgroup_path: str | Path = "/proc/self/cgroup",
    mountinfo_path: str | Path = "/proc/self/mountinfo",
) -> dict[str, Any]:
    """Inspect all discoverable v1/v2 visible ancestor limits, fail closed on bad data.

    ``cgroup_root`` is the visible mount root and permits deterministic fixtures.
    We never resolve ``..`` from membership data into an unrelated host path.
    """
    visible_root = Path(cgroup_root)
    groups, membership_issues = _read_memberships(proc_cgroup_path)
    mounts, mount_issue = _mount_records(mountinfo_path)
    uncertainty: list[str] = list(membership_issues)
    if mount_issue:
        uncertainty.append(mount_issue)
    if not visible_root.is_dir():
        return {
            "source": "unavailable", "status": "invalid", "usage_bytes": None,
            "limit_bytes": None, "headroom_bytes": None, "visibility_uncertainty": uncertainty + [f"visible cgroup root is missing: {visible_root}"],
            "visible_levels": [], "memory_events": None,
        }
    if any("malformed" in issue for issue in membership_issues):
        return {
            "source": "unavailable", "status": "invalid", "usage_bytes": None,
            "limit_bytes": None, "headroom_bytes": None, "visibility_uncertainty": uncertainty,
            "visible_levels": [], "memory_events": None,
        }
    if mount_issue is not None and "malformed" in mount_issue:
        return {
            "source": "unavailable", "status": "invalid", "usage_bytes": None,
            "limit_bytes": None, "headroom_bytes": None, "visibility_uncertainty": uncertainty,
            "visible_levels": [], "memory_events": None,
        }
    if visible_root.is_symlink():
        return {
            "source": "unavailable", "status": "invalid", "usage_bytes": None,
            "limit_bytes": None, "headroom_bytes": None, "visibility_uncertainty": uncertainty + [f"visible cgroup root is redirected: {visible_root}"],
            "visible_levels": [], "memory_events": None,
        }

    if not groups:
        # Do not infer "unlimited" merely because /proc/self/cgroup was absent.
        return {
            "source": "unavailable", "status": "invalid", "usage_bytes": None,
            "limit_bytes": None, "headroom_bytes": None, "visibility_uncertainty": uncertainty,
            "visible_levels": [], "memory_events": None,
        }

    inspected: list[tuple[str, dict[str, Any]]] = []
    try:
        for kind, membership in groups:
            record, suffix = _matching_mount(kind, membership, mounts)
            if record is None:
                parts = _posix_components(membership)
                if parts is None:
                    levels, mapped = [visible_root], False
                    uncertainty.append(f"{kind} membership contains traversal components; inspected visible mount root only")
                elif mount_issue is not None:
                    # Mountinfo may be hidden in containers. Inspect the supplied
                    # mount root and record, never assume absent ancestors are unlimited.
                    base = visible_root / ("memory" if kind == "cgroup_v1" else "")
                    leaf = base.joinpath(*parts)
                    if not parts:
                        levels, mapped = [base], True
                    elif not leaf.exists():
                        raise ValueError(f"process cgroup directory is not visible beneath supplied root: {leaf}")
                    else:
                        levels = [base]
                        cursor = base
                        for component in parts:
                            cursor = cursor / component
                            levels.append(cursor)
                        mapped = True
                    uncertainty.append(f"{kind} mount root could not be confirmed from mountinfo; only supplied visible tree inspected")
                else:
                    # A readable mount table that cannot map the membership is
                    # inconsistent. It is unsafe to infer its ancestor chain.
                    raise ValueError(f"cannot map {kind} membership {membership!r} to a visible memory mount")
            else:
                mountpoint_parts = _posix_components(record["mount_point"])
                if mountpoint_parts is None:
                    raise ValueError("mountinfo contains a malformed or redirected memory mountpoint")
                if visible_root == Path("/sys/fs/cgroup"):
                    base = Path("/").joinpath(*mountpoint_parts)
                else:
                    # Explicit roots are injectable only for isolated synthetic
                    # fixtures; the live default uses the mountinfo path.
                    base = visible_root / ("memory" if kind == "cgroup_v1" else "")
                record_root = _posix_components(record["mount_root"])
                if record_root is None:
                    raise ValueError("mountinfo contains traversal in a cgroup mount root")
                # The testable root is the mount point. The suffix, not the
                # proc path's absolute spelling, is joined beneath it.
                levels, mapped = _visible_chain(base, suffix)
            if not mapped:
                uncertainty.append(f"{kind} process membership is not fully mapped inside the visible namespace")
            inspected.append((kind, _inspect_hierarchy(kind, levels)))
    except (OSError, ValueError) as exc:
        return {
            "source": "+".join(kind for kind, _ in inspected) or "unavailable",
            "status": "invalid", "usage_bytes": None, "limit_bytes": None,
            "headroom_bytes": None, "visibility_uncertainty": uncertainty + [str(exc)],
            "visible_levels": [row for _, details in inspected for row in details["rows"]],
            "memory_events": None,
        }

    all_levels = [row for _, details in inspected for row in details["rows"]]
    finite = [details["headroom_bytes"] for _, details in inspected if details["status"] == "finite"]
    limits = [details["finite_limit_bytes"] for _, details in inspected if details["finite_limit_bytes"] is not None]
    # Optional events are diagnostic only: inability to read them does not
    # turn an otherwise valid memory.current/max/high reading into capacity.
    events = None
    for _, details in inspected:
        leaf = Path(details["rows"][-1]["path"])
        try:
            parsed: dict[str, int] = {}
            for line in (leaf / "memory.events").read_text(encoding="ascii").splitlines():
                key, value = line.split()
                parsed[key] = int(value)
            events = parsed
            break
        except (OSError, UnicodeError, ValueError):
            continue
    if any(details["status"] == "finite" for _, details in inspected):
        status = "finite"
        headroom = min(value for value in finite if value is not None)
    else:
        status, headroom = "visible_unlimited", None
    return {
        "source": "+".join(kind for kind, _ in inspected),
        "status": status,
        "usage_bytes": inspected[0][1]["leaf_usage_bytes"],
        "limit_bytes": min(limits) if limits else None,
        "headroom_bytes": headroom,
        "visibility_uncertainty": sorted(set(uncertainty + ["limits outside the visible cgroup mount/namespace are not observable"])),
        "visible_levels": all_levels,
        "memory_events": events,
    }


def _process_high_water_bytes() -> tuple[int | None, str]:
    if resource is None:
        return None, "unavailable"
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not usage or usage < 0:
        return None, "unavailable"
    # Linux reports KiB; macOS reports bytes. This module is Linux-only.
    return int(usage * 1024), "resource.getrusage.ru_maxrss_linux_kib"


def snapshot(*, disk_path: str | Path = "/content", cgroup_root: str | Path = "/sys/fs/cgroup",
             proc_cgroup_path: str | Path = "/proc/self/cgroup",
             mountinfo_path: str | Path = "/proc/self/mountinfo",
             process: psutil.Process | None = None) -> dict[str, Any]:
    if not platform.system().lower().startswith("linux"):
        raise LinuxResourceError("Colab resource adapter is Linux-only")
    active = process or psutil.Process()
    rss = int(active.memory_info().rss)
    high_water, source = _process_high_water_bytes()
    cgroup = read_cgroup_memory(cgroup_root, proc_cgroup_path=proc_cgroup_path, mountinfo_path=mountinfo_path)
    return {
        "process_rss_bytes": rss,
        "process_peak_bytes": high_water,
        "process_peak_source": source,
        "os_available_ram_bytes": int(psutil.virtual_memory().available),
        "disk_free_bytes": int(shutil.disk_usage(disk_path).free),
        "cgroup_source": cgroup["source"],
        "cgroup_status": cgroup["status"],
        "cgroup_memory_usage_bytes": cgroup["usage_bytes"],
        "cgroup_memory_limit_bytes": cgroup["limit_bytes"],
        "cgroup_memory_headroom_bytes": cgroup["headroom_bytes"],
        "cgroup_visibility_uncertainty": cgroup["visibility_uncertainty"],
        "cgroup_visible_levels": cgroup["visible_levels"],
        "cgroup_memory_events": cgroup["memory_events"],
    }


def _validate_snapshot(current: Mapping[str, Any], *, in_run: bool = False) -> None:
    required = {
        "process_rss_bytes", "process_peak_bytes", "process_peak_source", "os_available_ram_bytes",
        "disk_free_bytes", "cgroup_source", "cgroup_status", "cgroup_memory_usage_bytes",
        "cgroup_memory_limit_bytes", "cgroup_memory_headroom_bytes", "cgroup_visibility_uncertainty",
        "cgroup_visible_levels", "cgroup_memory_events",
    }
    label = "In-run resource snapshot" if in_run else "Resource snapshot"
    if set(current) != required:
        raise LinuxResourceError(f"{label} is incomplete")
    for key in ("process_rss_bytes", "os_available_ram_bytes", "disk_free_bytes"):
        if not isinstance(current[key], int) or current[key] < 0 or (key == "process_rss_bytes" and current[key] == 0):
            raise LinuxResourceError(f"{label} field is invalid: {key}")
    peak = current["process_peak_bytes"]
    if not isinstance(peak, int) or peak <= 0:
        raise LinuxResourceError(f"{label} process high-water is unavailable or invalid")
    if not isinstance(current["process_peak_source"], str) or not current["process_peak_source"]:
        raise LinuxResourceError(f"{label} process high-water source is unknown")
    if not isinstance(current["cgroup_source"], str) or not current["cgroup_source"]:
        raise LinuxResourceError(f"{label} cgroup source is unknown")
    usage = current["cgroup_memory_usage_bytes"]
    if not isinstance(usage, int) or usage < 0:
        raise LinuxResourceError(f"{label} cgroup usage is missing or invalid")
    levels = current["cgroup_visible_levels"]
    if not isinstance(levels, list) or not levels:
        raise LinuxResourceError(f"{label} visible cgroup levels are malformed")
    finite_limit_found = False
    for level in levels:
        if not isinstance(level, dict) or not isinstance(level.get("path"), str):
            raise LinuxResourceError(f"{label} visible cgroup level is malformed")
        level_usage = level.get("usage_bytes")
        limits = level.get("limits")
        if not isinstance(level_usage, int) or level_usage < 0 or not isinstance(limits, list) or not limits:
            raise LinuxResourceError(f"{label} visible cgroup measurements are incomplete")
        for item in limits:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise LinuxResourceError(f"{label} visible cgroup limit is malformed")
            kind = item.get("kind")
            value = item.get("limit_bytes")
            if kind == "finite":
                if not isinstance(value, int) or value < 0:
                    raise LinuxResourceError(f"{label} finite visible cgroup limit is invalid")
                finite_limit_found = True
            elif kind in {"unlimited", "unlimited_sentinel"}:
                if value is not None:
                    raise LinuxResourceError(f"{label} unlimited visible cgroup limit conflicts with a value")
            else:
                raise LinuxResourceError(f"{label} visible cgroup limit kind is unknown")
    events = current["cgroup_memory_events"]
    if events is not None and (not isinstance(events, dict) or any(
        not isinstance(key, str) or not isinstance(value, int) or value < 0 for key, value in events.items()
    )):
        raise LinuxResourceError(f"{label} cgroup event counters are malformed")
    if current["cgroup_status"] not in {"finite", "visible_unlimited"}:
        uncertainty = current["cgroup_visibility_uncertainty"]
        raise LinuxResourceError(f"Cgroup measurements are invalid/unavailable: {uncertainty}")
    if not isinstance(current["cgroup_visibility_uncertainty"], list) or any(
        not isinstance(item, str) for item in current["cgroup_visibility_uncertainty"]
    ):
        raise LinuxResourceError(f"{label} cgroup visibility uncertainty is malformed")
    headroom = current["cgroup_memory_headroom_bytes"]
    if current["cgroup_status"] == "finite":
        if not finite_limit_found or not isinstance(current["cgroup_memory_limit_bytes"], int) or current["cgroup_memory_limit_bytes"] < 0:
            raise LinuxResourceError(f"{label} finite cgroup limit is missing or invalid")
        if not isinstance(headroom, int) or headroom < 0:
            raise LinuxResourceError(f"{label} finite cgroup headroom is missing or invalid")
    elif finite_limit_found or headroom is not None or current["cgroup_memory_limit_bytes"] is not None:
        raise LinuxResourceError(f"{label} unlimited cgroup status conflicts with finite measurements")


def _check_policy(current: Mapping[str, Any], *, startup: bool) -> None:
    _validate_snapshot(current)
    if current["disk_free_bytes"] < MIN_FREE_DISK_BYTES:
        raise LinuxResourceError(f"Disk free space is below {MIN_FREE_DISK_BYTES} bytes")
    required_host = STARTUP_HEADROOM_BYTES if startup else OS_AVAILABLE_RESERVE_BYTES
    if current["os_available_ram_bytes"] < required_host:
        label = "startup headroom" if startup else "reserve"
        raise LinuxResourceError(f"OS-available memory is below {label} {required_host} bytes")
    if current["cgroup_status"] == "finite":
        required_cgroup = STARTUP_HEADROOM_BYTES if startup else OS_AVAILABLE_RESERVE_BYTES
        if current["cgroup_memory_headroom_bytes"] < required_cgroup:
            label = "startup headroom" if startup else "reserve"
            raise LinuxResourceError(f"Cgroup memory headroom is below {label} {required_cgroup} bytes")
    if current["process_peak_bytes"] > PROCESS_PEAK_LIMIT_BYTES:
        raise LinuxResourceError(f"Process high-water exceeds {PROCESS_PEAK_LIMIT_BYTES} bytes")


def preflight(*, disk_path: str | Path = "/content", snapshot_fn: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    current = dict((snapshot_fn or (lambda: snapshot(disk_path=disk_path)))())
    _check_policy(current, startup=True)
    return {
        "status": "pass",
        "snapshot": current,
        "policy": {
            "policy_id": RESOURCE_POLICY_ID,
            "process_peak_limit_bytes": PROCESS_PEAK_LIMIT_BYTES,
            "startup_headroom_bytes": STARTUP_HEADROOM_BYTES,
            "in_run_os_available_reserve_bytes": OS_AVAILABLE_RESERVE_BYTES,
            "in_run_cgroup_headroom_reserve_bytes": OS_AVAILABLE_RESERVE_BYTES,
            "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
            "process_peak_measurement": current["process_peak_source"],
            "visible_unlimited_cgroup_policy": "allow literal unlimited visible limits only; host and process guards remain mandatory",
            "cgroup_visibility_limitation": "finite limits outside the visible mount/namespace cannot be discovered and are recorded as uncertainty",
            "invalid_or_missing_cgroup_policy": "fail closed before output creation or training",
        },
    }


class LinuxMemoryMonitor:
    """Drop-in ``sample``/``summary`` interface accepted by the v2 trainer."""

    def __init__(self, *, snapshot_fn: Callable[[], Mapping[str, Any]] | None = None) -> None:
        self.snapshot_fn = snapshot_fn or snapshot
        self.samples = 0
        self.max_rss_bytes = 0
        self.max_high_water_bytes = 0
        self.min_os_available_bytes: int | None = None
        self.min_cgroup_headroom_bytes: int | None = None
        self.min_disk_free_bytes: int | None = None
        self.sources: set[str] = set()
        self.last: dict[str, Any] | None = None
        self.cgroup_uncertainties: set[str] = set()

    def sample(self, phase: str) -> dict[str, Any]:
        values = dict(self.snapshot_fn())
        _validate_snapshot(values, in_run=True)
        self.samples += 1
        self.max_rss_bytes = max(self.max_rss_bytes, int(values["process_rss_bytes"]))
        high = int(values["process_peak_bytes"])
        self.max_high_water_bytes = max(self.max_high_water_bytes, high)
        self.sources.add(str(values["process_peak_source"]))
        available = int(values["os_available_ram_bytes"])
        disk_free = int(values["disk_free_bytes"])
        self.min_os_available_bytes = available if self.min_os_available_bytes is None else min(self.min_os_available_bytes, available)
        self.min_disk_free_bytes = disk_free if self.min_disk_free_bytes is None else min(self.min_disk_free_bytes, disk_free)
        self.cgroup_uncertainties.update(values["cgroup_visibility_uncertainty"])
        self.last = values
        if values["cgroup_status"] == "finite":
            headroom = int(values["cgroup_memory_headroom_bytes"])
            self.min_cgroup_headroom_bytes = headroom if self.min_cgroup_headroom_bytes is None else min(self.min_cgroup_headroom_bytes, headroom)
        _check_policy(values, startup=(phase == "startup_before_output"))
        if values["cgroup_status"] == "finite" and self.min_cgroup_headroom_bytes is not None:
            reserve = STARTUP_HEADROOM_BYTES if phase == "startup_before_output" else OS_AVAILABLE_RESERVE_BYTES
            if self.min_cgroup_headroom_bytes < reserve:
                raise LinuxResourceError(f"Cgroup memory headroom is below {reserve} bytes during {phase}")
        return values

    def summary(self) -> dict[str, Any]:
        return {
            "policy_id": RESOURCE_POLICY_ID,
            "samples": self.samples,
            "max_sampled_process_rss_bytes": self.max_rss_bytes,
            "max_process_high_water_bytes": self.max_high_water_bytes if self.sources else None,
            "process_peak_measurement_sources": sorted(self.sources),
            "min_os_available_ram_bytes": self.min_os_available_bytes,
            "min_cgroup_headroom_bytes": self.min_cgroup_headroom_bytes,
            "min_disk_free_bytes": self.min_disk_free_bytes,
            "cgroup_visibility_uncertainty": sorted(self.cgroup_uncertainties),
            "process_peak_limit_bytes": PROCESS_PEAK_LIMIT_BYTES,
            "startup_headroom_bytes": STARTUP_HEADROOM_BYTES,
            "in_run_os_available_reserve_bytes": OS_AVAILABLE_RESERVE_BYTES,
            "in_run_cgroup_headroom_reserve_bytes": OS_AVAILABLE_RESERVE_BYTES,
            "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
            "last_snapshot": self.last,
            "measurement_semantics": "RSS, ru_maxrss high-water, host-available RAM, cgroup usage/visible limits, and disk are separate; visible unlimited is not a claim about inaccessible ancestors",
        }
