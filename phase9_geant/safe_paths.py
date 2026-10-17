"""Fail-closed path checks for isolated GEANT v2 experiment I/O.

These checks detect existing symlink, junction, and reparse-point components.
They do not provide race-free OS handles against a privileged concurrent writer.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePath


V2_RELATIVE = Path("data") / "processed" / "geant_validity_v2"
_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _reject_parent_traversal(path: Path) -> None:
    if ".." in PurePath(os.fspath(path)).parts:
        raise ValueError(f"Parent traversal is not allowed in path: {path}")


def _file_attributes(path: Path, info: os.stat_result) -> int:
    """Return platform file attributes (separate for deterministic guard tests)."""
    del path
    return int(getattr(info, "st_file_attributes", 0))


def checked_absolute(path: str | Path, *, allow_missing: bool = False) -> Path:
    """Normalize and inspect existing components, stopping at first missing one.

    ``allow_missing`` is retained for caller compatibility. A missing suffix is
    not itself a redirection: callers that read an input must separately require
    that the expected root/file exists, while create paths still use exclusive
    creation and post-creation containment checks.
    """
    original = Path(path)
    _reject_parent_traversal(original)
    absolute = Path(os.path.abspath(os.fspath(original)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if current.is_symlink():
            raise ValueError(f"Symlink path component is not allowed: {current}")
        is_junction = getattr(current, "is_junction", None)
        if is_junction is not None and is_junction():
            raise ValueError(f"Junction path component is not allowed: {current}")
        if _file_attributes(current, info) & _REPARSE_POINT_ATTRIBUTE:
            raise ValueError(f"Reparse-point path component is not allowed: {current}")
    return absolute


def canonical_repository_root(repository_root: str | Path) -> Path:
    original = checked_absolute(repository_root)
    if not original.is_dir():
        raise NotADirectoryError(f"Repository root is not a directory: {original}")
    canonical = original.resolve(strict=True)
    if canonical != original:
        raise ValueError(f"Repository root resolves through an unexpected alias: {original}")
    return canonical


def validate_v2_root(data_root: str | Path, repository_root: str | Path) -> Path:
    """Require the canonical v2 root or a named, versioned rebuild sibling."""
    repo = canonical_repository_root(repository_root)
    root = checked_absolute(data_root)
    expected = repo / V2_RELATIVE
    processed = repo / "data" / "processed"
    is_original = os.path.normcase(os.fspath(root)) == os.path.normcase(os.fspath(expected))
    is_versioned = (
        root.parent == processed
        and root.name.startswith("geant_validity_v2_rebuild_")
        and len(root.name) > len("geant_validity_v2_rebuild_")
        and all(c.isalnum() or c in "_-" for c in root.name)
    )
    if not (is_original or is_versioned):
        raise ValueError(
            f"Data root must be the canonical v2 root or a direct named rebuild child: {root}"
        )
    if not root.exists():
        raise FileNotFoundError(f"GEANT v2 data root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"GEANT v2 data root is not a directory: {root}")
    if processed.resolve(strict=True) != processed:
        raise ValueError("Processed-data parent is not canonical")
    canonical = root.resolve(strict=True)
    if canonical != root or not canonical.is_dir():
        raise ValueError(f"Data root is not a canonical directory: {root}")
    return canonical


def validate_fresh_child(
    data_root: str | Path,
    output_path: str | Path,
    repository_root: str | Path,
    *,
    expected_name: str,
) -> tuple[Path, Path]:
    root = validate_v2_root(data_root, repository_root)
    output = checked_absolute(output_path, allow_missing=True)
    expected = root / expected_name
    if os.path.normcase(os.fspath(output)) != os.path.normcase(os.fspath(expected)):
        raise ValueError(f"Output must be the dedicated child {expected}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to reuse existing output directory: {output}")
    if output.parent.resolve(strict=True) != root:
        raise ValueError("Output parent does not resolve to the validated v2 root")
    return root, output


def validate_existing_child(
    data_root: str | Path,
    child_path: str | Path,
    repository_root: str | Path,
    *,
    expected_name: str,
) -> tuple[Path, Path]:
    root = validate_v2_root(data_root, repository_root)
    child = checked_absolute(child_path)
    expected = root / expected_name
    if os.path.normcase(os.fspath(child)) != os.path.normcase(os.fspath(expected)):
        raise ValueError(f"Input must be the allow-listed child {expected}")
    resolved = child.resolve(strict=True)
    if resolved != expected or resolved.parent != root or not resolved.is_dir():
        raise ValueError(f"Input directory is not a canonical direct child: {child}")
    return root, resolved


def verify_new_directory_containment(output: Path, data_root: Path) -> None:
    checked = checked_absolute(output)
    root = validate_v2_root(data_root, data_root.parents[2])
    resolved = checked.resolve(strict=True)
    if resolved.parent != root or not resolved.is_dir():
        raise ValueError(f"New output escaped the validated v2 root: {resolved}")


def require_regular_file(path: Path, parent: Path) -> Path:
    checked = checked_absolute(path)
    resolved_parent = parent.resolve(strict=True)
    resolved = checked.resolve(strict=True)
    if resolved.parent != resolved_parent or not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"Expected a regular direct-child file: {path}")
    return resolved

