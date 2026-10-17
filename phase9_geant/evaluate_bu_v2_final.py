"""Guarded GEANT v2 final-test evaluator (currently blocked pending source freeze).

Importing this module performs no filesystem access. The default CLI path is
preflight-only. Real evaluation remains disabled until a reviewed mechanism
binds imported code to approved source bytes. If enabled in a reviewed future
version, test arrays would be opened only after explicit CLI authorization and
durable attempt-state publication.
"""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import time
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


PROTOCOL_ID = "geant-validity-v2-final-test-bu-v1"
DATA_PROTOCOL_ID = "geant-validity-v2-train-median"
CANONICAL_REPO_ROOT = Path(r"C:\Users\Admin\OneDrive\Desktop\DN_proj")
TOPOLOGY_SHA256 = "38C90629F30AF70FE840E33071B12CC40498BB910E8DBEB2FA7863A7B645D121"
TOPOLOGY_REL = Path("data/raw/geant_topology/geant.txt")
V2_REL = Path("data/processed/geant_validity_v2")
PROTOCOL_REL = V2_REL / "final_test_protocol_v1" / "PROTOCOL.md"
OUTPUT_REL = V2_REL / "final_test_evaluation_v1"
TEST_X_REL = V2_REL / "X_test.npy"
TEST_Y_REL = V2_REL / "y_test.npy"
WINDOWS_REL = V2_REL / "window_starts.json"
METADATA_REL = V2_REL / "metadata.json"
HISTORY = 5
LINKS = 72
NODES = 22
UNDIRECTED_LINKS = 36
SPLITS = {"train": (0, 8020), "validation": (8020, 9740), "test": (9740, 11460)}
SCALER_MEAN = 0.031580968237832956
SCALER_SCALE = 0.042026947460037614
EWMA_ALPHA = 0.8
AR_COEFFICIENTS = (-0.01057, 0.07411, 0.13645, 0.13489, 0.65767)
AR_INTERCEPT = 0.000212

# Exact source-to-runtime identity is not established by hashing source files
# after this module has already been imported. Keep the real evaluator blocked
# until a separately reviewed freeze mechanism binds imported code to approved
# source bytes. Synthetic lifecycle tests may override this flag only in-process.
SOURCE_FREEZE_REVIEWED = False

CHECKPOINTS = {
    42: (
        Path("data/processed/geant_validity_v2/neural_bu_v2_full_seed_42/best_validation.pt"),
        "8869a4bd2b36c439456b63395323e4ac09f89b2957f70dfddb5aab2da00a47f3",
    ),
    123: (
        Path("data/processed/geant_validity_v2/neural_bu_v2_full_seed_123/best_validation.pt"),
        "efa5e5f0662bf1f241ebfceca2f613eefac0727ea28203a92e1ae56995ecc554",
    ),
    2026: (
        Path("data/processed/geant_validity_v2/neural_bu_v2_full_seed_2026/best_validation.pt"),
        "072f57af991e2b6c84e25331eea094634ce2b3fba02da24d6127228469c66938",
    ),
}

# IO_REPARSE_TAG_CLOUD and CLOUD_1..CLOUD_F. Other reparse tags fail closed.
_CLOUD_TAGS = frozenset(0x9000001A + (index << 12) for index in range(16))
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
# Retained only to make old direct-call attempts fail explicitly. This object
# grants no authority; unrestricted code in this process is outside the gate.
_CLI_AUTHORIZATION_TOKEN = object()
_CLI_AUTHORIZATION_CONTEXT = contextvars.ContextVar("final_eval_cli_authorized", default=False)

_ISSUED_PREFLIGHTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class Preflight:
    repo_root: Path
    source_root: Path
    approved_manifest_sha256: str | None
    data_root: Path
    output: Path
    topology: Path
    test_x: Path
    test_y: Path
    window_map: Path
    checkpoints: Mapping[int, Path]
    manifests: Mapping[int, Path]
    topology_edges: tuple[tuple[str, str], ...]
    checkpoint_hashes: Mapping[int, str]
    checkpoint_identities: Mapping[int, Mapping[str, int]]
    manifest_identities: Mapping[int, Mapping[str, int]]
    test_identities: Mapping[str, Mapping[str, int]]
    topology_identity: Mapping[str, int]
    metadata_identity: Mapping[str, int]
    protocol_identity: Mapping[str, int]
    protocol_sha256: str
    topology_snapshot: bytes
    source_snapshot_sha256: str
    source_snapshot_identities: Mapping[str, Mapping[str, int]]


def _canonical_case(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path.absolute())))


def _component_reparse_tag(path: Path) -> int | None:
    """Read metadata only; never follows or opens file contents."""
    info = path.lstat()
    attrs = getattr(info, "st_file_attributes", 0)
    tag = getattr(info, "st_reparse_tag", None)
    # Some Windows stat implementations expose st_reparse_tag=0 for ordinary
    # files. A zero tag without the REPARSE_POINT attribute is not a reparse
    # point; zero with that attribute remains unverifiable and is rejected.
    if not (attrs & _FILE_ATTRIBUTE_REPARSE_POINT) and tag in (None, 0):
        return None
    if tag is None:
        raise ValueError(f"Cannot identify reparse tag safely: {path}")
    return int(tag)


def _stat_identity(info: os.stat_result, *, label: str) -> dict[str, int]:
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected a regular file for identity check: {label}")
    fields = {
        "device": int(info.st_dev),
        "file_id": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
    }
    if fields["file_id"] <= 0:
        raise ValueError(f"Filesystem does not expose a reliable file ID: {label}")
    return fields


def _path_identity(path: Path) -> dict[str, int]:
    return _stat_identity(path.lstat(), label=str(path))


def _validate_cloud_identity(tag: int | None, identity: Mapping[str, int], path: Path) -> None:
    if tag in _CLOUD_TAGS and (
        identity.get("device", -1) < 0
        or identity.get("file_id", 0) <= 0
        or identity.get("size", -1) < 0
        or identity.get("mtime_ns", -1) < 0
    ):
        raise ValueError(f"Cloud placeholder lacks verifiable file identity: {path}")


def _open_pinned_file(path: Path, expected_identity: Mapping[str, int], repo_root: Path):
    """Open once and prove the path and handle refer to the preflight file ID."""
    validate_path_components(path, repo_root=repo_root, must_exist=True)
    before = _path_identity(path)
    if dict(expected_identity) != before:
        raise ValueError(f"Approved file identity changed since preflight: {path}")
    handle = path.open("rb")
    try:
        opened = _stat_identity(os.fstat(handle.fileno()), label=str(path))
        after = _path_identity(path)
        if opened != before or after != before:
            raise ValueError(f"Opened file does not match the preflight identity: {path}")
        return handle
    except BaseException:
        handle.close()
        raise


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().upper()


def validate_path_components(
    path: Path,
    *,
    repo_root: Path,
    must_exist: bool,
    allowed_cloud_tags: frozenset[int] = _CLOUD_TAGS,
    reparse_tag_reader: Callable[[Path], int | None] = _component_reparse_tag,
) -> Path:
    """Validate every existing path component and reject redirection aliases."""
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Path must be absolute and contain no traversal: {candidate}")
    root = Path(repo_root).absolute()
    # Inspect ancestors as well as descendants (including the OneDrive root).
    anchor = Path(candidate.anchor)
    current = anchor
    parts = candidate.parts[1:]
    for part in parts:
        current = current / part
        if not os.path.lexists(current):
            if must_exist:
                raise FileNotFoundError(current)
            continue
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Symlink path component is forbidden: {current}")
        is_junction = getattr(current, "is_junction", None)
        if is_junction is not None and is_junction():
            raise ValueError(f"Junction path component is forbidden: {current}")
        # Mount points can redirect an otherwise ordinary-looking child path.
        # The volume root itself is the expected anchor; reject mount points
        # among all subsequent components.
        if current != anchor and current.is_mount():
            raise ValueError(f"Mount-point path component is forbidden: {current}")
        tag = reparse_tag_reader(current)
        if tag is not None and tag not in allowed_cloud_tags:
            raise ValueError(f"Unrecognized/path-redirection reparse tag {tag:#x}: {current}")
        if tag in allowed_cloud_tags:
            target_info = current.stat()
            if (
                int(getattr(info, "st_dev", -1)) < 0
                or int(getattr(info, "st_ino", 0)) <= 0
                or int(getattr(info, "st_dev", -1)) != int(getattr(target_info, "st_dev", -2))
                or int(getattr(info, "st_ino", 0)) != int(getattr(target_info, "st_ino", -1))
            ):
                raise ValueError(f"Cloud placeholder target identity is unverifiable: {current}")
        # Cloud placeholders carry provider state, not a substitute path. Their
        # resolved name must remain exactly the name in the approved path.
        resolved = current.resolve(strict=True)
        if _canonical_case(resolved) != _canonical_case(current):
            raise ValueError(f"Path component redirects from its canonical location: {current}")
    if must_exist and not os.path.lexists(candidate):
        raise FileNotFoundError(candidate)
    resolved_candidate = candidate.resolve(strict=must_exist)
    if _canonical_case(resolved_candidate) != _canonical_case(candidate):
        raise ValueError(f"Path resolves through an alias: {candidate} -> {resolved_candidate}")
    # `root` is checked independently by preflight; retain explicit containment.
    resolved_root = root.resolve(strict=True)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        # Paths such as the topology/checkpoints are repository children too.
        raise ValueError(f"Path escapes canonical repository root: {candidate}") from exc
    return resolved_candidate


def _require_exact_path(candidate: Path, expected: Path, repo_root: Path) -> Path:
    expected_abs = (repo_root / expected).absolute()
    candidate_abs = Path(candidate).absolute()
    if _canonical_case(candidate_abs) != _canonical_case(expected_abs):
        raise ValueError(f"Artifact path is not allow-listed: {candidate}")
    return candidate_abs


def directed_edges(links: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for source, destination in links:
        if source == destination:
            raise ValueError("Topology contains a self-link")
        pair = frozenset((source, destination))
        if pair in seen:
            raise ValueError("Topology contains a duplicate undirected link")
        seen.add(pair)
        result.extend(((source, destination), (destination, source)))
    return tuple(result)


def validate_topology(nodes: Mapping[str, object], links: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    if len(nodes) != NODES or len(set(nodes)) != NODES:
        raise ValueError(f"Expected {NODES} unique topology nodes, got {len(set(nodes))}")
    if len(links) != UNDIRECTED_LINKS:
        raise ValueError(f"Expected {UNDIRECTED_LINKS} undirected links, got {len(links)}")
    if any(u not in nodes or v not in nodes for u, v in links):
        raise ValueError("Topology link references an undeclared node")
    edges = directed_edges(links)
    if len(edges) != LINKS:
        raise ValueError(f"Expected {LINKS} directed links, got {len(edges)}")
    return edges


def _validate_protocol_text(content: bytes) -> None:
    text = content.decode("utf-8")
    required = (
        PROTOCOL_ID,
        DATA_PROTOCOL_ID,
        str(CANONICAL_REPO_ROOT),
        "[9740,11460)",
        "X_test.npy",
        "y_test.npy",
        "window_starts.json",
        "metadata.json",
        "file_id",
        "on-disk source snapshot SHA-256",
        "Final evaluation is blocked pending an independently reviewed source freeze",
        str(TOPOLOGY_SHA256),
        "22 unique nodes, 36 undirected links, and 72 directed links",
        "Do not retry",
        "separate, explicit approval",
        f"mean `mu = {SCALER_MEAN}`",
        f"scale `sigma = {SCALER_SCALE}`",
        f"alpha = {EWMA_ALPHA}",
        "[-0.01057, 0.07411, 0.13645, 0.13489, 0.65767]",
        f"b = {AR_INTERCEPT}",
        "float64",
    )
    required += tuple(
        str(path).replace("\\", "/") + "` | `" + digest + "`"
        for path, digest in CHECKPOINTS.values()
    )
    absent = [item for item in required if item not in text]
    if absent:
        raise ValueError(f"Frozen protocol is missing required invariant(s): {absent}")


def _capture_source_snapshot(source_root: Path) -> tuple[str, dict[str, Mapping[str, int]], bytes]:
    """Hash pinned on-disk snapshots; this does not identify loaded Python code."""
    digest = hashlib.sha256()
    identities: dict[str, Mapping[str, int]] = {}
    protocol_bytes = b""
    for relative in (
        Path("phase9_geant/evaluate_bu_v2_final.py"),
        Path("phase9_geant/parse_geant.py"),
        Path("src/gnn/model.py"),
        PROTOCOL_REL,
    ):
        source_path = source_root / relative
        validate_path_components(source_path, repo_root=source_root, must_exist=True)
        identity = _path_identity(source_path)
        with _open_pinned_file(source_path, identity, source_root) as handle:
            data = handle.read()
            if (_stat_identity(os.fstat(handle.fileno()), label=str(source_path)) != identity
                    or _path_identity(source_path) != identity):
                raise ValueError(f"Source snapshot changed while reading: {source_path}")
        identities[str(relative)] = identity
        if relative == PROTOCOL_REL:
            protocol_bytes = data
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest().upper(), identities, protocol_bytes


def _preflight_fingerprint(check: Preflight) -> str:
    """Detect mutation of the issued record, including its nested mappings."""
    payload = {
        "paths": {name: str(getattr(check, name)) for name in (
            "repo_root", "source_root", "data_root", "output", "topology", "test_x", "test_y", "window_map"
        )},
        "checkpoints": {str(k): str(v) for k, v in sorted(check.checkpoints.items())},
        "manifests": {str(k): str(v) for k, v in sorted(check.manifests.items())},
        "edges": [list(edge) for edge in check.topology_edges],
        "checkpoint_hashes": {str(k): v for k, v in sorted(check.checkpoint_hashes.items())},
        "checkpoint_identities": check.checkpoint_identities,
        "manifest_identities": check.manifest_identities,
        "test_identities": check.test_identities,
        "topology_identity": check.topology_identity,
        "metadata_identity": check.metadata_identity,
        "protocol_identity": check.protocol_identity,
        "protocol_sha256": check.protocol_sha256,
        "topology_snapshot_sha256": _sha256_bytes(check.topology_snapshot),
        "source_snapshot_sha256": check.source_snapshot_sha256,
        "approved_source_manifest_sha256": check.approved_manifest_sha256,
        "source_snapshot_identities": check.source_snapshot_identities,
        "approved_manifest_sha256": check.approved_manifest_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _issue_preflight(check: Preflight) -> Preflight:
    _ISSUED_PREFLIGHTS[check] = _preflight_fingerprint(check)
    return check


def _build_provenance(check: Preflight) -> dict[str, object]:
    return {
        "invocation": [sys.executable, *sys.argv],
        "protocol_id": PROTOCOL_ID,
        "dataset_protocol_id": DATA_PROTOCOL_ID,
        "source_snapshot_sha256": check.source_snapshot_sha256,
        "source_snapshot_scope": [
            "phase9_geant/evaluate_bu_v2_final.py",
            "phase9_geant/parse_geant.py",
            "src/gnn/model.py",
            str(PROTOCOL_REL),
        ],
        "source_snapshot_file_identities": dict(check.source_snapshot_identities),
        "source_snapshot_is_runtime_code_identity": False,
        "input_paths": {
            "X_test.npy": str(check.test_x),
            "y_test.npy": str(check.test_y),
            "window_starts.json": str(check.window_map),
        },
        "source_bundle_root": str(check.source_root),
        "input_file_identities_at_preflight": dict(check.test_identities),
        "dataset_metadata": {
            "path": str(check.repo_root / METADATA_REL),
            "file_identity": dict(check.metadata_identity),
            "scope": "protocol ID and chronological split bounds only",
        },
        "checkpoints": {
            str(seed): {
                "path": str(check.checkpoints[seed]),
                "sha256": check.checkpoint_hashes[seed],
                "file_identity": dict(check.checkpoint_identities[seed]),
                "manifest_path": str(check.manifests[seed]),
                "manifest_file_identity": dict(check.manifest_identities[seed]),
            }
            for seed in sorted(check.checkpoints)
        },
        "topology": {
            "path": str(check.topology),
            "sha256": TOPOLOGY_SHA256,
            "file_identity": dict(check.topology_identity),
            "provenance": "unverified upstream; hash identifies only this local file",
        },
        "model": {"name": "EdgeQoSForecaster", "seeds": [42, 123, 2026], "history": HISTORY, "horizon": 1,
                  "hidden_channels": 16, "gru_hidden_size": 16},
        "normalization": {"mean": SCALER_MEAN, "scale": SCALER_SCALE, "fit_scope": "training only"},
        "split": {name: list(bounds) for name, bounds in SPLITS.items()},
        "baselines": {
            "persistence": "last history observation",
            "ewma_alpha": EWMA_ALPHA,
            "ar5_coefficients_oldest_to_newest": list(AR_COEFFICIENTS),
            "ar5_intercept": AR_INTERCEPT,
        },
        "metrics": {"names": ["MAE", "RMSE"], "units": "computed-utilization-ratio", "accumulation_dtype": "float64"},
    }


def _validate_dataset_metadata(path: Path, repo_root: Path) -> Mapping[str, int]:
    identity = _path_identity(path)
    with _open_pinned_file(path, identity, repo_root) as handle:
        metadata = json.loads(handle.read().decode("utf-8"))
        if (_stat_identity(os.fstat(handle.fileno()), label=str(path)) != identity
                or _path_identity(path) != identity):
            raise ValueError("GEANT v2 metadata identity changed while reading")
    expected_splits = {name: list(bounds) for name, bounds in SPLITS.items()}
    if metadata.get("protocol_id") != DATA_PROTOCOL_ID or metadata.get("split_bounds") != expected_splits:
        raise ValueError("GEANT v2 metadata protocol or split bounds mismatch")
    return identity


def _validate_run_manifest(
    path: Path, seed: int, expected_edges: Sequence[tuple[str, str]], repo_root: Path,
    identity: Mapping[str, int],
) -> None:
    with _open_pinned_file(path, identity, repo_root) as handle:
        manifest = json.loads(handle.read().decode("utf-8"))
        if (_stat_identity(os.fstat(handle.fileno()), label=str(path)) != dict(identity)
                or _path_identity(path) != dict(identity)):
            raise ValueError(f"Checkpoint manifest identity changed while reading: {path}")
    expected_model_settings = {
        "hidden_channels": 16,
        "gru_hidden_size": 16,
    }
    if (
        manifest.get("status") != "complete"
        or manifest.get("mode") != "full"
        or manifest.get("seed") != seed
        or manifest.get("protocol_id") != DATA_PROTOCOL_ID
        or manifest.get("candidate") != "B-U"
        or manifest.get("model") != "EdgeQoSForecaster (unchanged)"
        or manifest.get("history_length") != HISTORY
        or manifest.get("prediction_horizon") != 1
    ):
        raise ValueError(f"Checkpoint manifest identity/protocol mismatch: {path}")
    settings = manifest.get("model_settings")
    if not isinstance(settings, dict) or any(settings.get(k) != v for k, v in expected_model_settings.items()):
        raise ValueError(f"Checkpoint architecture settings mismatch: {path}")
    normalization = manifest.get("normalization", {})
    if (
        normalization.get("mean") != SCALER_MEAN
        or normalization.get("scale") != SCALER_SCALE
        or normalization.get("fit_scope") != "all scalar values in X_train.npy and y_train.npy only"
        or normalization.get("accumulation_dtype") != "float64"
    ):
        raise ValueError(f"Checkpoint normalization provenance mismatch: {path}")
    expected_inputs = [
        str(repo_root / V2_REL / name)
        for name in ("X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy")
    ]
    if manifest.get("array_paths") != expected_inputs:
        raise ValueError(f"Checkpoint manifest input paths are not the approved train/validation paths: {path}")
    edge_rows = manifest.get("directed_edge_order")
    expected_rows = [
        {"index": i, "source": u, "destination": v}
        for i, (u, v) in enumerate(expected_edges)
    ]
    if edge_rows != expected_rows:
        raise ValueError(f"Checkpoint directed-edge ordering mismatch: {path}")


def preflight(
    repo_root: Path | None = None,
    *,
    source_root: Path | None = None,
    approved_manifest_sha256: str | None = None,
    topology_parser: Callable[[str], tuple[dict, list]] | None = None,
    hash_content: Callable[[bytes], str] = _sha256_bytes,
    reparse_tag_reader: Callable[[Path], int | None] = _component_reparse_tag,
) -> Preflight:
    """Validate non-test inputs and paths without opening test observations."""
    repo = Path(repo_root or CANONICAL_REPO_ROOT).absolute()
    approved_root = Path(CANONICAL_REPO_ROOT).absolute()
    if _canonical_case(repo) != _canonical_case(approved_root):
        raise ValueError(f"Repository root is not the approved canonical checkout: {repo}")
    repo = validate_path_components(
        repo, repo_root=repo, must_exist=True, reparse_tag_reader=reparse_tag_reader
    )
    source = Path(source_root or repo).absolute()
    validate_path_components(source, repo_root=source, must_exist=True, reparse_tag_reader=reparse_tag_reader)
    if source_root is not None:
        if not approved_manifest_sha256:
            raise ValueError("Frozen source root requires an independently approved manifest digest")
        from phase9_geant.source_freeze import verify_source_bundle

        verify_source_bundle(source, approved_manifest_sha256)
    v2 = repo / V2_REL
    validate_path_components(v2, repo_root=repo, must_exist=True, reparse_tag_reader=reparse_tag_reader)
    source_snapshot_sha256, source_snapshot_identities, protocol_bytes = _capture_source_snapshot(source)
    protocol = _require_exact_path(source / PROTOCOL_REL, PROTOCOL_REL, source)
    protocol_identity = source_snapshot_identities[str(PROTOCOL_REL)]
    protocol_sha256 = _sha256_bytes(protocol_bytes)
    _validate_protocol_text(protocol_bytes)

    output = _require_exact_path(repo / OUTPUT_REL, OUTPUT_REL, repo)
    validate_path_components(output, repo_root=repo, must_exist=False, reparse_tag_reader=reparse_tag_reader)
    if os.path.lexists(output):
        raise FileExistsError(f"Output or prior-attempt state already exists; no retry is allowed: {output}")

    metadata_path = _require_exact_path(repo / METADATA_REL, METADATA_REL, repo)
    validate_path_components(metadata_path, repo_root=repo, must_exist=True, reparse_tag_reader=reparse_tag_reader)
    if not metadata_path.is_file():
        raise ValueError("GEANT v2 metadata is not a regular file")
    metadata_identity = _validate_dataset_metadata(metadata_path, repo)

    topology = _require_exact_path(repo / TOPOLOGY_REL, TOPOLOGY_REL, repo)
    validate_path_components(topology, repo_root=repo, must_exist=True, reparse_tag_reader=reparse_tag_reader)
    if not topology.is_file():
        raise ValueError(f"Topology is not a regular file: {topology}")
    topology_identity = _path_identity(topology)
    with _open_pinned_file(topology, topology_identity, repo) as handle:
        topology_bytes = handle.read()
        if (_stat_identity(os.fstat(handle.fileno()), label=str(topology)) != topology_identity
                or _path_identity(topology) != topology_identity):
            raise ValueError("Topology file identity changed while reading")
    topology_hash = hash_content(topology_bytes)
    if topology_hash.upper() != TOPOLOGY_SHA256:
        raise ValueError(f"Topology SHA-256 mismatch: {topology_hash}")
    if topology_parser is None:
        from phase9_geant.parse_geant import parse_topology_text

        topology_parser = parse_topology_text
    nodes, links = topology_parser(topology_bytes.decode("utf-8"))
    edges = validate_topology(nodes, links)

    approved_files: dict[str, Path] = {}
    test_identities: dict[str, Mapping[str, int]] = {}
    for relative in (TEST_X_REL, TEST_Y_REL, WINDOWS_REL):
        expected = _require_exact_path(repo / relative, relative, repo)
        validate_path_components(expected, repo_root=repo, must_exist=True, reparse_tag_reader=reparse_tag_reader)
        if not expected.is_file():
            raise ValueError(f"Approved input is not a regular file: {expected}")
        identity = _path_identity(expected)
        tag = reparse_tag_reader(expected)
        _validate_cloud_identity(tag, identity, expected)
        test_identities[relative.name] = identity
        approved_files[relative.name] = expected

    checkpoint_paths: dict[int, Path] = {}
    manifest_paths: dict[int, Path] = {}
    checkpoint_hashes: dict[int, str] = {}
    checkpoint_identities: dict[int, Mapping[str, int]] = {}
    manifest_identities: dict[int, Mapping[str, int]] = {}
    for seed, (relative, expected_hash) in CHECKPOINTS.items():
        checkpoint = _require_exact_path(repo / relative, relative, repo)
        manifest = checkpoint.parent / "run_metadata.json"
        validate_path_components(checkpoint, repo_root=repo, must_exist=True, reparse_tag_reader=reparse_tag_reader)
        validate_path_components(manifest, repo_root=repo, must_exist=True, reparse_tag_reader=reparse_tag_reader)
        if not checkpoint.is_file() or not manifest.is_file():
            raise ValueError(f"Checkpoint or manifest is not a regular file for seed {seed}")
        checkpoint_identity = _path_identity(checkpoint)
        manifest_identity = _path_identity(manifest)
        with _open_pinned_file(checkpoint, checkpoint_identity, repo) as handle:
            checkpoint_bytes = handle.read()
            if _stat_identity(os.fstat(handle.fileno()), label=str(checkpoint)) != checkpoint_identity:
                raise ValueError(f"Checkpoint identity changed while reading: {checkpoint}")
        actual_hash = hash_content(checkpoint_bytes).upper()
        if actual_hash != expected_hash.upper():
            raise ValueError(f"Checkpoint SHA-256 mismatch for seed {seed}: {actual_hash}")
        _validate_run_manifest(manifest, seed, edges, repo, manifest_identity)
        checkpoint_paths[seed] = checkpoint
        manifest_paths[seed] = manifest
        checkpoint_hashes[seed] = actual_hash
        checkpoint_identities[seed] = checkpoint_identity
        manifest_identities[seed] = manifest_identity

    return _issue_preflight(Preflight(
        repo_root=repo,
        source_root=source,
        approved_manifest_sha256=approved_manifest_sha256,
        data_root=v2,
        output=output,
        topology=topology,
        test_x=approved_files["X_test.npy"],
        test_y=approved_files["y_test.npy"],
        window_map=approved_files["window_starts.json"],
        checkpoints=checkpoint_paths,
        manifests=manifest_paths,
        topology_edges=edges,
        checkpoint_hashes=checkpoint_hashes,
        checkpoint_identities=checkpoint_identities,
        manifest_identities=manifest_identities,
        test_identities=test_identities,
        topology_identity=topology_identity,
        metadata_identity=metadata_identity,
        protocol_identity=protocol_identity,
        protocol_sha256=protocol_sha256,
        topology_snapshot=topology_bytes,
        source_snapshot_sha256=source_snapshot_sha256,
        source_snapshot_identities=source_snapshot_identities,
    ))


def _validate_preflight_for_execution(check: Preflight) -> None:
    """Revalidate an issued preflight without reading test-file contents."""
    if not isinstance(check, Preflight):
        raise TypeError("Execution requires a Preflight produced by this module")
    issued_fingerprint = _ISSUED_PREFLIGHTS.get(check)
    if issued_fingerprint is None or issued_fingerprint != _preflight_fingerprint(check):
        raise ValueError("Preflight is forged, unissued, or was modified after validation")

    repo = Path(CANONICAL_REPO_ROOT).absolute()
    source = Path(check.source_root).absolute()
    if _canonical_case(check.repo_root) != _canonical_case(repo):
        raise ValueError("Preflight repository root is not the canonical checkout")
    validate_path_components(repo, repo_root=repo, must_exist=True)
    validate_path_components(source, repo_root=source, must_exist=True)
    if check.approved_manifest_sha256 is not None:
        from phase9_geant.source_freeze import verify_source_bundle

        verify_source_bundle(source, check.approved_manifest_sha256)
    elif _canonical_case(source) != _canonical_case(repo):
        raise ValueError("A separate source root requires its approved freeze manifest")
    for relative, identity in check.source_snapshot_identities.items():
        source_path = source / Path(relative)
        validate_path_components(source_path, repo_root=source, must_exist=True)
        if _path_identity(source_path) != dict(identity):
            raise ValueError(f"Source file identity changed after preflight: {source_path}")
    expected_paths = {
        "data_root": V2_REL,
        "output": OUTPUT_REL,
        "topology": TOPOLOGY_REL,
        "test_x": TEST_X_REL,
        "test_y": TEST_Y_REL,
        "window_map": WINDOWS_REL,
    }
    for field_name, relative in expected_paths.items():
        candidate = Path(getattr(check, field_name))
        _require_exact_path(candidate, relative, repo)
        validate_path_components(candidate, repo_root=repo, must_exist=field_name != "output")
    if os.path.lexists(check.output):
        raise FileExistsError(f"Output or prior-attempt state already exists: {check.output}")

    if set(check.checkpoints) != set(CHECKPOINTS) or set(check.manifests) != set(CHECKPOINTS):
        raise ValueError("Preflight checkpoint set differs from the frozen allow-list")
    if set(check.checkpoint_hashes) != set(CHECKPOINTS):
        raise ValueError("Preflight checkpoint hashes differ from the frozen allow-list")
    if set(check.test_identities) != {TEST_X_REL.name, TEST_Y_REL.name, WINDOWS_REL.name}:
        raise ValueError("Preflight test-input identity set differs from the allow-list")

    protocol = source / PROTOCOL_REL
    _require_exact_path(protocol, PROTOCOL_REL, source)
    if _path_identity(protocol) != dict(check.protocol_identity):
        raise ValueError("Frozen protocol identity changed after preflight")
    with _open_pinned_file(protocol, check.protocol_identity, source) as handle:
        protocol_bytes = handle.read()
        if _path_identity(protocol) != dict(check.protocol_identity):
            raise ValueError("Frozen protocol changed while being revalidated")
    if _sha256_bytes(protocol_bytes) != check.protocol_sha256:
        raise ValueError("Frozen protocol bytes differ from the preflight snapshot")
    _validate_protocol_text(protocol_bytes)

    metadata_identity = _validate_dataset_metadata(repo / METADATA_REL, repo)
    if dict(metadata_identity) != dict(check.metadata_identity):
        raise ValueError("GEANT metadata identity changed after preflight")

    topology = repo / TOPOLOGY_REL
    if _path_identity(topology) != dict(check.topology_identity):
        raise ValueError("Topology identity changed after preflight")
    with _open_pinned_file(topology, check.topology_identity, repo) as handle:
        topology_bytes = handle.read()
        if _path_identity(topology) != dict(check.topology_identity):
            raise ValueError("Topology changed while being revalidated")
    if topology_bytes != check.topology_snapshot:
        raise ValueError("Topology snapshot differs from the issued preflight")
    if len(check.topology_edges) != LINKS or len({node for edge in check.topology_edges for node in edge}) != NODES:
        raise ValueError("Issued preflight topology structure is invalid")
    if any(check.topology_edges[i + 1] != check.topology_edges[i][::-1] for i in range(0, LINKS, 2)):
        raise ValueError("Issued preflight topology edge order is invalid")

    for relative in (TEST_X_REL, TEST_Y_REL, WINDOWS_REL):
        path = repo / relative
        if _path_identity(path) != dict(check.test_identities[relative.name]):
            raise ValueError(f"Approved test path identity changed after preflight: {path}")

    for seed, (relative, expected_hash) in CHECKPOINTS.items():
        checkpoint = repo / relative
        manifest = checkpoint.parent / "run_metadata.json"
        if _canonical_case(check.checkpoints[seed]) != _canonical_case(checkpoint):
            raise ValueError(f"Checkpoint path is not the exact allow-listed path for seed {seed}")
        if _canonical_case(check.manifests[seed]) != _canonical_case(manifest):
            raise ValueError(f"Manifest path is not the expected path for seed {seed}")
        if check.checkpoint_hashes[seed].upper() != expected_hash.upper():
            raise ValueError(f"Preflight checkpoint hash differs from the frozen identity for seed {seed}")
        if _path_identity(checkpoint) != dict(check.checkpoint_identities[seed]):
            raise ValueError(f"Checkpoint identity changed after preflight for seed {seed}")
        if _path_identity(manifest) != dict(check.manifest_identities[seed]):
            raise ValueError(f"Checkpoint manifest identity changed after preflight for seed {seed}")
        _validate_run_manifest(manifest, seed, check.topology_edges, repo, check.manifest_identities[seed])


def _atomic_json(path: Path, payload: object) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Initial creation happens inside a newly exclusive output directory;
        # subsequent state changes are atomic sibling replacements.
        os.replace(temp_path, path)
    except BaseException:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _validate_test_inputs(x, y, starts: object):
    import numpy as np

    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("Test arrays must be NumPy arrays")
    if x.dtype != np.float32 or y.dtype != np.float32:
        raise ValueError("Test arrays must have the v2 float32 dtype")
    if x.ndim != 3 or x.shape[1:] != (HISTORY, LINKS):
        raise ValueError(f"X_test must have shape [N,{HISTORY},{LINKS}], got {x.shape}")
    if y.shape != (x.shape[0], LINKS) or x.shape[0] == 0:
        raise ValueError(f"y_test must have shape [N,{LINKS}] aligned to X_test, got {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Test arrays contain nonfinite values")
    if not isinstance(starts, list) or len(starts) != x.shape[0]:
        raise ValueError("Test window-start list length does not match test array rows")
    if any(type(start) is not int for start in starts):
        raise ValueError("Test window starts must be integer indices")
    if starts != sorted(set(starts)):
        raise ValueError("Test window starts must be unique and strictly chronological")
    split_start, split_end = SPLITS["test"]
    if any(start < split_start or start + HISTORY >= split_end for start in starts):
        raise ValueError("Test window is not fully contained in the frozen test split")
    targets = [start + HISTORY for start in starts]
    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate test target index")
    return x, y, tuple(starts), tuple(targets)


def _metrics(prediction, target) -> dict[str, float | int]:
    import numpy as np

    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    if prediction64.shape != target64.shape or not np.isfinite(prediction64).all():
        raise ValueError("Prediction shape/finite-value check failed")
    error = prediction64 - target64
    return {
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
        "scalar_pairs": int(error.size),
    }


def _baseline_predictions(x):
    import numpy as np

    persistence = x[:, -1, :].astype(np.float64, copy=True)
    raw_weights = np.asarray(
        [EWMA_ALPHA * (1 - EWMA_ALPHA) ** (HISTORY - 1 - i) for i in range(HISTORY)],
        dtype=np.float64,
    )
    weights = raw_weights / raw_weights.sum(dtype=np.float64)
    ewma = np.tensordot(x.astype(np.float64), weights, axes=([1], [0]))
    coefficients = np.asarray(AR_COEFFICIENTS, dtype=np.float64)
    ar5 = np.einsum("nte,t->ne", x.astype(np.float64), coefficients) + AR_INTERCEPT
    return {"Persistence": persistence, "EWMA alpha=0.8": ewma, "AR(5) legacy": ar5}


def _make_sequence(x_batch, edge_index, node_count: int, mean: float, scale: float):
    import numpy as np
    import torch
    from types import SimpleNamespace

    batch, history, edge_count = x_batch.shape
    offsets = torch.arange(batch, dtype=torch.long) * node_count
    batched_edges = torch.cat([edge_index + offset for offset in offsets], dim=1)
    node_x = torch.ones((batch * node_count, 1), dtype=torch.float32)
    result = []
    for timestep in range(history):
        standardized = (x_batch[:, timestep, :].astype(np.float64) - mean) / scale
        edge_attr = torch.as_tensor(standardized.astype(np.float32).reshape(batch * edge_count, 1))
        result.append(SimpleNamespace(x=node_x, edge_index=batched_edges, edge_attr=edge_attr, timestep=timestep))
    return result


def _load_frozen_models(preflight_result: Preflight):
    import torch
    import src
    import src.gnn
    import src.gnn.dataset
    import src.gnn.preprocessing
    from src.gnn.model import EdgeQoSForecaster
    import src.gnn.model as model_module

    source_root = preflight_result.source_root.resolve(strict=True)
    if tuple(Path(p).resolve(strict=True) for p in src.__path__) != ((source_root / "src").resolve(strict=True),):
        raise RuntimeError("Namespace package src resolved outside the verified source bundle")
    expected_modules = {
        "src.gnn": source_root / "src/gnn/__init__.py",
        "src.gnn.dataset": source_root / "src/gnn/dataset.py",
        "src.gnn.preprocessing": source_root / "src/gnn/preprocessing.py",
        "src.gnn.model": source_root / "src/gnn/model.py",
    }
    for name, expected in expected_modules.items():
        if Path(sys.modules[name].__file__).resolve(strict=True) != expected.resolve(strict=True):
            raise RuntimeError(f"First-party dependency {name} did not load from the verified source bundle")

    models = {}
    if set(preflight_result.checkpoints) != set(CHECKPOINTS):
        raise ValueError("Frozen checkpoint set does not match the protocol allow-list")
    for seed, checkpoint in preflight_result.checkpoints.items():
        # Recheck approved path and identity immediately before deserialization.
        _require_exact_path(checkpoint, CHECKPOINTS[seed][0], preflight_result.repo_root)
        checkpoint_bytes = _verified_checkpoint_snapshot(
            checkpoint, preflight_result.checkpoint_identities[seed],
            preflight_result.checkpoint_hashes[seed], preflight_result.repo_root,
        )
        model = EdgeQoSForecaster(edge_feature_count=1, target_count=1, hidden_channels=16, gru_hidden_size=16)
        state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        models[seed] = model
    return models


def _verified_checkpoint_snapshot(
    path: Path, identity: Mapping[str, int], expected_hash: str, repo_root: Path,
    *, hash_content: Callable[[bytes], str] = _sha256_bytes,
) -> bytes:
    with _open_pinned_file(path, identity, repo_root) as handle:
        content = handle.read()
        if hash_content(content).upper() != expected_hash.upper():
            raise ValueError(f"Checkpoint changed after preflight: {path}")
        if _stat_identity(os.fstat(handle.fileno()), label=str(path)) != dict(identity):
            raise ValueError(f"Checkpoint identity changed while being read: {path}")
    # The caller deserializes this exact immutable snapshot, not the pathname.
    return content


def _predict_loaded_models(preflight_result: Preflight, models, x, *, batch_size: int = 32):
    import numpy as np
    import torch
    node_ids = sorted({node for edge in preflight_result.topology_edges for node in edge})
    node_index = {node: i for i, node in enumerate(node_ids)}
    edge_index = torch.tensor(
        [[node_index[u] for u, _ in preflight_result.topology_edges],
         [node_index[v] for _, v in preflight_result.topology_edges]],
        dtype=torch.long,
    )
    predictions: dict[int, np.ndarray] = {}
    for seed, model in models.items():
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for begin in range(0, len(x), batch_size):
                block = x[begin:begin + batch_size]
                sequence = _make_sequence(block, edge_index, len(node_ids), SCALER_MEAN, SCALER_SCALE)
                output = model(sequence).reshape(len(block), LINKS, 1).squeeze(-1)
                if not torch.isfinite(output).all():
                    raise ValueError(f"Model produced nonfinite output for seed {seed}")
                raw = output.cpu().numpy().astype(np.float64) * SCALER_SCALE + SCALER_MEAN
                rows.append(raw)
        predictions[seed] = np.concatenate(rows, axis=0)
    return predictions


def _write_state(output: Path, state: dict[str, object]) -> None:
    _atomic_json(output / "run_state.json", state)


def execute_authorized(
    preflight_result: Preflight,
    *,
    authorization: object | None = None,
    **_kwargs,
) -> Path:
    """Reject direct API execution; the supported entrypoint is ``main``.

    The legacy token parameter is intentionally ignored. This is not a security
    boundary against arbitrary code in the same process or filesystem owner.
    """
    raise PermissionError("Direct evaluator calls are unsupported; authorize through the CLI entrypoint")


def _execute_after_cli_authorization(
    preflight_result: Preflight,
    *,
    model_loader: Callable[[Preflight], object] | None = None,
    test_loader: Callable[[Preflight], tuple[object, object, object, object]] | None = None,
    prediction_fn: Callable[[Preflight, object, object], Mapping[int, object]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Path:
    """Internal CLI path; still subject to the documented same-process limit."""
    if not _CLI_AUTHORIZATION_CONTEXT.get():
        raise PermissionError("Internal evaluator path requires the CLI authorization context")
    _validate_preflight_for_execution(preflight_result)
    if not SOURCE_FREEZE_REVIEWED:
        raise RuntimeError(
            "Final evaluation is blocked: no independently reviewed freeze binds imported code to approved source bytes"
        )
    output = preflight_result.output
    # A second exact path check, then exclusive creation: existing or raced-in
    # destinations are never reopened, resumed, or overwritten.
    _require_exact_path(output, OUTPUT_REL, preflight_result.repo_root)
    validate_path_components(output, repo_root=preflight_result.repo_root, must_exist=False)
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing prior/ambiguous attempt: {output}")
    output.mkdir(parents=False, exist_ok=False)
    state: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "status": "authorization_consumed",
        "test_access_state": "not_started",
        "attempt_id": str(uuid.uuid4()),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authorization": "explicit --authorize-final-test-evaluation flag consumed; no resume/retry",
        "output": str(output),
        "provenance": _build_provenance(preflight_result),
    }
    _atomic_json(output / "run_state.json", state)
    started = clock()
    try:
        # Checkpoint deserialization is permitted, non-test access. Do it after
        # consuming authorization but before opening any test artifact.
        models = (model_loader or _load_frozen_models)(preflight_result)
        # Record uncertainty durably BEFORE any test array or window map read.
        state.update(status="test_access_started", test_access_state="may_have_started")
        _write_state(output, state)
        if test_loader is None:
            test_loader = _load_test_inputs
        loaded = test_loader(preflight_result)
        if not isinstance(loaded, tuple) or len(loaded) != 4:
            raise ValueError("Test loader must return arrays, window starts, and opened-file identities")
        x, y, starts, observed_identities = loaded
        if set(observed_identities) != {"X_test.npy", "y_test.npy", "window_starts.json"}:
            raise ValueError("Test loader did not provide identity evidence for every approved input")
        state["provenance"]["opened_test_input_identities"] = observed_identities
        _write_state(output, state)
        x, y, starts, target_indices = _validate_test_inputs(x, y, starts)
        predictor = prediction_fn or _predict_loaded_models
        neural = predictor(preflight_result, models, x)
        baselines = _baseline_predictions(x)
        all_predictions: dict[str, object] = {
            **{f"B-U seed {seed}": neural[seed] for seed in (42, 123, 2026)},
            **baselines,
        }
        if set(neural) != {42, 123, 2026}:
            raise ValueError("Prediction set must contain exactly the three frozen B-U seeds")
        metrics = {name: _metrics(prediction, y) for name, prediction in all_predictions.items()}
        seed_mae = [metrics[f"B-U seed {seed}"]["mae"] for seed in (42, 123, 2026)]
        seed_rmse = [metrics[f"B-U seed {seed}"]["rmse"] for seed in (42, 123, 2026)]
        import numpy as np

        results = {
            "protocol_id": PROTOCOL_ID,
            "dataset_protocol_id": DATA_PROTOCOL_ID,
            "status": "metrics_written; run_state.json is the completion authority",
            "split": {"bounds": list(SPLITS["test"]), "start_indices": list(starts), "target_indices": list(target_indices)},
            "history_length": HISTORY,
            "prediction_horizon": 1,
            "directed_links": LINKS,
            "scalar_pairs": int(y.size),
            "dtype": str(y.dtype),
            "coverage": {
                "windows": int(len(starts)),
                "target_indices": list(target_indices),
                "finite_input_scalars": int(x.size),
                "finite_target_scalars": int(y.size),
                "finite_prediction_scalars_by_method": {
                    name: int(np.asarray(prediction).size)
                    for name, prediction in all_predictions.items()
                },
                "invalid_input_target_or_prediction_scalars": 0,
            },
            "metrics": metrics,
            "bu_seed_mean_sample_std": {
                "mae_mean": float(np.mean(seed_mae, dtype=np.float64)),
                "mae_std_ddof1": float(np.std(seed_mae, ddof=1, dtype=np.float64)),
                "rmse_mean": float(np.mean(seed_rmse, dtype=np.float64)),
                "rmse_std_ddof1": float(np.std(seed_rmse, ddof=1, dtype=np.float64)),
            },
            "checkpoint_hashes": dict(preflight_result.checkpoint_hashes),
            "topology_sha256": TOPOLOGY_SHA256,
            "topology_provenance": "unverified upstream; hash is local identity only",
            "metric_units": "computed-utilization-ratio",
            "metric_formula": "float64 MAE and RMSE over identical finite N x 72 targets",
            "runtime_seconds": clock() - started,
            "provenance": state["provenance"],
            "attempt_id": state["attempt_id"],
        }
        _atomic_json(output / "results.json", results)
        state.update(status="complete", test_access_state="completed", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        _write_state(output, state)
        return output
    except BaseException as exc:
        failed = dict(state)
        failed.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            test_access_state=state.get("test_access_state", "not_started"),
        )
        try:
            _write_state(output, failed)
        except BaseException:
            # Preserve the original error and the last durable state. Output
            # existence makes any later attempt fail closed.
            pass
        raise


def _load_test_inputs(check: Preflight):
    import numpy as np

    # These are the first reads of any test observation/window content, and
    # the CLI path has durably recorded test_access_state=may_have_started.
    paths = {
        "X_test.npy": (check.test_x, TEST_X_REL),
        "y_test.npy": (check.test_y, TEST_Y_REL),
        "window_starts.json": (check.window_map, WINDOWS_REL),
    }
    handles = {}
    try:
        for name, (path, relative) in paths.items():
            _require_exact_path(path, relative, check.repo_root)
            handles[name] = _open_pinned_file(path, check.test_identities[name], check.repo_root)
        x = np.load(handles["X_test.npy"], allow_pickle=False)
        y = np.load(handles["y_test.npy"], allow_pickle=False)
        mapping = json.loads(handles["window_starts.json"].read().decode("utf-8"))
        observed_identities = {}
        for name, handle in handles.items():
            opened = _stat_identity(os.fstat(handle.fileno()), label=name)
            path = paths[name][0]
            if opened != dict(check.test_identities[name]) or _path_identity(path) != dict(check.test_identities[name]):
                raise ValueError(f"Approved test input changed while being consumed: {path}")
            observed_identities[name] = opened
    finally:
        for handle in handles.values():
            handle.close()
    if not isinstance(mapping, dict) or "test" not in mapping:
        raise ValueError("window_starts.json lacks the test mapping")
    return x, y, mapping["test"], observed_identities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight-only", action="store_true", help="Validate non-test inputs; never read test observations")
    group.add_argument(
        "--authorize-final-test-evaluation",
        action="store_true",
        help="Explicitly authorize and consume the single final-test evaluation attempt",
    )
    parser.add_argument("--source-bundle-root", help="Independently reviewed immutable source bundle")
    parser.add_argument("--approved-manifest-sha256", help="Manifest digest from the separate review record")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    frozen_source_root: Path | None = None,
    approved_manifest_sha256: str | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    cli_source_root = Path(args.source_bundle_root) if args.source_bundle_root else None
    source_root = Path(frozen_source_root or cli_source_root).absolute() if (frozen_source_root or cli_source_root) else None
    manifest_digest = approved_manifest_sha256 or args.approved_manifest_sha256
    if bool(source_root) != bool(manifest_digest):
        raise ValueError("Frozen source root and independently approved manifest digest must be supplied together")
    if args.authorize_final_test_evaluation and source_root is None:
        raise PermissionError("Authorized execution requires the reviewed source bundle")
    if source_root is not None:
        import phase9_geant
        import phase9_geant.parse_geant as parser_module
        import phase9_geant.source_freeze as freeze_module
        from phase9_geant.source_freeze import verify_source_bundle

        verify_source_bundle(source_root, manifest_digest)
        source_root = source_root.resolve(strict=True)
        expected_evaluator = (source_root / "phase9_geant/evaluate_bu_v2_final.py").resolve(strict=True)
        if Path(__file__).resolve(strict=True) != expected_evaluator:
            raise RuntimeError("Evaluator was not imported from the verified source bundle")
        if Path(phase9_geant.__file__).resolve(strict=True) != (source_root / "phase9_geant/__init__.py").resolve(strict=True):
            raise RuntimeError("Phase 9 package initializer was not imported from the verified bundle")
        if Path(parser_module.__file__).resolve(strict=True) != (source_root / "phase9_geant/parse_geant.py").resolve(strict=True):
            raise RuntimeError("Topology parser was not imported from the verified source bundle")
        if Path(freeze_module.__file__).resolve(strict=True) != (source_root / "phase9_geant/source_freeze.py").resolve(strict=True):
            raise RuntimeError("Source-freeze verifier was not imported from the verified bundle")
    checked = preflight(source_root=source_root, approved_manifest_sha256=manifest_digest)
    if not args.authorize_final_test_evaluation:
        print(json.dumps({
            "status": "preflight_passed_evaluation_blocked",
            "evaluation_ready": False,
            "test_access": "none",
            "output": str(checked.output),
        }, indent=2))
        return 0
    authorization_context = _CLI_AUTHORIZATION_CONTEXT.set(True)
    try:
        output = _execute_after_cli_authorization(checked)
    finally:
        _CLI_AUTHORIZATION_CONTEXT.reset(authorization_context)
    print(f"Final evaluation completed; results: {output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
