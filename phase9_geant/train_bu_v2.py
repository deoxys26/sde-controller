"""Opt-in B-U training path for GEANT validity-v2 train/validation windows.

Importing this module performs no I/O. The CLI requires an explicit run mode;
the run path loads only the four named train/validation arrays below. The new
provenance bindings apply only to runs created by this workflow; historical
B-U manifests are not retrofitted and cannot qualify through this metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from phase9_geant.parse_geant import parse_topology
from src.gnn.dataset import GraphSnapshot
from src.gnn.model import EdgeQoSForecaster
from phase9_geant.window_provenance_v2 import (
    PROVENANCE_FILENAME,
    array_fingerprint,
    canonical_json_sha256,
    validate_train_validation_provenance,
)
from phase9_geant.safe_paths import (
    canonical_repository_root,
    checked_absolute,
    require_regular_file,
    validate_fresh_child,
    validate_v2_root,
    verify_new_directory_containment,
)
from phase9_geant.split_access import load_permitted_split_array


PROTOCOL_ID = "geant-validity-v2-train-median"
HISTORY_LENGTH = 5
PREDICTION_HORIZON = 1
EXPECTED_NODES = 22
EXPECTED_UNDIRECTED_LINKS = 36
EXPECTED_DIRECTED_LINKS = 72
SEED = 42
TRAINING = {
    "epochs": 5,
    "early_stopping_patience": 2,
    "learning_rate": 0.005,
    "batch_size": 32,
    "optimizer": "Adam",
    "loss": "SmoothL1Loss",
    "hidden_channels": 16,
    "gru_hidden_size": 16,
    "initialization": "from_scratch",
}
ARRAY_NAMES = (
    "X_train.npy",
    "y_train.npy",
    "X_validation.npy",
    "y_validation.npy",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V2_DATA_ROOT = REPOSITORY_ROOT / "data" / "processed" / "geant_validity_v2"
V2_OUTPUT_ROOT = V2_DATA_ROOT / "neural_bu_v2"
TOPOLOGY_PATH = REPOSITORY_ROOT / "data" / "raw" / "geant_topology" / "geant.txt"
RUN_IDENTITY_SOURCE_FILES = (
    "phase9_geant/train_bu_v2.py",
    "phase9_geant/window_provenance_v2.py",
    "phase9_geant/validity_v2.py",
    "phase9_geant/build_validity_v2.py",
    "phase9_geant/parse_geant.py",
    "phase9_geant/safe_paths.py",
    "phase9_geant/split_access.py",
    "src/gnn/dataset.py",
    "src/gnn/model.py",
)


@dataclass(frozen=True)
class WindowArrays:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray


@dataclass(frozen=True)
class Normalization:
    mean: float
    scale: float
    fit_scope: str = "all scalar values in X_train.npy and y_train.npy only"


def validate_output_dir(output_dir: Path, data_root: Path = V2_DATA_ROOT) -> Path:
    """Check path components before resolving and require a direct v2 child.

    The full canonical repository-root contract is rechecked by
    ``validate_fresh_child`` immediately before creation.
    """
    root_arg = checked_absolute(Path(data_root), allow_missing=True)
    output_arg = checked_absolute(Path(output_dir), allow_missing=True)
    if os.path.normcase(os.fspath(output_arg)) == os.path.normcase(os.fspath(root_arg)):
        raise ValueError("B-U output must be a dedicated child directory")
    expected = root_arg / output_arg.name
    if os.path.normcase(os.fspath(output_arg)) != os.path.normcase(os.fspath(expected)):
        raise ValueError("B-U output must be a direct child of the v2 data root")
    root_resolved = root_arg.resolve()
    output = output_arg.resolve()
    try:
        output.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Output path must be inside {root_resolved}: {output}") from exc
    if output == root_resolved:
        raise ValueError("B-U output must be a dedicated child directory")
    legacy_roots = {
        (root_resolved.parent / "geant").resolve(),
        (REPOSITORY_ROOT / "data" / "processed" / "geant").resolve(),
    }
    if any(output == legacy or legacy in output.parents for legacy in legacy_roots):
        raise ValueError("Output path overlaps the legacy processed-data tree")
    return output


def validate_window_arrays(arrays: WindowArrays, *, expected_links: int = EXPECTED_DIRECTED_LINKS) -> None:
    named = (
        ("X_train", arrays.x_train, 3),
        ("y_train", arrays.y_train, 2),
        ("X_validation", arrays.x_validation, 3),
        ("y_validation", arrays.y_validation, 2),
    )
    for name, value, dimensions in named:
        if not isinstance(value, np.ndarray) or value.ndim != dimensions:
            raise ValueError(f"{name} must be a {dimensions}-dimensional NumPy array")
        if value.dtype != np.dtype(np.float32):
            raise ValueError(f"{name} must have v2 float32 dtype, got {value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinite values")

    for split, x, y in (
        ("train", arrays.x_train, arrays.y_train),
        ("validation", arrays.x_validation, arrays.y_validation),
    ):
        if x.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError(f"{split} arrays must contain at least one sample")
        if x.shape[1] != HISTORY_LENGTH:
            raise ValueError(f"{split} history must contain exactly {HISTORY_LENGTH} steps")
        if y.shape != (x.shape[0], x.shape[2]):
            raise ValueError(
                f"{split} targets must have shape {(x.shape[0], x.shape[2])}, got {y.shape}"
            )
        if x.shape[2] != expected_links:
            raise ValueError(f"{split} must contain {expected_links} directed links, got {x.shape[2]}")

    if arrays.x_train.shape[2] != arrays.x_validation.shape[2]:
        raise ValueError("Train and validation arrays have different directed-link counts")


def load_v2_windows(
    data_root: Path = V2_DATA_ROOT,
    *,
    array_loader: Callable = np.load,
    expected_links: int = EXPECTED_DIRECTED_LINKS,
    repository_root: Path = REPOSITORY_ROOT,
) -> WindowArrays:
    """Load exactly the four fixed v2 train/validation filenames, never a glob."""
    root = validate_v2_root(data_root, repository_root)
    loaded: dict[str, np.ndarray] = {}
    for filename in ARRAY_NAMES:
        loaded[filename] = load_permitted_split_array(
            root, filename, array_loader=array_loader
        )

    arrays = WindowArrays(
        x_train=loaded["X_train.npy"],
        y_train=loaded["y_train.npy"],
        x_validation=loaded["X_validation.npy"],
        y_validation=loaded["y_validation.npy"],
    )
    validate_window_arrays(arrays, expected_links=expected_links)
    return arrays


def fit_training_normalization(x_train: np.ndarray, y_train: np.ndarray) -> Normalization:
    """Fit one scalar utilization scaler from training arrays only, in float64."""
    if x_train.dtype != np.float32 or y_train.dtype != np.float32:
        raise ValueError("Training normalization expects v2 float32 arrays")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("Training normalization refuses nonfinite values")
    values = np.concatenate((x_train.reshape(-1), y_train.reshape(-1))).astype(np.float64)
    mean = float(values.mean(dtype=np.float64))
    scale = float(values.std(dtype=np.float64))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training utilization scale must be finite and positive")
    return Normalization(mean=mean, scale=scale)


def directed_edges_v2(undirected_links: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Match build_validity_v2.py: each topology link then its reverse."""
    edges: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for source, destination in undirected_links:
        if source == destination:
            raise ValueError("Topology contains a self-link")
        key = frozenset((source, destination))
        if key in seen:
            raise ValueError("Topology contains a duplicate undirected link")
        seen.add(key)
        edges.extend(((source, destination), (destination, source)))
    return tuple(edges)


def assert_edge_order(
    array_edges: Sequence[tuple[str, str]], model_edges: Sequence[tuple[str, str]]
) -> None:
    if tuple(array_edges) != tuple(model_edges):
        raise ValueError("Array directed-link ordering does not match model edge_index ordering")


def build_edge_index(
    nodes: Sequence[str], undirected_links: Sequence[tuple[str, str]]
) -> tuple[torch.Tensor, tuple[str, ...], tuple[tuple[str, str], ...]]:
    node_order = tuple(sorted(nodes))
    if len(node_order) != len(set(node_order)):
        raise ValueError("Topology node identifiers are not unique")
    node_index = {name: index for index, name in enumerate(node_order)}
    edges = directed_edges_v2(undirected_links)
    if any(source not in node_index or destination not in node_index for source, destination in edges):
        raise ValueError("A topology edge references a missing node")
    edge_index = torch.tensor(
        [[node_index[source] for source, _ in edges],
         [node_index[destination] for _, destination in edges]],
        dtype=torch.long,
    )
    return edge_index, node_order, edges


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(repository_root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    """Fingerprint the fixed first-party code/configuration inputs for a new run."""
    root = canonical_repository_root(repository_root)
    result: dict[str, str] = {}
    for relative in RUN_IDENTITY_SOURCE_FILES:
        path = root / relative
        resolved = require_regular_file(path, path.parent)
        result[relative] = _sha256_file(resolved)
    return result


def _topology_edge_order() -> tuple[tuple[str, str], ...]:
    topology_path = checked_absolute(TOPOLOGY_PATH)
    require_regular_file(topology_path, topology_path.parent)
    nodes, links = parse_topology(str(topology_path))
    if len(nodes) != EXPECTED_NODES or len(links) != EXPECTED_UNDIRECTED_LINKS:
        raise ValueError("Topology dimensions differ from the v2 GEANT topology contract")
    edge_index, node_order, edges = build_edge_index(tuple(nodes), links)
    model_edges = tuple(
        (node_order[int(source)], node_order[int(destination)])
        for source, destination in edge_index.T.tolist()
    )
    assert_edge_order(directed_edges_v2(links), model_edges)
    if len(edges) != EXPECTED_DIRECTED_LINKS:
        raise ValueError("Topology does not produce exactly 72 directed links")
    return edges


def load_window_provenance(
    data_root: Path, *, repository_root: Path = REPOSITORY_ROOT
) -> tuple[dict[str, object], str, dict[str, str]]:
    """Read and structurally validate only the train/validation provenance sidecar."""
    root = validate_v2_root(data_root, repository_root)
    resolved = require_regular_file(root / PROVENANCE_FILENAME, root)
    raw = resolved.read_bytes()
    provenance = json.loads(raw.decode("utf-8"))
    # Structural validation is possible before output creation; array fingerprints
    # are checked against the loaded train/validation arrays before training.
    validate_train_validation_provenance(provenance, {}, required_splits=())
    split_hashes = {
        name: canonical_json_sha256(provenance["splits"][name])
        for name in ("train", "validation")
    }
    return provenance, _sha256_bytes(raw), split_hashes


def _preflight_run_identity(
    *, mode: str, seed: int, run_id: str | None,
    provenance_file_sha256: str, provenance_hashes: dict[str, str],
    edges: Sequence[tuple[str, str]], source_sha256: dict[str, str],
) -> dict[str, object]:
    edge_pairs = [list(edge) for edge in edges]
    edge_order = [
        {"index": index, "source": source, "destination": destination}
        for index, (source, destination) in enumerate(edges)
    ]
    if (
        len(edge_pairs) != EXPECTED_DIRECTED_LINKS
        or any(len(edge) != 2 or any(not isinstance(node, str) or not node for node in edge) for edge in edge_pairs)
        or len(set(map(tuple, edge_pairs))) != len(edge_pairs)
    ):
        raise ValueError("Run edge identity is incomplete or contains duplicates")
    if set(provenance_hashes) != {"train", "validation"}:
        raise ValueError("Both train and validation provenance identities are required")
    if not _is_sha256(provenance_file_sha256) or any(not _is_sha256(value) for value in provenance_hashes.values()):
        raise ValueError("Window provenance hashes are missing or malformed")
    if not source_sha256 or any(not _is_sha256(value) for value in source_sha256.values()):
        raise ValueError("Run source hashes are missing or malformed")
    config = {
        "protocol_id": PROTOCOL_ID,
        "mode": mode,
        "seed": seed,
        "run_id": run_id,
        "training": dict(TRAINING),
        "history_length": HISTORY_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "directed_edge_order": edge_order,
    }
    return {
        "window_provenance_sha256": {
            "file": provenance_file_sha256,
            **provenance_hashes,
        },
        "directed_edge_order": edge_order,
        "edge_order_sha256": canonical_json_sha256(edge_pairs),
        "source_sha256": dict(source_sha256),
        "configuration_sha256": canonical_json_sha256(config),
    }


def build_array_provenance(
    arrays: WindowArrays,
    provenance: object,
    *,
    provenance_file_sha256: str,
    edges: Sequence[tuple[str, str]],
    preflight_identity: dict[str, object],
) -> dict[str, object]:
    """Bind loaded permitted arrays to the prevalidated window/edge identities."""
    validate_window_arrays(arrays)
    split_hashes = validate_train_validation_provenance(
        provenance,
        {
            "train": (arrays.x_train, arrays.y_train),
            "validation": (arrays.x_validation, arrays.y_validation),
        },
    )
    expected = preflight_identity["window_provenance_sha256"]
    if expected.get("file") != provenance_file_sha256 or any(
        expected.get(name) != split_hashes[name] for name in ("train", "validation")
    ):
        raise ValueError("Loaded window provenance changed after preflight")
    expected_edges = [
        {"index": index, "source": source, "destination": destination}
        for index, (source, destination) in enumerate(edges)
    ]
    if preflight_identity.get("directed_edge_order") != expected_edges:
        raise ValueError("Runtime directed-edge ordering differs from preflight identity")
    fingerprints = {
        "X_train.npy": array_fingerprint(arrays.x_train),
        "y_train.npy": array_fingerprint(arrays.y_train),
        "X_validation.npy": array_fingerprint(arrays.x_validation),
        "y_validation.npy": array_fingerprint(arrays.y_validation),
    }
    return {
        **preflight_identity,
        "array_fingerprints": fingerprints,
    }


def validate_required_run_metadata(
    metadata: object,
    *,
    expected_edges: Sequence[tuple[str, str]] | None = None,
    expected_fingerprints: dict[str, str] | None = None,
) -> None:
    required = {
        "window_provenance_sha256", "array_fingerprints", "directed_edge_order",
        "edge_order_sha256", "source_sha256", "configuration_sha256", "checkpoint_sha256",
    }
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError("B-U run metadata lacks required provenance or checkpoint identity")
    fingerprints = metadata.get("array_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != set(ARRAY_NAMES):
        raise ValueError("B-U run metadata has missing or unexpected array fingerprints")
    provenance = metadata.get("window_provenance_sha256")
    if not isinstance(provenance, dict) or set(provenance) != {"file", "train", "validation"}:
        raise ValueError("B-U run metadata has incomplete train/validation provenance")
    for value in [*fingerprints.values(), *provenance.values(), metadata.get("edge_order_sha256"),
                  metadata.get("configuration_sha256"), metadata.get("checkpoint_sha256")]:
        if not _is_sha256(value):
            raise ValueError("B-U run metadata contains a malformed SHA-256 identity")
    if not isinstance(metadata.get("source_sha256"), dict) or not metadata["source_sha256"]:
        raise ValueError("B-U run metadata has no source hashes")
    if any(not _is_sha256(value) for value in metadata["source_sha256"].values()):
        raise ValueError("B-U run metadata contains malformed source hashes")
    edge_order = metadata.get("directed_edge_order")
    if not isinstance(edge_order, list) or len(edge_order) != EXPECTED_DIRECTED_LINKS:
        raise ValueError("B-U run metadata has an invalid directed-edge order")
    if any(
        not isinstance(item, dict)
        or set(item) != {"index", "source", "destination"}
        or type(item.get("index")) is not int
        or item.get("index") != index
        or not isinstance(item.get("source"), str)
        or not isinstance(item.get("destination"), str)
        for index, item in enumerate(edge_order)
    ):
        raise ValueError("B-U run metadata has malformed directed-edge entries")
    edge_pairs = [[item["source"], item["destination"]] for item in edge_order]
    if canonical_json_sha256(edge_pairs) != metadata["edge_order_sha256"]:
        raise ValueError("B-U directed-edge order hash mismatch")
    if expected_edges is not None:
        expected_manifest_edges = [
            {"index": index, "source": source, "destination": destination}
            for index, (source, destination) in enumerate(expected_edges)
        ]
        if edge_order != expected_manifest_edges:
            raise ValueError("B-U directed-edge order differs from the validated topology")
    if expected_fingerprints is not None and fingerprints != expected_fingerprints:
        raise ValueError("B-U array fingerprints differ from the validated arrays")


def verify_checkpoint_sha256(path: Path, expected: str) -> str:
    if not _is_sha256(expected):
        raise ValueError("Checkpoint SHA-256 is missing or malformed")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError("Checkpoint SHA-256 mismatch")
    return actual


def make_bu_sequence(
    history_batch: np.ndarray,
    edge_index: torch.Tensor,
    node_count: int,
    normalization: Normalization,
) -> list[GraphSnapshot]:
    """Convert [B,5,E] windows to five disconnected directed graph copies."""
    if history_batch.ndim != 3 or history_batch.shape[1] != HISTORY_LENGTH:
        raise ValueError("History batch must have shape [batch, 5, directed_links]")
    if history_batch.shape[2] != edge_index.shape[1]:
        raise ValueError("History link axis and edge_index columns are not aligned")
    if not np.isfinite(history_batch).all():
        raise ValueError("History batch contains nonfinite values")

    batch_size = history_batch.shape[0]
    offsets = torch.arange(batch_size, dtype=torch.long) * node_count
    batched_edge_index = torch.cat([edge_index + offset for offset in offsets], dim=1)
    node_features = torch.ones((batch_size * node_count, 1), dtype=torch.float32)
    sequence: list[GraphSnapshot] = []
    for step in range(HISTORY_LENGTH):
        values = (history_batch[:, step, :].astype(np.float64) - normalization.mean) / normalization.scale
        edge_attr = torch.as_tensor(values.astype(np.float32).reshape(-1, 1))
        sequence.append(
            GraphSnapshot(
                x=node_features,
                edge_index=batched_edge_index,
                edge_attr=edge_attr,
                timestep=step,
            )
        )
    return sequence


def normalize_targets(values: np.ndarray, normalization: Normalization) -> torch.Tensor:
    if not np.isfinite(values).all():
        raise ValueError("Target batch contains nonfinite values")
    standardized = (values.astype(np.float64) - normalization.mean) / normalization.scale
    return torch.as_tensor(standardized.astype(np.float32)[:, :, None])


def validation_loss(
    model: nn.Module,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    edge_index: torch.Tensor,
    node_count: int,
    normalization: Normalization,
    *,
    batch_size: int = TRAINING["batch_size"],
) -> float:
    """Evaluate validation SmoothL1 loss without gradients or parameter updates."""
    if batch_size <= 0 or len(x_validation) == 0:
        raise ValueError("Validation requires nonempty arrays and a positive batch size")
    criterion = nn.SmoothL1Loss(reduction="mean")
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(x_validation), batch_size):
            stop = start + batch_size
            x_batch = make_bu_sequence(x_validation[start:stop], edge_index, node_count, normalization)
            target = normalize_targets(y_validation[start:stop], normalization)
            prediction = model(x_batch).reshape(len(target), edge_index.shape[1], 1)
            if not torch.isfinite(prediction).all():
                raise ValueError("Model produced a nonfinite validation output")
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Validation loss is nonfinite")
            total += float(loss) * target.numel()
            count += target.numel()
    return total / count


def validation_loss_improved(candidate_validation_loss: float, best_validation_loss: float) -> bool:
    """Checkpoint decision depends solely on the current and best validation loss."""
    if not np.isfinite(candidate_validation_loss):
        raise ValueError("Candidate validation loss must be finite")
    return candidate_validation_loss < best_validation_loss


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def _validate_run_id(run_id: str | None) -> None:
    if run_id is None:
        return
    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 48
        or not run_id[0].isalnum()
        or any(not (character.isalnum() or character in "_-") for character in run_id)
    ):
        raise ValueError("run_id must be 1-48 alphanumeric, underscore, or hyphen characters")


def output_dir_for_run(
    mode: str,
    seed: int,
    data_root: Path = V2_DATA_ROOT,
    *,
    run_id: str | None = None,
) -> Path:
    """Return a deterministic isolated destination for a smoke or full run."""
    _validate_run_id(run_id)
    if mode == "smoke":
        if run_id is not None:
            raise ValueError("run_id is available only for full runs")
        if seed != SEED:
            raise ValueError(f"Smoke mode retains its original seed {SEED}")
        requested = Path(data_root) / "neural_bu_v2"
    elif mode == "full":
        run_suffix = f"_{run_id}" if run_id is not None else ""
        requested = Path(data_root) / f"neural_bu_v2_full{run_suffix}_seed_{seed}"
    else:
        raise ValueError("mode must be 'smoke' or 'full'")
    return validate_output_dir(requested, data_root)


def assert_output_available(output: Path) -> None:
    """Fail before data loading if a run destination already exists."""
    if output.exists():
        manifest_path = output / "run_metadata.json"
        status = "manifest missing"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                status = str(manifest.get("status", "status not recorded"))
            except (OSError, json.JSONDecodeError):
                status = "manifest unreadable"
        raise FileExistsError(
            f"Refusing to reuse existing v2 B-U output (status={status}): {output}; "
            "recovery requires a separately reviewed procedure"
        )


def atomic_write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Atomically replace a JSON manifest using a temporary sibling file."""
    path = Path(path)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def validate_run_configuration(mode: str, seed: int, run_id: str | None = None) -> None:
    """Reject invalid run identity or protocol settings before creating output."""
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if mode == "smoke" and seed != SEED:
        raise ValueError(f"Smoke mode retains its original seed {SEED}")
    _validate_run_id(run_id)
    if mode == "smoke" and run_id is not None:
        raise ValueError("run_id is available only for full runs")
    expected_training = {
        "epochs": 5,
        "early_stopping_patience": 2,
        "learning_rate": 0.005,
        "batch_size": 32,
        "optimizer": "Adam",
        "loss": "SmoothL1Loss",
        "hidden_channels": 16,
        "gru_hidden_size": 16,
        "initialization": "from_scratch",
    }
    if TRAINING != expected_training:
        raise ValueError("B-U training configuration differs from the frozen v2 protocol")
    if HISTORY_LENGTH != 5 or PREDICTION_HORIZON != 1:
        raise ValueError("B-U window dimensions differ from the frozen v2 protocol")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initial_run_manifest(
    mode: str, seed: int, output: Path, data_root: Path, run_id: str | None = None,
    preflight_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "candidate": "B-U",
        "mode": mode,
        "status": "incomplete",
        "seed": seed,
        "model": "EdgeQoSForecaster (unchanged)",
        "model_settings": dict(TRAINING),
        "history_length": HISTORY_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "data_split": {
            "training": "X_train.npy and y_train.npy",
            "validation": "X_validation.npy and y_validation.npy",
            "test": "not loaded or evaluated",
        },
        "array_paths": [str(Path(data_root) / filename) for filename in ARRAY_NAMES],
        "normalization": {
            "fit_scope": "all scalar values in X_train.npy and y_train.npy only",
            "status": "pending training-array load",
            "accumulation_dtype": "float64",
        },
        "checkpoint_selection": "minimum validation SmoothL1 loss on standardized utilization",
        "history": [],
        "early_stopping": {
            "patience": TRAINING["early_stopping_patience"],
            "triggered": None,
            "criterion": "validation SmoothL1 loss only",
        },
        "started_at_utc": _utc_now(),
        "checkpoint": str((output / "best_validation.pt").resolve()),
        "manifest_path": str((output / "run_metadata.json").resolve()),
    }
    if run_id is not None:
        manifest["run_id"] = run_id
    if preflight_identity is not None:
        manifest.update(preflight_identity)
    return manifest


def build_run_manifest(
    *,
    mode: str,
    seed: int,
    arrays: WindowArrays,
    normalization: Normalization,
    edges: Sequence[tuple[str, str]],
    best_epoch: int,
    best_validation_loss: float,
    epochs_completed: int,
    history: list[dict[str, float | int]],
    early_stopping_triggered: bool,
    training_seconds: float,
    checkpoint_path: Path,
    manifest_path: Path,
    run_identity: dict[str, object],
    data_root: Path = V2_DATA_ROOT,
) -> dict[str, object]:
    """Build reproducibility metadata without reading or writing artifacts."""
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    manifest: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "candidate": "B-U",
        "mode": mode,
        "seed": seed,
        "model": "EdgeQoSForecaster (unchanged)",
        "model_settings": dict(TRAINING),
        "history_length": HISTORY_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "data_split": {
            "training": "X_train.npy and y_train.npy",
            "validation": "X_validation.npy and y_validation.npy",
            "test": "not loaded or evaluated",
        },
        "array_paths": [str(Path(data_root) / filename) for filename in ARRAY_NAMES],
        "array_shapes": {
            "X_train": list(arrays.x_train.shape),
            "y_train": list(arrays.y_train.shape),
            "X_validation": list(arrays.x_validation.shape),
            "y_validation": list(arrays.y_validation.shape),
        },
        "sample_counts": {
            "train": int(arrays.x_train.shape[0]),
            "validation": int(arrays.x_validation.shape[0]),
        },
        "directed_edge_order": [
            {"index": index, "source": source, "destination": destination}
            for index, (source, destination) in enumerate(edges)
        ],
        "edge_order_provenance": (
            "same topology file/order and interleaved (u,v),(v,u) construction as "
            "phase9_geant/build_validity_v2.py; node indices sorted lexicographically"
        ),
        "normalization": {
            "mean": normalization.mean,
            "scale": normalization.scale,
            "fit_scope": normalization.fit_scope,
            "accumulation_dtype": "float64",
        },
        "checkpoint_selection": "minimum validation SmoothL1 loss on standardized utilization",
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": epochs_completed,
        "early_stopping": {
            "patience": TRAINING["early_stopping_patience"],
            "triggered": early_stopping_triggered,
            "criterion": "validation SmoothL1 loss only",
        },
        "training_seconds": training_seconds,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
    }
    manifest.update(run_identity)
    validate_required_run_metadata(
        {**manifest, "checkpoint_sha256": "0" * 64},
        expected_edges=edges,
        expected_fingerprints=run_identity.get("array_fingerprints"),
    )
    return manifest


def _train_bu_arrays(
    arrays: WindowArrays, *, mode: str, seed: int, output: Path, data_root: Path,
    run_identity: dict[str, object],
) -> dict[str, object]:
    """Train from already loaded train/validation arrays and save the checkpoint."""
    nodes, undirected_links = parse_topology(str(TOPOLOGY_PATH))
    if len(nodes) != EXPECTED_NODES or len(undirected_links) != EXPECTED_UNDIRECTED_LINKS:
        raise ValueError("Topology dimensions differ from the v2 GEANT topology contract")
    edge_index, node_order, edges = build_edge_index(tuple(nodes), undirected_links)
    if len(edges) != EXPECTED_DIRECTED_LINKS:
        raise ValueError("Topology does not produce exactly 72 directed links")
    # Reconstruct the model's edge_index columns and compare them with the
    # interleaved order used by v2 preprocessing. The arrays do not themselves
    # carry edge labels, so this is a source-provenance contract, not a checksum.
    model_edges = tuple(
        (node_order[int(source)], node_order[int(destination)])
        for source, destination in edge_index.T.tolist()
    )
    assert_edge_order(directed_edges_v2(undirected_links), model_edges)
    if run_identity.get("directed_edge_order") != [
        {"index": index, "source": source, "destination": destination}
        for index, (source, destination) in enumerate(edges)
    ]:
        raise ValueError("Training topology edge order differs from the preflighted run identity")

    normalization = fit_training_normalization(arrays.x_train, arrays.y_train)
    _seed_everything(seed)
    model = EdgeQoSForecaster(
        edge_feature_count=1,
        target_count=1,
        hidden_channels=TRAINING["hidden_channels"],
        gru_hidden_size=TRAINING["gru_hidden_size"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["learning_rate"])
    criterion = nn.SmoothL1Loss(reduction="mean")
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    # The lifecycle wrapper owns directory creation and has already written the
    # initial incomplete manifest before invoking this training function.
    checkpoint_path = output / "best_validation.pt"
    manifest_path = output / "run_metadata.json"
    started = time.perf_counter()
    early_stopping_triggered = False

    for epoch in range(1, TRAINING["epochs"] + 1):
        model.train()
        order = np.random.permutation(len(arrays.x_train))
        batch_losses: list[float] = []
        for begin in range(0, len(order), TRAINING["batch_size"]):
            indices = order[begin:begin + TRAINING["batch_size"]]
            sequence = make_bu_sequence(arrays.x_train[indices], edge_index, len(node_order), normalization)
            target = normalize_targets(arrays.y_train[indices], normalization)
            prediction = model(sequence).reshape(len(indices), len(edges), 1)
            if not torch.isfinite(prediction).all():
                raise ValueError("Model produced a nonfinite training output")
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is nonfinite")
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        val_loss = validation_loss(
            model,
            arrays.x_validation,
            arrays.y_validation,
            edge_index,
            len(node_order),
            normalization,
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses, dtype=np.float64)),
            "validation_loss": val_loss,
        })
        if validation_loss_improved(val_loss, best_loss):
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs >= TRAINING["early_stopping_patience"]:
            early_stopping_triggered = True
            break

    metadata = build_run_manifest(
        mode=mode,
        seed=seed,
        arrays=arrays,
        normalization=normalization,
        edges=edges,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        epochs_completed=len(history),
        history=history,
        early_stopping_triggered=early_stopping_triggered,
        training_seconds=time.perf_counter() - started,
        checkpoint_path=checkpoint_path,
        manifest_path=output / "run_metadata.json",
        run_identity=run_identity,
        data_root=data_root,
    )
    metadata["checkpoint_sha256"] = _sha256_file(checkpoint_path)
    return metadata


def _run_validation_training(
    *,
    mode: str,
    seed: int,
    data_root: Path = V2_DATA_ROOT,
    array_loader: Callable[[Path], WindowArrays] | None = None,
    training_fn: Callable[..., dict[str, object]] | None = None,
    run_id: str | None = None,
    provenance_loader: Callable[[Path], tuple[dict[str, object], str, dict[str, str]]] | None = None,
    topology_loader: Callable[[], tuple[tuple[str, str], ...]] | None = None,
    source_hasher: Callable[[Path], dict[str, str]] | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Run with durable incomplete/failed/complete lifecycle state transitions."""
    validate_run_configuration(mode, seed, run_id)
    output = output_dir_for_run(mode, seed, data_root, run_id=run_id)
    assert_output_available(output)

    # Validate static provenance, topology ordering, source identity, and the
    # frozen configuration before creating an output directory or training.
    provenance, provenance_file_hash, provenance_hashes = (provenance_loader or load_window_provenance)(data_root)
    edges = (topology_loader or _topology_edge_order)()
    source_hashes = (source_hasher or _source_sha256)(repository_root)
    preflight_identity = _preflight_run_identity(
        mode=mode, seed=seed, run_id=run_id,
        provenance_file_sha256=provenance_file_hash,
        provenance_hashes=provenance_hashes,
        edges=edges,
        source_sha256=source_hashes,
    )

    # Bind the actual loaded arrays to their provenance before reserving an
    # output directory. A malformed or mismatched source therefore leaves no
    # run directory that could be mistaken for a started training attempt.
    loader = array_loader or load_v2_windows
    arrays = loader(data_root)
    run_identity = build_array_provenance(
        arrays, provenance,
        provenance_file_sha256=provenance_file_hash,
        edges=edges,
        preflight_identity=preflight_identity,
    )
    source_hashes_after_preflight = (source_hasher or _source_sha256)(repository_root)
    if source_hashes_after_preflight != source_hashes:
        raise ValueError("Run source files changed during preflight; refusing output creation")

    # Recheck the canonical root and every existing component immediately
    # before reserving the destination. Do not remove the directory if this
    # post-create check detects a concurrent replacement or redirection.
    _, output = validate_fresh_child(
        data_root,
        output,
        repository_root,
        expected_name=output.name,
    )
    assert_output_available(output)
    output.mkdir(parents=False, exist_ok=False)
    verify_new_directory_containment(output, data_root)
    manifest_path = output / "run_metadata.json"
    current_manifest = _initial_run_manifest(
        mode, seed, output, data_root, run_id, preflight_identity
    )
    current_manifest.update(run_identity)
    atomic_write_manifest(manifest_path, current_manifest)
    started = time.perf_counter()

    try:
        executor = training_fn or _train_bu_arrays
        if training_fn is None:
            result = executor(
                arrays, mode=mode, seed=seed, output=output, data_root=data_root,
                run_identity=run_identity,
            )
        else:
            result = executor(
                arrays, mode=mode, seed=seed, output=output, data_root=data_root
            )
        checkpoint_path = Path(str(result["checkpoint"]))
        expected_checkpoint = output / "best_validation.pt"
        if checkpoint_path.resolve() != expected_checkpoint.resolve():
            raise ValueError("Training returned a checkpoint outside its run directory")
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise RuntimeError("Training did not produce a nonempty best-validation checkpoint")

        declared_checkpoint_hash = result.get("checkpoint_sha256")
        actual_checkpoint_hash = (
            verify_checkpoint_sha256(checkpoint_path, declared_checkpoint_hash)
            if declared_checkpoint_hash is not None
            else _sha256_file(checkpoint_path)
        )
        source_hashes_after = (source_hasher or _source_sha256)(repository_root)
        if source_hashes_after != source_hashes:
            raise ValueError("Run source files changed during training; refusing completion")

        completed_manifest = dict(result)
        completed_manifest.update(run_identity)
        completed_manifest.update(
            {
                "mode": mode,
                "seed": seed,
                "status": "complete",
                "finished_at_utc": _utc_now(),
                "training_seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": actual_checkpoint_hash,
                "source_sha256_posttraining": source_hashes_after,
                "manifest_path": str(manifest_path.resolve()),
            }
        )
        validate_required_run_metadata(
            completed_manifest,
            expected_edges=edges,
            expected_fingerprints=run_identity["array_fingerprints"],
        )
        if run_id is not None:
            completed_manifest["run_id"] = run_id
        # This is intentionally the final lifecycle transition and final operation.
        atomic_write_manifest(manifest_path, completed_manifest)
        return output
    except BaseException as original_error:
        failed_manifest = dict(current_manifest)
        failed_manifest.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "training_seconds": time.perf_counter() - started,
                "exception": {
                    "type": type(original_error).__name__,
                    "message": str(original_error)[:500],
                },
            }
        )
        try:
            atomic_write_manifest(manifest_path, failed_manifest)
        except BaseException:
            # Preserve the original exception; atomic replacement leaves the last
            # durable manifest (normally status=incomplete) intact.
            pass
        raise


def run_validation_smoke(*, seed: int = SEED) -> Path:
    """Run the unchanged five-epoch-configured smoke path at seed 42."""
    return _run_validation_training(mode="smoke", seed=seed)


def run_validation_full(
    *, seed: int, run_id: str | None = None, data_root: Path = V2_DATA_ROOT
) -> Path:
    """Run one explicitly selected full protocol repetition."""
    return _run_validation_training(
        mode="full", seed=seed, run_id=run_id, data_root=data_root
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-smoke",
        action="store_true",
        help="run the original seed-42 validation-only smoke path",
    )
    mode.add_argument(
        "--run-full",
        action="store_true",
        help="run one explicitly seeded validation-only full repetition",
    )
    parser.add_argument("--seed", type=int, help="required seed for --run-full")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=V2_DATA_ROOT,
        help="canonical v2 root or explicitly versioned rebuild root (full mode only)",
    )
    parser.add_argument(
        "--run-id",
        help="optional alphanumeric label for an isolated fresh full-run output directory",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not args.run_smoke and not args.run_full:
        parser.error("Choose an explicit mode: --run-smoke or --run-full")
    if args.run_full and args.seed is None:
        parser.error("--run-full requires --seed")
    if args.run_smoke and args.seed is not None:
        parser.error("--run-smoke retains seed 42; --seed is only valid with --run-full")
    if args.run_smoke and args.run_id is not None:
        parser.error("--run-id is only valid with --run-full")
    if args.run_smoke and args.data_root != V2_DATA_ROOT:
        parser.error("--data-root is only valid with --run-full; smoke mode retains its original path")
    if args.run_id is not None:
        try:
            _validate_run_id(args.run_id)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = (
        run_validation_smoke()
        if args.run_smoke
        else run_validation_full(seed=args.seed, run_id=args.run_id, data_root=args.data_root)
    )
    print(output)


if __name__ == "__main__":
    main()
