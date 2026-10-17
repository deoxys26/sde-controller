"""Synthetic-only tests for guarded final-evaluator control flow."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from phase9_geant import evaluate_bu_v2_final as final


@pytest.fixture(autouse=True)
def _synthetic_roots_only(monkeypatch, tmp_path):
    # Every repository used by this file is created under pytest's temporary
    # directory. Real final evaluation remains blocked outside these tests.
    monkeypatch.setattr(final, "CANONICAL_REPO_ROOT", tmp_path / "DN_proj")
    monkeypatch.setattr(final, "SOURCE_FREEZE_REVIEWED", True)


def _fake_topology():
    nodes = {f"n{i:02d}": None for i in range(final.NODES)}
    links = [(f"n{i:02d}", f"n{i + 1:02d}") for i in range(final.NODES - 1)]
    for i in range(final.NODES):
        for j in range(i + 2, final.NODES):
            pair = (f"n{i:02d}", f"n{j:02d}")
            if pair not in links:
                links.append(pair)
            if len(links) == final.UNDIRECTED_LINKS:
                break
        if len(links) == final.UNDIRECTED_LINKS:
            break
    return nodes, links


def _repo_fixture(tmp_path: Path):
    repo = tmp_path / "DN_proj"
    v2 = repo / final.V2_REL
    (repo / (final.V2_REL / "final_test_protocol_v1")).mkdir(parents=True)
    for directory in (
        v2 / "neural_bu_v2_full_seed_42",
        v2 / "neural_bu_v2_full_seed_123",
        v2 / "neural_bu_v2_full_seed_2026",
    ):
        directory.mkdir(parents=True)
    (repo / final.TOPOLOGY_REL).parent.mkdir(parents=True)
    (repo / final.TOPOLOGY_REL).write_text("synthetic topology fixture", encoding="utf-8")
    (repo / final.PROTOCOL_REL).write_text(
        "\n".join(
            (
                final.PROTOCOL_ID,
                final.DATA_PROTOCOL_ID,
                str(final.CANONICAL_REPO_ROOT),
                "[9740,11460)",
                "X_test.npy",
                "y_test.npy",
                "window_starts.json",
                "metadata.json",
                "file_id",
                "on-disk source snapshot SHA-256",
                "Final evaluation is blocked pending an independently reviewed source freeze",
                final.TOPOLOGY_SHA256,
                "22 unique nodes, 36 undirected links, and 72 directed links",
                "Do not retry",
                "separate, explicit approval",
                f"mean `mu = {final.SCALER_MEAN}`",
                f"scale `sigma = {final.SCALER_SCALE}`",
                f"alpha = {final.EWMA_ALPHA}",
                "[-0.01057, 0.07411, 0.13645, 0.13489, 0.65767]",
                f"b = {final.AR_INTERCEPT}",
                "float64",
                *[str(path).replace("\\", "/") + "` | `" + digest + "`"
                  for path, digest in final.CHECKPOINTS.values()],
            )
        ),
        encoding="utf-8",
    )
    (v2 / "metadata.json").write_text(json.dumps({
        "protocol_id": final.DATA_PROTOCOL_ID,
        "split_bounds": {name: list(bounds) for name, bounds in final.SPLITS.items()},
    }), encoding="utf-8")
    (repo / "phase9_geant").mkdir(parents=True, exist_ok=True)
    (repo / "phase9_geant" / "evaluate_bu_v2_final.py").write_text("synthetic source", encoding="utf-8")
    (repo / "phase9_geant" / "parse_geant.py").write_text("synthetic parser", encoding="utf-8")
    (repo / "src" / "gnn").mkdir(parents=True)
    (repo / "src" / "gnn" / "model.py").write_text("synthetic model", encoding="utf-8")
    for name in ("X_test.npy", "y_test.npy", "window_starts.json"):
        (v2 / name).write_text("not opened in preflight", encoding="utf-8")

    nodes, links = _fake_topology()
    edges = final.validate_topology(nodes, links)
    expected_inputs = [
        str(repo / final.V2_REL / name)
        for name in ("X_train.npy", "y_train.npy", "X_validation.npy", "y_validation.npy")
    ]
    for seed, (checkpoint_rel, _) in final.CHECKPOINTS.items():
        checkpoint = repo / checkpoint_rel
        checkpoint.write_bytes(b"synthetic checkpoint identity fixture")
        manifest = {
            "status": "complete",
            "mode": "full",
            "seed": seed,
            "protocol_id": final.DATA_PROTOCOL_ID,
            "candidate": "B-U",
            "model": "EdgeQoSForecaster (unchanged)",
            "history_length": 5,
            "prediction_horizon": 1,
            "model_settings": {"hidden_channels": 16, "gru_hidden_size": 16},
            "normalization": {
                "mean": final.SCALER_MEAN,
                "scale": final.SCALER_SCALE,
                "fit_scope": "all scalar values in X_train.npy and y_train.npy only",
                "accumulation_dtype": "float64",
            },
            "array_paths": expected_inputs,
            "directed_edge_order": [
                {"index": i, "source": u, "destination": v}
                for i, (u, v) in enumerate(edges)
            ],
        }
        (checkpoint.parent / "run_metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
    return repo, nodes, links


def _preflight(repo: Path, nodes, links, *, digest=final.TOPOLOGY_SHA256):
    call_count = 0
    checkpoint_hashes = [pair[1] for pair in final.CHECKPOINTS.values()]
    def fake_hash(_content: bytes) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return digest
        return checkpoint_hashes[call_count - 2]

    return final.preflight(
        repo,
        topology_parser=lambda _path: (nodes, links),
        hash_content=fake_hash,
    )


def _authorized_synthetic_call(*args, **kwargs):
    context = final._CLI_AUTHORIZATION_CONTEXT.set(True)
    try:
        return final._execute_after_cli_authorization(*args, **kwargs)
    finally:
        final._CLI_AUTHORIZATION_CONTEXT.reset(context)


def _synthetic_inputs():
    x = np.arange(2 * final.HISTORY * final.LINKS, dtype=np.float32).reshape(2, final.HISTORY, final.LINKS)
    x /= 1000.0
    y = np.zeros((2, final.LINKS), dtype=np.float32)
    return x, y, [9740, 9741]


def test_preflight_does_not_open_test_arrays_or_window_map(tmp_path, monkeypatch):
    repo, nodes, links = _repo_fixture(tmp_path)
    forbidden = {
        (repo / final.TEST_X_REL).absolute(),
        (repo / final.TEST_Y_REL).absolute(),
        (repo / final.WINDOWS_REL).absolute(),
    }
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.absolute() in forbidden:
            raise AssertionError(f"test artifact was opened during preflight: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    checked = _preflight(repo, nodes, links)
    assert checked.topology_edges == final.directed_edges(links)


def test_wrong_topology_hash_stops_before_parser_and_test_access(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    called = False

    def parser(_path):
        nonlocal called
        called = True
        return nodes, links

    with pytest.raises(ValueError, match="Topology SHA-256 mismatch"):
        final.preflight(repo, topology_parser=parser,
                        hash_content=lambda _content: "BAD")
    assert not called


def test_canonical_root_mismatch_is_rejected(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    with pytest.raises(ValueError, match="approved canonical checkout"):
        final.preflight(tmp_path / "different" / "DN_proj",
                        topology_parser=lambda _text: (nodes, links))


def test_direct_execution_with_or_without_module_token_is_rejected(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    called = False
    def loader(_check):
        nonlocal called
        called = True
        raise AssertionError("test access must not occur")
    with pytest.raises(PermissionError, match="Direct evaluator calls are unsupported"):
        final.execute_authorized(checked, model_loader=lambda _check: object(), test_loader=loader)
    with pytest.raises(PermissionError, match="Direct evaluator calls are unsupported"):
        final.execute_authorized(
            checked, authorization=final._CLI_AUTHORIZATION_TOKEN,
            model_loader=lambda _check: object(), test_loader=loader,
        )
    assert not called
    assert not checked.output.exists()


def test_forged_and_nested_modified_preflight_are_rejected(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    forged = final.Preflight(**checked.__dict__)
    with pytest.raises(ValueError, match="forged, unissued, or was modified"):
        _authorized_synthetic_call(
            forged,
            model_loader=lambda _check: pytest.fail("loader must not run"),
            test_loader=lambda _check: pytest.fail("test loader must not run"),
        )

    checked.test_identities["X_test.npy"]["size"] += 1
    with pytest.raises(ValueError, match="forged, unissued, or was modified"):
        _authorized_synthetic_call(
            checked,
            model_loader=lambda _check: pytest.fail("loader must not run"),
            test_loader=lambda _check: pytest.fail("test loader must not run"),
        )


def test_real_execution_remains_blocked_until_source_freeze_is_reviewed(tmp_path, monkeypatch):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    monkeypatch.setattr(final, "SOURCE_FREEZE_REVIEWED", False)
    with pytest.raises(RuntimeError, match="no independently reviewed freeze"):
        _authorized_synthetic_call(
            checked,
            model_loader=lambda _check: pytest.fail("model loader must not run"),
            test_loader=lambda _check: pytest.fail("test loader must not run"),
        )
    assert not checked.output.exists()


def test_protocol_and_source_provenance_use_validated_snapshots(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    before = final._build_provenance(checked)
    protocol = repo / final.PROTOCOL_REL
    protocol.write_text(protocol.read_text(encoding="utf-8") + "\nsynthetic later edit\n", encoding="utf-8")
    after = final._build_provenance(checked)
    assert after["source_snapshot_sha256"] == before["source_snapshot_sha256"]
    assert after["source_snapshot_is_runtime_code_identity"] is False
    with pytest.raises(ValueError, match="Source file identity changed after preflight"):
        final._validate_preflight_for_execution(checked)

    checked_again = _preflight(repo, nodes, links)
    model_source = repo / "src" / "gnn" / "model.py"
    model_source.write_text("synthetic later model edit", encoding="utf-8")
    assert final._build_provenance(checked_again)["source_snapshot_sha256"] == checked_again.source_snapshot_sha256
    with pytest.raises(ValueError, match="Source file identity changed after preflight"):
        final._validate_preflight_for_execution(checked_again)


def test_cloud_tag_without_file_identity_fails_closed(tmp_path):
    path = tmp_path / "cloud-placeholder"
    identity = {"device": 0, "file_id": 0, "size": 10, "mtime_ns": 1}
    with pytest.raises(ValueError, match="lacks verifiable file identity"):
        final._validate_cloud_identity(next(iter(final._CLOUD_TAGS)), identity, path)


def test_recognized_cloud_tag_is_allowed_only_with_matching_path_target_identity(tmp_path):
    repo = tmp_path / "DN_proj"
    repo.mkdir()
    file = repo / "synthetic-cloud-placeholder"
    file.write_text("synthetic", encoding="utf-8")
    cloud_tag = next(iter(final._CLOUD_TAGS))
    final.validate_path_components(
        file,
        repo_root=repo,
        must_exist=True,
        reparse_tag_reader=lambda path: cloud_tag if path == file else None,
    )
    final._validate_cloud_identity(cloud_tag, final._path_identity(file), file)


def test_simulated_junction_component_fails_closed(tmp_path, monkeypatch):
    repo = tmp_path / "DN_proj"
    repo.mkdir()
    junction = repo / "synthetic-junction"
    junction.mkdir()
    candidate = junction / "not-opened.txt"

    def is_synthetic_junction(path):
        return path == junction

    monkeypatch.setattr(type(junction), "is_junction", is_synthetic_junction, raising=False)
    with pytest.raises(ValueError, match="Junction path component"):
        final.validate_path_components(candidate, repo_root=repo, must_exist=False)


def test_pinned_open_rejects_simulated_identity_change_after_open(tmp_path, monkeypatch):
    repo = tmp_path / "DN_proj"
    repo.mkdir()
    path = repo / "synthetic-input.bin"
    path.write_bytes(b"synthetic only")
    identity = final._path_identity(path)
    original_identity = final._path_identity
    calls = 0

    def changed_path_identity(candidate):
        nonlocal calls
        observed = original_identity(candidate)
        if candidate == path:
            calls += 1
            if calls >= 2:
                observed["mtime_ns"] += 1
        return observed

    monkeypatch.setattr(final, "_path_identity", changed_path_identity)
    with pytest.raises(ValueError, match="Opened file does not match"):
        final._open_pinned_file(path, identity, repo)


def test_topology_path_replacement_after_hash_cannot_change_parsed_snapshot(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    topology = repo / final.TOPOLOGY_REL
    original = topology.read_bytes()
    called = 0
    checkpoint_hashes = [pair[1] for pair in final.CHECKPOINTS.values()]
    def hash_content(content):
        nonlocal called
        called += 1
        if called == 1:
            topology.write_bytes(b"replacement after snapshot")
            return final.TOPOLOGY_SHA256
        return checkpoint_hashes[called - 2]
    parsed = []
    checked = final.preflight(
        repo, hash_content=hash_content,
        topology_parser=lambda text: (parsed.append(text) or (nodes, links)),
    )
    assert parsed == [original.decode("utf-8")]
    assert checked.topology_edges == final.directed_edges(links)


def test_checkpoint_replacement_after_preflight_is_rejected_before_load(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    checkpoint = checked.checkpoints[42]
    checkpoint.write_bytes(b"replaced synthetic checkpoint")
    with pytest.raises(ValueError, match="identity changed|does not match"):
        final._load_frozen_models(checked)


def test_checkpoint_deserializer_snapshot_is_unchanged_by_later_path_replacement(tmp_path):
    checkpoint = tmp_path / "synthetic.pt"
    original = b"synthetic verified checkpoint bytes"
    checkpoint.write_bytes(original)
    identity = final._path_identity(checkpoint)
    snapshot = final._verified_checkpoint_snapshot(
        checkpoint, identity, final._sha256_bytes(original), tmp_path
    )
    checkpoint.write_bytes(b"replacement synthetic checkpoint")
    assert snapshot == original


def test_test_input_replacement_after_preflight_is_rejected_before_array_read(tmp_path, monkeypatch):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    checked.test_x.write_text("replacement synthetic input", encoding="utf-8")
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("array content read before identity rejection"))
    with pytest.raises(ValueError, match="identity changed"):
        final._load_test_inputs(checked)


def test_test_inputs_are_loaded_from_pinned_handles_and_identity_is_recorded(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    v2 = repo / final.V2_REL
    x = np.arange(2 * 5 * 72, dtype=np.float32).reshape(2, 5, 72)
    y = np.arange(2 * 72, dtype=np.float32).reshape(2, 72)
    with (v2 / "X_test.npy").open("wb") as handle:
        np.save(handle, x, allow_pickle=False)
    with (v2 / "y_test.npy").open("wb") as handle:
        np.save(handle, y, allow_pickle=False)
    (v2 / "window_starts.json").write_text(json.dumps({"test": [9740, 9741]}), encoding="utf-8")
    checked = _preflight(repo, nodes, links)
    loaded_x, loaded_y, starts, identities = final._load_test_inputs(checked)
    np.testing.assert_array_equal(loaded_x, x)
    np.testing.assert_array_equal(loaded_y, y)
    assert starts == [9740, 9741]
    assert set(identities) == {"X_test.npy", "y_test.npy", "window_starts.json"}


def test_test_input_path_replacement_after_open_is_detected(tmp_path, monkeypatch):
    repo, nodes, links = _repo_fixture(tmp_path)
    v2 = repo / final.V2_REL
    x, y, starts = _synthetic_inputs()
    for name, values in (("X_test.npy", x), ("y_test.npy", y)):
        with (v2 / name).open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
    (v2 / "window_starts.json").write_text(json.dumps({"test": starts}), encoding="utf-8")
    checked = _preflight(repo, nodes, links)
    original_open = final._open_pinned_file
    replaced = False
    def replace_after_open(path, identity, root):
        nonlocal replaced
        handle = original_open(path, identity, root)
        if path == checked.test_x and not replaced:
            replacement = path.with_name("X_test.replacement.npy")
            with replacement.open("wb") as other:
                np.save(other, x + 1, allow_pickle=False)
            try:
                replacement.replace(path)
            except OSError:
                handle.close()
                pytest.skip("filesystem does not permit replacement while the old handle is open")
            replaced = True
        return handle
    monkeypatch.setattr(final, "_open_pinned_file", replace_after_open)
    with pytest.raises(ValueError, match="changed while being consumed"):
        final._load_test_inputs(checked)
    assert replaced


def test_concurrent_authorized_attempts_only_one_consumes_output_gate(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    x, y, starts = _synthetic_inputs()
    successes = []
    failures = []
    loader_calls = []
    def worker():
        try:
            result = _authorized_synthetic_call(
                checked,
                model_loader=lambda _check: object(),
                test_loader=lambda _check: (loader_calls.append(1) or
                                            (x, y, starts, checked.test_identities)),
                prediction_fn=lambda _check, _models, _x: {s: np.zeros_like(y) for s in (42, 123, 2026)},
            )
            successes.append(result)
        except BaseException as exc:
            failures.append(exc)
    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(successes) == 1
    assert len(failures) == 1 and isinstance(failures[0], FileExistsError)
    assert len(loader_calls) == 1


@pytest.mark.parametrize("node_count,link_count", [(21, 36), (22, 35)])
def test_topology_contract_fails_closed_on_wrong_structure(node_count, link_count):
    nodes, links = _fake_topology()
    altered_nodes = dict(list(nodes.items())[:node_count])
    with pytest.raises(ValueError):
        final.validate_topology(altered_nodes, links[:link_count])


def test_topology_contract_accepts_exact_synthetic_shape():
    nodes, links = _fake_topology()
    assert len(final.validate_topology(nodes, links)) == 72


def test_malformed_synthetic_topology_text_is_rejected():
    from phase9_geant.parse_geant import parse_topology_text
    with pytest.raises(ValueError, match="first line must start"):
        parse_topology_text("not an SNDlib topology")


def test_edge_order_interleaves_each_reverse_pair():
    _nodes, links = _fake_topology()
    result = final.directed_edges(links)
    assert len(result) == 72
    for index, (source, target) in enumerate(links):
        assert result[index * 2:index * 2 + 2] == ((source, target), (target, source))


def test_checkpoint_manifest_edge_order_mismatch_is_rejected(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    edges = final.validate_topology(nodes, links)
    manifest_path = repo / final.CHECKPOINTS[42][0].parent / "run_metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["directed_edge_order"][0], manifest["directed_edge_order"][1] = (
        manifest["directed_edge_order"][1], manifest["directed_edge_order"][0]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="directed-edge ordering mismatch"):
        final._validate_run_manifest(manifest_path, 42, edges, repo, final._path_identity(manifest_path))


def test_exact_artifact_allow_list_rejects_alternate_location(tmp_path):
    repo = tmp_path / "DN_proj"
    repo.mkdir()
    with pytest.raises(ValueError, match="not allow-listed"):
        final._require_exact_path(repo / "elsewhere" / "X_test.npy", final.TEST_X_REL, repo)


def test_path_validation_rejects_symlink_component(tmp_path):
    repo = tmp_path / "DN_proj"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = repo / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this test environment")
    candidate = link / "file.txt"
    with pytest.raises(ValueError, match="Symlink"):
        final.validate_path_components(candidate, repo_root=repo, must_exist=False)


def test_unknown_reparse_tag_fails_closed(tmp_path):
    repo = tmp_path / "DN_proj"
    repo.mkdir()
    file = repo / "placeholder"
    file.write_text("synthetic", encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized/path-redirection reparse tag"):
        final.validate_path_components(
            file,
            repo_root=repo,
            must_exist=True,
            reparse_tag_reader=lambda _p: 0xDEADBEEF,
        )


def test_preflight_only_cli_never_calls_test_loader(tmp_path, monkeypatch, capsys):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    monkeypatch.setattr(final, "preflight", lambda: checked)
    monkeypatch.setattr(final, "_load_test_inputs", lambda _check: pytest.fail("test loader called"))
    assert final.main(["--preflight-only"]) == 0
    assert '"test_access": "none"' in capsys.readouterr().out
    assert not checked.output.exists()


def test_cli_authorization_passes_claim_to_central_execution_gate(tmp_path, monkeypatch):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    captured = {}
    monkeypatch.setattr(final, "preflight", lambda: checked)
    def fake_execute(value):
        captured["preflight"] = value
        captured["authorization"] = True
        return checked.output
    monkeypatch.setattr(final, "_execute_after_cli_authorization", fake_execute)
    assert final.main(["--authorize-final-test-evaluation"]) == 0
    assert captured["preflight"] is checked
    assert captured["authorization"] is True
    assert not checked.output.exists()


def test_authorization_state_is_durable_before_first_test_read(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    x, y, starts = _synthetic_inputs()
    loader_called = False

    def loader(_check):
        nonlocal loader_called
        loader_called = True
        state = json.loads((checked.output / "run_state.json").read_text(encoding="utf-8"))
        assert state["status"] == "test_access_started"
        assert state["test_access_state"] == "may_have_started"
        return x, y, starts, checked.test_identities

    fake_models = object()
    def predictor(_check, models, values):
        assert models is fake_models
        assert values is x
        return {seed: np.zeros_like(y) for seed in (42, 123, 2026)}

    output = _authorized_synthetic_call(
        checked,
        model_loader=lambda _check: fake_models,
        test_loader=loader,
        prediction_fn=predictor,
    )
    assert loader_called
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["test_access_state"] == "completed"
    assert results["split"]["target_indices"] == [9745, 9746]
    assert results["metrics"]["Persistence"]["scalar_pairs"] == 144
    assert results["attempt_id"] == state["attempt_id"]
    provenance = results["provenance"]
    assert provenance["invocation"] and provenance["source_snapshot_sha256"]
    assert provenance["source_snapshot_is_runtime_code_identity"] is False
    assert set(provenance["input_paths"]) == {"X_test.npy", "y_test.npy", "window_starts.json"}
    assert set(provenance["checkpoints"]) == {"42", "123", "2026"}
    assert provenance["topology"]["sha256"] == final.TOPOLOGY_SHA256
    assert provenance["model"]["history"] == 5 and provenance["normalization"]["fit_scope"] == "training only"


def test_existing_or_failed_destination_blocks_retry_before_loader(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    checked.output.mkdir()
    (checked.output / "run_state.json").write_text('{"status":"failed"}', encoding="utf-8")
    called = False

    def loader(_check):
        nonlocal called
        called = True
        raise AssertionError("must not access test data")

    with pytest.raises(FileExistsError, match="Output or prior-attempt"):
        _authorized_synthetic_call(checked,
                                 model_loader=lambda _check: object(), test_loader=loader)
    assert not called


def test_alternate_output_path_is_rejected_before_directory_creation(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    forged = final.Preflight(
        **{**checked.__dict__, "output": repo / "elsewhere" / "attempt"}
    )
    with pytest.raises(ValueError, match="forged, unissued, or was modified"):
        _authorized_synthetic_call(
            forged,
            model_loader=lambda _check: pytest.fail("model loader must not run"),
            test_loader=lambda _check: pytest.fail("test loader must not run"),
        )
    assert not (repo / "elsewhere").exists()


def test_failure_after_access_begins_is_recorded_and_not_resumable(tmp_path):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    expected = RuntimeError("synthetic loader failure")

    def loader(_check):
        raise expected

    with pytest.raises(RuntimeError, match="synthetic loader failure") as caught:
        _authorized_synthetic_call(checked,
                                 model_loader=lambda _check: object(), test_loader=loader)
    assert caught.value is expected
    failed = json.loads((checked.output / "run_state.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["test_access_state"] == "may_have_started"
    with pytest.raises(FileExistsError):
        _authorized_synthetic_call(checked,
                                 model_loader=lambda _check: object(), test_loader=loader)


def test_failure_recording_error_does_not_mask_original(tmp_path, monkeypatch):
    repo, nodes, links = _repo_fixture(tmp_path)
    checked = _preflight(repo, nodes, links)
    original_write = final._write_state
    expected = RuntimeError("primary failure")

    def fail_failed_write(output, state):
        if state.get("status") == "failed":
            raise OSError("synthetic manifest failure")
        original_write(output, state)

    monkeypatch.setattr(final, "_write_state", fail_failed_write)
    with pytest.raises(RuntimeError, match="primary failure") as caught:
        _authorized_synthetic_call(
            checked,
            model_loader=lambda _check: object(),
            test_loader=lambda _check: (_ for _ in ()).throw(expected),
        )
    assert caught.value is expected
    durable = json.loads((checked.output / "run_state.json").read_text(encoding="utf-8"))
    assert durable["status"] == "test_access_started"


def test_atomic_json_replacement_failure_preserves_previous_state(tmp_path, monkeypatch):
    target = tmp_path / "run_state.json"
    target.write_text('{"status":"test_access_started"}\n', encoding="utf-8")
    previous = target.read_text(encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(final.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        final._atomic_json(target, {"status": "complete"})
    assert target.read_text(encoding="utf-8") == previous
    assert list(tmp_path.glob(".run_state.json.*.tmp")) == []


def test_invalid_shapes_and_window_boundary_are_rejected():
    x, y, starts = _synthetic_inputs()
    with pytest.raises(ValueError, match="X_test"):
        final._validate_test_inputs(x[:, :, :71], y, starts)
    with pytest.raises(ValueError, match="fully contained"):
        final._validate_test_inputs(x, y, [11455, 11456])
    with pytest.raises(ValueError, match="nonfinite"):
        bad = x.copy()
        bad[0, 0, 0] = np.nan
        final._validate_test_inputs(bad, y, starts)


def test_baselines_have_expected_shapes_and_formulas():
    x, _y, _starts = _synthetic_inputs()
    predicted = final._baseline_predictions(x)
    assert set(predicted) == {"Persistence", "EWMA alpha=0.8", "AR(5) legacy"}
    assert all(value.shape == (2, 72) for value in predicted.values())
    np.testing.assert_array_equal(predicted["Persistence"], x[:, -1, :])
    weights = np.asarray([.8 * .2 ** (4 - i) for i in range(5)], dtype=np.float64)
    weights /= weights.sum()
    np.testing.assert_allclose(predicted["EWMA alpha=0.8"], np.tensordot(x.astype(np.float64), weights, axes=([1], [0])))
    np.testing.assert_allclose(
        predicted["AR(5) legacy"],
        np.einsum("nte,t->ne", x.astype(np.float64), np.asarray(final.AR_COEFFICIENTS)) + final.AR_INTERCEPT,
    )

