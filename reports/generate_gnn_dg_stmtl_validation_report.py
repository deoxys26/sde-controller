"""Generate a descriptive report from the metric values supplied for this study.

This script deliberately reads no experiment artifacts. All source metrics are
embedded below, and output files are created only when their exact destinations
do not already exist.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EWMA = {"mae": 0.006502658485291489, "rmse": 0.017641687598330374}
DG_RECORDED_EWMA_MAE = 0.006502658393768188

RUNS = [
    {"method": "GNN", "adjacency": "Topology", "seed": 42, "mae": 0.006164560900547242, "rmse": 0.01637964511169927, "runtime_seconds": None, "manifest_status": "metric previously recorded; individual run.json unavailable"},
    {"method": "GNN", "adjacency": "Topology", "seed": 123, "mae": 0.005960349072666915, "rmse": 0.01611846669036514, "runtime_seconds": None, "manifest_status": "reported metric"},
    {"method": "GNN", "adjacency": "Topology", "seed": 2026, "mae": 0.006132490537406109, "rmse": 0.016451684668496527, "runtime_seconds": None, "manifest_status": "reported metric"},
    {"method": "GNN", "adjacency": "Identity", "seed": 42, "mae": 0.00596160563625111, "rmse": 0.01629732022336022, "runtime_seconds": None, "manifest_status": "reported metric"},
    {"method": "GNN", "adjacency": "Identity", "seed": 123, "mae": 0.005859218496394163, "rmse": 0.016333039148517726, "runtime_seconds": None, "manifest_status": "reported metric"},
    {"method": "GNN", "adjacency": "Identity", "seed": 2026, "mae": 0.0060226222285615855, "rmse": 0.01639765587133747, "runtime_seconds": None, "manifest_status": "reported metric"},
    {"method": "GNN", "adjacency": "Shuffled", "seed": 42, "mae": 0.006164661037724068, "rmse": 0.016287520552661736, "runtime_seconds": None, "manifest_status": "reported metric"},
    {"method": "GNN", "adjacency": "Shuffled", "seed": 123, "mae": 0.0059590808181347946, "rmse": 0.01632711503963857, "runtime_seconds": None, "manifest_status": "reported metric"},
    {"method": "GNN", "adjacency": "Shuffled", "seed": 2026, "mae": 0.006182522989717427, "rmse": 0.01662989435032721, "runtime_seconds": None, "manifest_status": "reported metric"},
    {"method": "DG-STMTL-inspired", "adjacency": "Hybrid static/dynamic", "seed": 42, "mae": 0.00665609501657303, "rmse": 0.017735000431979193, "runtime_seconds": 332.402790654, "manifest_status": "completed; backup reported read-back verified"},
    {"method": "DG-STMTL-inspired", "adjacency": "Hybrid static/dynamic", "seed": 123, "mae": 0.006721429628146952, "rmse": 0.017765220881943874, "runtime_seconds": 291.214151643, "manifest_status": "completed; backup reported read-back verified"},
    {"method": "DG-STMTL-inspired", "adjacency": "Hybrid static/dynamic", "seed": 2026, "mae": 0.006550973042994777, "rmse": 0.017715223441816565, "runtime_seconds": 345.8363319, "manifest_status": "completed; backup reported read-back verified"},
]

GROUPS = {
    "Topology GNN (historical n=3)": [r for r in RUNS if r["method"] == "GNN" and r["adjacency"] == "Topology"],
    "Identity GNN": [r for r in RUNS if r["method"] == "GNN" and r["adjacency"] == "Identity"],
    "Shuffled GNN": [r for r in RUNS if r["method"] == "GNN" and r["adjacency"] == "Shuffled"],
    "DG-STMTL-inspired": [r for r in RUNS if r["method"] == "DG-STMTL-inspired"],
}


def summarize(rows: Iterable[dict], metric: str) -> tuple[float, float, int]:
    values = [float(row[metric]) for row in rows]
    return statistics.mean(values), statistics.stdev(values), len(values)


def improvement_percent(baseline: float, model: float) -> float:
    return 100.0 * (baseline - model) / baseline


def topology_two_seed_summary(metric: str) -> tuple[float, float, int]:
    rows = [r for r in GROUPS["Topology GNN (historical n=3)"] if r["seed"] in (123, 2026)]
    return summarize(rows, metric)


def _fmt(value: float, digits: int = 8) -> str:
    return f"{value:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report() -> str:
    aggregate_rows: list[list[str]] = []
    for name, rows in GROUPS.items():
        mae_mean, mae_sd, n = summarize(rows, "mae")
        rmse_mean, rmse_sd, _ = summarize(rows, "rmse")
        aggregate_rows.append([
            name,
            str(n),
            f"{_fmt(mae_mean)} ± {_fmt(mae_sd)}",
            f"{_fmt(rmse_mean)} ± {_fmt(rmse_sd)}",
            f"{improvement_percent(EWMA['mae'], mae_mean):+.3f}%",
            f"{improvement_percent(EWMA['rmse'], rmse_mean):+.3f}%",
            "52,513" if name == "DG-STMTL-inspired" else "not supplied",
            f"{statistics.mean(float(r['runtime_seconds']) for r in rows):.3f}" if rows[0]["runtime_seconds"] is not None else "not supplied",
        ])

    run_rows = []
    for row in RUNS:
        run_rows.append([
            row["method"], row["adjacency"], str(row["seed"]),
            _fmt(row["mae"], 12), _fmt(row["rmse"], 12),
            f"{improvement_percent(EWMA['mae'], row['mae']):+.3f}%",
            f"{improvement_percent(EWMA['rmse'], row['rmse']):+.3f}%",
            "52,513" if row["method"] == "DG-STMTL-inspired" else "not supplied",
            "not supplied" if row["runtime_seconds"] is None else f"{row['runtime_seconds']:.3f}",
        ])
    run_rows.append(["EWMA", "alpha=0.8, 5 observations", "—", _fmt(EWMA["mae"], 12), _fmt(EWMA["rmse"], 12), "0.000%", "0.000%", "0 trainable", "not supplied"])

    two_mae, two_mae_sd, two_n = topology_two_seed_summary("mae")
    two_rmse, two_rmse_sd, _ = topology_two_seed_summary("rmse")
    topo_mae = summarize(GROUPS["Topology GNN (historical n=3)"], "mae")[0]
    dg_mae = summarize(GROUPS["DG-STMTL-inspired"], "mae")[0]
    topo_rmse = summarize(GROUPS["Topology GNN (historical n=3)"], "rmse")[0]
    dg_rmse = summarize(GROUPS["DG-STMTL-inspired"], "rmse")[0]
    dg_runtime = statistics.mean(float(r["runtime_seconds"]) for r in GROUPS["DG-STMTL-inspired"])
    best_mae = min(RUNS, key=lambda r: r["mae"])
    best_rmse = min(RUNS, key=lambda r: r["rmse"])
    ranked_mae = sorted(GROUPS.items(), key=lambda item: summarize(item[1], "mae")[0])
    ranked_rmse = sorted(GROUPS.items(), key=lambda item: summarize(item[1], "rmse")[0])
    topo_rows = {r["seed"]: r for r in GROUPS["Topology GNN (historical n=3)"]}
    dg_rows = {r["seed"]: r for r in GROUPS["DG-STMTL-inspired"]}
    topo_dg_rows = [[
        str(seed),
        f"{improvement_percent(dg_rows[seed]['mae'], topo_rows[seed]['mae']):+.3f}%",
        f"{improvement_percent(dg_rows[seed]['rmse'], topo_rows[seed]['rmse']):+.3f}%",
        "historical Topology metric; manifest unavailable" if seed == 42 else "reported metric",
    ] for seed in (42, 123, 2026)]

    return f"""# GNN vs DG-STMTL-inspired vs EWMA: validation comparison

## Executive summary

This is a descriptive synthesis of previously recorded validation metrics supplied for this report; it is not a newly executed or provenance-verified benchmark. Across the reported three-seed averages, Identity GNN has the lowest MAE, while Topology GNN has the lowest RMSE. The DG-STMTL-inspired adaptation is worse than the recorded EWMA baseline on both average metrics. Identity GNN seed 123 has the lowest individual MAE; Topology GNN seed 123 has the lowest individual RMSE. Both are below EWMA on their reported validation metrics. These observations do not establish statistical significance or test-set generalization.

The mean Topology GNN improvement versus EWMA is {improvement_percent(EWMA['mae'], topo_mae):.3f}% MAE and {improvement_percent(EWMA['rmse'], topo_rmse):.3f}% RMSE using all three historical values; seed 42's individual run manifest is unavailable in the accessible backups. On the independently available seeds 123 and 2026 only, Topology GNN averages MAE {_fmt(two_mae)} (sample SD {_fmt(two_mae_sd)}, n={two_n}) and RMSE {_fmt(two_rmse)} (sample SD {_fmt(two_rmse_sd)}, n={two_n}).

## Dataset and experimental setup

- Network representation: GEANT, 22 nodes and 72 directed links.
- Target: estimated link utilization derived from GEANT origin-destination traffic matrices; these are not measured link-level QoS observations.
- Sampling interval: 15 minutes; history length: 672 observations (7 days); horizon: 96 observations (24 hours ahead).
- Recorded data counts: 3,924 training windows and 953 validation windows; expected validation coverage is 68,616 window-link pairs (953 × 72).
- Split: chronological train/validation split. The official test split was sealed and not evaluated for the results summarized here.
- Metrics: the supplied validation MAE and RMSE values are reported in utilization-ratio units. No uncertainty interval is calculated here.

## Models compared

- **Original GNN:** EdgeQoS/GNN result groups with Topology, Identity, or Shuffled adjacency. The graph variants are presented as recorded; they are not averaged together.
- **DG-STMTL-inspired adaptation:** hybrid static/dynamic adjacency, gating, and grouped spatio-temporal processing adapted to single-target link-utilization forecasting, with an EWMA residual head. It is an adaptation, not an exact reproduction of the published DG-STMTL paper. The stated model size is 52,513 parameters.
- **EWMA:** fixed alpha 0.8 over the five observations available to each forecast. The canonical original-GNN evaluation baseline is used in comparison calculations: MAE {_fmt(EWMA['mae'], 15)}, RMSE {_fmt(EWMA['rmse'], 15)}.

The DG-STMTL runs recorded EWMA MAE {_fmt(DG_RECORDED_EWMA_MAE, 15)}, differing from the canonical value by {abs(DG_RECORDED_EWMA_MAE - EWMA['mae']):.3g}. The recorded values are preserved; for the summary percentage comparisons, the single canonical EWMA value above is used unchanged. No EWMA error bars are shown because only one baseline aggregate was supplied.

## Complete per-seed results

Improvement is calculated as 100 × (EWMA error − model error) / EWMA error. Positive values indicate lower error than EWMA. The EWMA row is the same fixed baseline, not a seed-specific observation.

{_markdown_table(['Method', 'Adjacency', 'Seed', 'MAE', 'RMSE', 'MAE improvement vs EWMA', 'RMSE improvement vs EWMA', 'Parameters', 'Training seconds'], run_rows)}

## Three-seed aggregate comparison

Values are mean ± sample standard deviation (ddof=1) across the listed seeds; these are descriptive seed-to-seed summaries, not formal confidence intervals. Topology's n=3 historical summary includes seed 42's previously recorded values despite its missing individual run manifest. The other neural groups have n=3 reported runs.

{_markdown_table(['Model group', 'n', 'MAE mean ± SD', 'RMSE mean ± SD', 'MAE improvement vs EWMA', 'RMSE improvement vs EWMA', 'Parameters', 'Mean training seconds'], aggregate_rows)}

Ranking by mean MAE (lower is better): {" → ".join(name for name, _ in ranked_mae)}. Ranking by mean RMSE (lower is better): {" → ".join(name for name, _ in ranked_rmse)}.

Topology-versus-DG-STMTL-inspired relative difference, defined as 100 × (DG mean − Topology mean) / DG mean: MAE {improvement_percent(dg_mae, topo_mae):+.3f}%; RMSE {improvement_percent(dg_rmse, topo_rmse):+.3f}%. Positive means the Topology mean is lower. This is descriptive and does not compare the published DG-STMTL model.

Paired-seed Topology-versus-DG-STMTL-inspired differences use the same formula; positive values favor the Topology GNN:

{_markdown_table(['Seed', 'Topology MAE reduction vs DG', 'Topology RMSE reduction vs DG', 'Evidence note'], topo_dg_rows)}

## Two-seed manifest-available Topology sensitivity comparison

The following sensitivity summary excludes Topology seed 42 because its individual `run.json` is missing from accessible backups. It must not be described as the full three-seed Topology mean.

{_markdown_table(['Model group', 'Seeds', 'n', 'MAE mean ± SD', 'RMSE mean ± SD', 'MAE improvement vs EWMA', 'RMSE improvement vs EWMA'], [["Topology GNN (available manifests)", "123, 2026", str(two_n), f"{_fmt(two_mae)} ± {_fmt(two_mae_sd)}", f"{_fmt(two_rmse)} ± {_fmt(two_rmse_sd)}", f"{improvement_percent(EWMA['mae'], two_mae):+.3f}%", f"{improvement_percent(EWMA['rmse'], two_rmse):+.3f}%"]])}

## Best individual runs and runtime

- Best individual MAE among the supplied neural runs: {best_mae['method']} / {best_mae['adjacency']} / seed {best_mae['seed']}, MAE {_fmt(best_mae['mae'], 12)}.
- Best individual RMSE among the supplied neural runs: {best_rmse['method']} / {best_rmse['adjacency']} / seed {best_rmse['seed']}, RMSE {_fmt(best_rmse['rmse'], 12)}.
- Mean DG-STMTL-inspired training runtime: {dg_runtime:.3f} seconds across three runs. No comparable GNN runtime values were supplied, so runtime ranking is unavailable.

## Graph interpretation

Identity GNN achieved lower three-seed mean MAE than Topology GNN, while Topology GNN achieved lower three-seed mean RMSE. Thus Topology does not dominate every GNN variant on every metric. These results alone do not identify why the adjacency variants differ, and they do not establish that network structure is or is not useful in general.

The selected Topology GNN seed 123 result (MAE 0.0059603491; RMSE 0.0161184667) is below the supplied EWMA values on validation. This is one selected seed among reported runs; it is not evidence of a generalizable advantage.

## Why the DG-STMTL-inspired adaptation may have underperformed

All three supplied adaptation runs selected their best checkpoint at epoch 1 and their validation errors were slightly above EWMA. Plausible explanations include optimization behavior, the residual-head initialization or learning dynamics, model complexity relative to the available windows, and characteristics of this estimated-utilization task. None of these explanations has been experimentally isolated here; they are hypotheses, not findings.

## Implications for choosing a forecasting model for the GA

If a downstream GA needs a provisional validation-scored forecast, these results can inform a candidate shortlist, but they do not demonstrate routing benefit. Estimated utilization forecasts are not evidence of improved latency, packet loss, security, or GA outcomes. Any integration decision should be evaluated separately under a fixed and comparable protocol, and validation-based selection must remain distinct from final-test claims.

## Limitations and reproducibility status

- This is a descriptive comparison of supplied historical numbers, not a newly executed provenance-verified benchmark.
- The strict Colab manifest comparator did not produce a validated report because of metadata/provenance compatibility checks.
- Original Topology GNN seed 42 metrics were previously recorded, but its individual `run.json` is missing from accessible backups; the historical three-seed Topology aggregate therefore has that verification caveat. The n=2 sensitivity result reports seeds 123 and 2026 only.
- The DG-STMTL implementation is an architectural adaptation, not an exact reproduction of the paper.
- Original GNN parameter counts were not supplied; they are marked unavailable rather than inferred. The adaptation's supplied count is 52,513 parameters.
- Validation performance is not sealed-test performance. No sealed test evaluation was performed for this report.
- No measured latency comparison, packet-loss improvement, security improvement, or GA routing benefit has been established.
- The reported best seed must not be interpreted as a predeclared model-selection result. Final-test/model-selection claims remain separate.
- Three seeds do not justify statistical-significance claims; no confidence intervals are fabricated.
- These results do not demonstrate superiority over the published DG-STMTL model.

## Conclusion

On the supplied GEANT 15-minute estimated link-utilization validation results, Identity GNN has the lowest mean MAE, Topology GNN has the lowest mean RMSE, and the DG-STMTL-inspired adaptation is worse than EWMA on both mean metrics. Identity seed 123 has the best individual MAE, while Topology seed 123 has the best individual RMSE; each beats EWMA on its respective reported metric. The missing Topology seed-42 manifest limits independent verification of its full three-seed aggregate. The evidence is useful for documenting these runs, but it is not a provenance-verified head-to-head benchmark, does not establish test-set generalization, and does not establish downstream GA benefit.
"""


def _plot_aggregate(metric: str, title: str, ylabel: str, destination: Path) -> None:
    short_labels = ["Topology\nGNN (n=3*)", "Identity\nGNN (n=3)", "Shuffled\nGNN (n=3)", "DG-STMTL\nadaptation (n=3)"]
    labels = list(GROUPS)
    means, errors, counts = [], [], []
    for name in labels:
        mean, sd, n = summarize(GROUPS[name], metric)
        means.append(mean)
        errors.append(sd)
        counts.append(n)
    labels = short_labels
    labels.append("EWMA\n(single baseline)")
    means.append(EWMA[metric])
    errors.append(0.0)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = list(range(len(labels)))
    ax.bar(x, means, color=["#4472C4", "#70AD47", "#A5A5A5", "#ED7D31", "#5B9BD5"])
    ax.errorbar(x[:-1], means[:-1], yerr=errors[:-1], fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.01, -0.22, "Bars show mean; neural whiskers are ±1 sample SD (descriptive, not CI). EWMA has no error bar. *Topology includes previously recorded seed 42; its run manifest is unavailable.", transform=ax.transAxes, fontsize=8.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_per_seed_rmse(destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    groups = [("Topology GNN", "Topology"), ("Identity GNN", "Identity"), ("Shuffled GNN", "Shuffled"), ("DG-STMTL-inspired", "Hybrid static/dynamic")]
    seeds = [42, 123, 2026]
    width = 0.19
    for index, (label, adjacency) in enumerate(groups):
        rows = [r for r in RUNS if r["adjacency"] == adjacency]
        values = [next(r["rmse"] for r in rows if r["seed"] == seed) for seed in seeds]
        ax.bar([s + (index - 1.5) * width for s in range(len(seeds))], values, width, label=label)
    ax.axhline(EWMA["rmse"], color="black", linestyle="--", linewidth=1.2, label="EWMA baseline")
    ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Validation RMSE (estimated utilization ratio)")
    ax.set_title("Per-seed validation RMSE")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.12), ncols=3, frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.01, 0.015, "Topology seed 42 is a previously recorded value; its individual run manifest is unavailable.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_improvement(destination: Path) -> None:
    labels, means, errors = [], [], []
    for name, rows in GROUPS.items():
        values = [improvement_percent(EWMA["mae"], r["mae"]) for r in rows]
        labels.append(name)
        means.append(statistics.mean(values))
        errors.append(statistics.stdev(values))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#4472C4" if value > 0 else "#ED7D31" for value in means]
    ax.bar(range(len(labels)), means, color=colors)
    ax.errorbar(range(len(labels)), means, yerr=errors, fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(labels)), ["Topology GNN\n(n=3*)", "Identity GNN\n(n=3)", "Shuffled GNN\n(n=3)", "DG-STMTL adapt.\n(n=3)"])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylabel("MAE improvement vs EWMA (%)")
    ax.set_title("Relative validation MAE improvement versus EWMA")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.01, -0.24, "Positive means lower MAE than EWMA. Whiskers are ±1 sample SD across seeds, not CI. *Topology includes previously recorded seed 42; its run manifest is unavailable.", transform=ax.transAxes, fontsize=8.1)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path) -> None:
    fields = ["record_type", "method", "adjacency", "seed", "n", "mae", "rmse", "mae_sd", "rmse_sd", "mae_improvement_vs_ewma_pct", "rmse_improvement_vs_ewma_pct", "topology_mae_reduction_vs_dg_pct", "topology_rmse_reduction_vs_dg_pct", "parameter_count", "runtime_seconds", "runtime_mean_seconds", "verification_note"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in RUNS:
            writer.writerow({
                "record_type": "run", "method": row["method"], "adjacency": row["adjacency"], "seed": row["seed"], "n": 1,
                "mae": format(row["mae"], ".17g"), "rmse": format(row["rmse"], ".17g"),
                "mae_improvement_vs_ewma_pct": format(improvement_percent(EWMA["mae"], row["mae"]), ".17g"),
                "rmse_improvement_vs_ewma_pct": format(improvement_percent(EWMA["rmse"], row["rmse"]), ".17g"),
                "parameter_count": "52513" if row["method"] == "DG-STMTL-inspired" else "not supplied",
                "runtime_seconds": "" if row["runtime_seconds"] is None else format(row["runtime_seconds"], ".17g"),
                "verification_note": row["manifest_status"],
            })
        for name, rows in GROUPS.items():
            mae, mae_sd, n = summarize(rows, "mae")
            rmse, rmse_sd, _ = summarize(rows, "rmse")
            writer.writerow({
                "record_type": "aggregate", "method": name, "n": n,
                "mae": format(mae, ".17g"), "rmse": format(rmse, ".17g"),
                "mae_sd": format(mae_sd, ".17g"), "rmse_sd": format(rmse_sd, ".17g"),
                "mae_improvement_vs_ewma_pct": format(improvement_percent(EWMA["mae"], mae), ".17g"),
                "rmse_improvement_vs_ewma_pct": format(improvement_percent(EWMA["rmse"], rmse), ".17g"),
                "parameter_count": "52513" if name == "DG-STMTL-inspired" else "not supplied",
                "runtime_mean_seconds": "" if rows[0]["runtime_seconds"] is None else format(statistics.mean(float(r["runtime_seconds"]) for r in rows), ".17g"),
            })
        writer.writerow({"record_type": "baseline", "method": "EWMA", "adjacency": "alpha=0.8; observations=5", "n": 1, "mae": format(EWMA["mae"], ".17g"), "rmse": format(EWMA["rmse"], ".17g"), "parameter_count": "0 trainable", "verification_note": "single supplied validation baseline; no seed SD or training runtime"})
        topology = {r["seed"]: r for r in GROUPS["Topology GNN (historical n=3)"]}
        dg = {r["seed"]: r for r in GROUPS["DG-STMTL-inspired"]}
        for seed in (42, 123, 2026):
            writer.writerow({
                "record_type": "paired_topology_vs_dg", "method": "Topology GNN vs DG-STMTL-inspired", "seed": seed,
                "topology_mae_reduction_vs_dg_pct": format(improvement_percent(dg[seed]["mae"], topology[seed]["mae"]), ".17g"),
                "topology_rmse_reduction_vs_dg_pct": format(improvement_percent(dg[seed]["rmse"], topology[seed]["rmse"]), ".17g"),
                "verification_note": "seed 42 topology value previously recorded; individual manifest unavailable" if seed == 42 else "reported metrics",
            })


def generate(output_dir: Path, *, replace_existing_outputs: bool = False) -> list[Path]:
    output_dir = output_dir.absolute()
    figure_dir = output_dir / "figures"
    outputs = [
        output_dir / "gnn_vs_dg_stmtl_validation_comparison.md",
        output_dir / "gnn_vs_dg_stmtl_validation_comparison.csv",
        figure_dir / "validation_mae_comparison.png",
        figure_dir / "validation_rmse_comparison.png",
        figure_dir / "per_seed_rmse.png",
        figure_dir / "improvement_vs_ewma.png",
    ]
    existing = [p for p in outputs if p.exists()]
    if existing and not replace_existing_outputs:
        raise FileExistsError("Refusing to overwrite existing report artifact(s): " + ", ".join(str(p) for p in existing))
    if replace_existing_outputs and existing and len(existing) != len(outputs):
        raise FileExistsError("Refusing partial replacement; expected all six known report outputs to exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gnn_vs_dg_stmtl_validation_comparison.md").write_text(build_report(), encoding="utf-8")
    write_csv(output_dir / "gnn_vs_dg_stmtl_validation_comparison.csv")
    _plot_aggregate("mae", "Mean validation MAE by model", "Validation MAE (estimated utilization ratio)", outputs[2])
    _plot_aggregate("rmse", "Mean validation RMSE by model", "Validation RMSE (estimated utilization ratio)", outputs[3])
    _plot_per_seed_rmse(outputs[4])
    _plot_improvement(outputs[5])
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="destination for new report files (existing outputs are never overwritten)")
    parser.add_argument("--replace-existing-outputs", action="store_true", help="regenerate only when all six expected report outputs already exist")
    args = parser.parse_args()
    generated = generate(args.output_dir, replace_existing_outputs=args.replace_existing_outputs)
    for path in generated:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
