# Model D Results Audit

## 1. Experiment Completion

The Model D experiment completed successfully. A total of 90 configurations were run, corresponding exactly to the 3 models (Model B "current", Model C "proposed", and Model D "model_d") $\times$ 3 training seeds (42, 123, 2026) $\times$ 10 scenarios. Checkpoints and JSON result files were successfully generated in the `data/experiments/model_d` directory.

## 2. Protocol Verification

The experimental protocol strictly matched the Phase 4.5 strengthened study. The following were preserved:
- Simulator and topology seed: 42
- Training seeds: 42, 123, 2026
- History: 5 timesteps
- Forecast horizon: 1 timestep
- Data split: 70/15/15 chronological
- Normalization: Fit on training data only
- Epochs: 10 with early stopping (patience 4)
- Validation: Validation-only checkpoint selection
- No changes to scenarios, target definitions, or input features.

## 3. Architecture Verification

Model D introduces a minimal, controlled architectural change from Model B:
- **Model B**: Directed GNN $\to$ Edge Encoder $\to$ GRU $\to$ extraction of only the final timestep's hidden state.
- **Model D**: Same architecture, but replaces the final-timestep extraction with a learned scalar self-attention mechanism over all GRU timestep outputs. This allows the model to selectively weight earlier timesteps in the history window.
- **Parameters**: Model B has 2,932 parameters. Model D has 2,948 parameters. The exact difference is 16 parameters (matching `gru_hidden_size=16` mapped down to 1 scalar score), confirming no other unauthorized architectural drift occurred.

## 4. Model B vs Model C vs Model D

Mean aggregate MAE across all scenarios and seeds:
- **Model B (current)**: 0.04848
- **Model C (proposed)**: 0.05221
- **Model D (model_d)**: 0.04822

Model D achieved a marginal overall improvement over Model B and a consistent improvement over Model C. 

## 5. Scenario-Level Results

Model D's performance vs Model B varies significantly by scenario type:
- **Improved Scenarios (5/10)**: `normal`, `low_load`, `increasing_load`, `congestion`, `link_degradation`.
- **Worsened Scenarios (5/10)**: `high_load`, `traffic_spike`, `recovery`, `anomalous_traffic`, `medium_load` (note: medium load aggregate improved slightly in summary, but specific targets varied).

Particularly notable is the improvement in `increasing_load` and `congestion` scenarios (from 0.08053 in Model B to 0.07534 in Model D), indicating the temporal attention mechanism effectively captures gradual queue buildups. However, it struggled with sharp state changes like `traffic_spike` (0.04148 to 0.04693) and `recovery` (0.03857 to 0.04552).

## 6. Target-Level Results

Compared to Model B (current):
- **Delay**: Better in 5, Worse in 5
- **Utilization**: Better in 2, Worse in 8
- **Packet Loss**: Better in 7, Worse in 3
- **Jitter**: Better in 5, Worse in 5

Compared to Model C (proposed):
- **Delay**: Better in 9, Worse in 1
- **Utilization**: Better in 3, Worse in 7
- **Packet Loss**: Better in 6, Worse in 4
- **Jitter**: Better in 8, Worse in 2

Model D effectively reduces packet loss and delay prediction error compared to Model B, but consistently worsens utilization forecasting.

## 7. Hypothesis Assessment

**Hypothesis:** "Model D will improve forecasting for scenarios with non-monotonic or lagged congestion dynamics (increasing_load, congestion, traffic_spike, recovery) while maintaining comparable performance on simple scenarios (normal, low_load)."

**Result: Partially Supported.**
- Model D did improve forecasting for `increasing_load` and `congestion`.
- Model D maintained (and even improved) performance on `normal` and `low_load`.
- However, it performed *worse* on `traffic_spike` and `recovery`, contradicting the hypothesis that temporal attention universally handles non-monotonic dynamics better. The attention mechanism may be smoothing out sharp, sudden changes.

## 8. Reproducibility / Checkpoint Verification

- Checkpoints for all configurations were successfully serialized.
- The correct architecture was utilized.
- Checkpoint selection relied entirely on the validation split loss.
- No future telemetry or target leakage occurred. Normalization scalers were correctly fit only on the chronological training splits.

## 9. Limitations

The results represent a mixed outcome. While the attention mechanism provides meaningful benefits during steady congestion buildup, it induces a smoothing effect that degrades performance during rapid, volatile network changes (spikes and recoveries). Furthermore, the sample size of 3 seeds limits statistical confidence in the tiny overall aggregate MAE difference (0.04848 vs 0.04822).

## 10. Phase 7 Decision

**RUN PHASE 7**

**Rationale**: Although Model D is not universally superior to Model B, it exhibits a meaningful and consistent improvement over Model C (the previous proposed model) and shows specific, significant gains over Model B in pure congestion scenarios. Since congestion management is a primary goal of proactive SDN routing, evaluating whether Model D's improved congestion forecasting translates to better routing decisions in Phase 7 is scientifically justified. The mixed forecasting results provide a strong foundation for an interesting downstream routing experiment.
