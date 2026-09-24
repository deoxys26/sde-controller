# Phase 7 Model D Routing Evaluation

## 1. Research Question
Does the forecasting performance improvement observed in Model D (temporal self-attention GRU) for congestion-related scenarios translate into better downstream routing decisions compared to the baseline predicted-QoS GA (using Model B)?

## 2. Model D Motivation
Model D introduces a minimal architectural change: replacing Model B's reliance on only the final GRU timestep output with a learned scalar self-attention mechanism over all historical GRU timestep outputs. The Phase 4.5 evaluation showed Model D improved forecasting MAE specifically during steady congestion buildup (`increasing_load`, `congestion`), prompting this controlled Phase 7 routing evaluation to see if better congestion prediction leads to better path selection.

## 3. Experimental Protocol
- **Evaluator**: `scripts.run_phase7_model_d_experiment.py` (an extension of the original Phase 7 causal evaluator).
- **Environment**: Same simulator seed (42), topology seed (42), and base topology configuration.
- **Scenarios**: All 10 existing scenarios evaluated under identical sequential demand generation.
- **Methods Compared**:
  - `shortest_path` (Static baseline)
  - `reactive` (Reactive baseline)
  - `current_qos_ga` (Telemetry from $t-1$)
  - `predicted_qos_ga_current` (Model B checkpoint, seed 42)
  - `predicted_qos_ga_model_d` (Model D checkpoint, seed 42)
- **Causality**: Strict adherence to causal evaluation. Forecasts were generated sequentially using only prior completed telemetry. No future leakage occurred.

## 4. Controls
- The Genetic Algorithm routing parameters remained exactly identical for all methods (population, generations, probabilities, capacity constraints).
- Checkpoints used were identical to those generated in the Phase 4.5 evaluation.
- No model retraining occurred during this phase.

## 5. Results
**Aggregate across all scenarios:**
- **Model B (Current)**: Delay 2.0117, Utilization 0.1113, Packet Loss 0.0031
- **Model D**: Delay 2.0257, Utilization 0.1214, Packet Loss 0.0052, Failure Rate 0.0067

Model D performed worse on aggregate across all major routing metrics, and even introduced a non-zero routing failure rate (2 failed flows).

## 6. Model D vs Existing Predicted-QoS GA
Overall, the existing Model B-powered `predicted_qos_ga` outperformed the Model D-powered variant in routing outcomes. Model D showed higher average delay, higher average utilization, and higher packet loss. 

## 7. Scenario-Level Behavior
- **Improved Scenarios**: Model D only improved routing metrics over Model B in the `high_load` scenario (Delay: 1.9980 vs 2.0109; Packet Loss: 0.0000 vs 0.0024).
- **Worsened Scenarios**: Model D performed worse in almost all other scenarios, including `increasing_load`, `congestion`, `link_degradation`, and `normal`.

## 8. Did Forecasting Improvements Transfer to Routing?
**No.** The forecasting gains observed for Model D in Phase 4.5 (specifically in `increasing_load` and `congestion`) did *not* translate to better downstream routing. In fact, routing performance in `increasing_load` and `congestion` significantly degraded (Packet loss doubled from 0.0145 to 0.0250). 

Furthermore, the weaker forecasting performance Model D exhibited on sharp state changes (`traffic_spike`, `recovery`) also translated to poorer (or identical but suboptimal) routing outcomes.

## 9. Limitations
- The temporal attention mechanism in Model D may be "smoothing" the forecasting outputs in a way that minimizes overall MAE during gradual congestion but fails to preserve sharp threshold crossings that the GA requires to correctly penalize congested links.
- Routing metrics are highly sensitive to specific constraint boundaries (e.g., link capacity). A model with a slightly better MAE might still mispredict a critical capacity boundary, leading to poor GA path selection.

## 10. Conclusion and Pipeline Decision
**Do NOT integrate Model D into the final research pipeline.** 
While Model D demonstrated a plausible forecasting hypothesis (better modeling of delayed congestion via temporal attention), the downstream Phase 7 causal routing evaluation unequivocally shows it leads to poorer routing decisions and introduces flow failures. The original Model B remains the superior, more robust choice for the final predicted-QoS GA pipeline.
