# Final Research Conclusion

## 1. The Original Research Problem
The core problem addressed by this project is the inability of reactive Software-Defined Networking (SDN) routing protocols to handle sudden congestion, traffic spikes, and network anomalies efficiently. Traditional reactive routing waits for QoS degradation (e.g., packet loss or high delay) to occur before rerouting flows, leading to suboptimal performance during the reaction window. The goal was to investigate whether proactive routing—using machine learning to forecast impending QoS degradation before it occurs—can improve overall network performance.

## 2. Implementation Overview
We implemented a comprehensive, simulated research pipeline in Python:
* **Synthetic SDN Simulator**: Generates topologies, continuous background traffic, and highly specific network scenarios (e.g., `traffic_spike`, `link_degradation`).
* **GNN Forecaster**: Models the network as a directed graph. The baseline Model B uses a combination of a Graph Convolutional Network (GCN) and a Gated Recurrent Unit (GRU) to encode spatiotemporal telemetry and predict $t+1$ QoS targets (delay, utilization, packet loss, jitter).
* **GA Router**: A multi-objective Genetic Algorithm that searches for optimal routing paths balancing delay, packet loss, capacity constraints, and a conceptual security risk.
* **Causal Evaluator**: An end-to-end routing evaluation environment that rigorously prevents future data leakage.

## 3. What Model B Demonstrated
Model B (`EdgeQoSForecaster`), utilizing a simple extraction of the final timestep from a GRU sequence, demonstrated strong baseline forecasting performance across various network scenarios. Its predictions were sufficiently accurate to be used as a proactive telemetry feed for downstream routing decisions.

## 4. What Model C Demonstrated
Model C (`DirectedCongestionDGSTMTLForecaster`) was proposed based on state-of-the-art literature (DG-STMTL). It utilized complex congestion-conditioned adjacency matrices and spatial pooling. However, the rigorous multi-seed evaluation in Phase 4.5 demonstrated that Model C actually performed *worse* than the simpler Model B on this specific SDN dataset, struggling with the preservation of directed temporal dependencies and introducing higher MAE. Model C served as an important lesson against blindly applying complex models without domains-specific validation.

## 5. What Model D Demonstrated
Model D sought to improve Model B by adding a learned scalar temporal self-attention mechanism over all GRU timestep outputs. The hypothesis was that this would better capture lagged, gradual congestion. While Model D did show slightly improved forecasting MAE specifically during steady congestion (`increasing_load`, `congestion`), it performed worse during sharp state changes (`traffic_spike`, `recovery`), indicating an over-smoothing effect.

## 6. What Phase 7 Demonstrated
Under the evaluated synthetic SDN scenarios, the predicted-QoS GA provided a functioning proactive routing mechanism, but the experiments did not establish overall superiority over the shortest-path or reactive baselines. Model D produced a small improvement in aggregate forecasting MAE over Model B, but this improvement did not transfer to downstream routing performance. This indicates that aggregate forecasting accuracy alone is not sufficient to characterize the usefulness of predictions for routing decisions.

## 7. Did Better Forecasting Necessarily Improve Routing?
**No.** The downstream Model D evaluation provided a powerful negative result. Despite Model D exhibiting slightly better aggregate forecasting MAE (and notably better MAE in congestion scenarios), its use in the Phase 7 GA router resulted in higher actual packet loss, higher routing delay, and the introduction of unroutable flow failures. 

## 8. Lessons from the Model D Negative Result
The Model D result teaches us that in strictly constrained optimization problems like network routing, raw regression metrics (like MAE) are insufficient proxy metrics for success. The attention mechanism smoothed out sharp capacity thresholds; missing a threshold by a tiny margin is lightly penalized by MAE but catastrophically penalized by the GA (which routes traffic onto a link that subsequently overflows).

## 9. Current Limitations
The research was conducted exclusively in a simulated environment using fluid-flow mechanics rather than packet-level TCP mechanics. The topology size was fixed, and the evaluation horizon was limited. Additionally, while the UNSW-NB15 dataset was successfully modeled for anomaly detection, it remains separated from the QoS pipeline because mapping arbitrary offline flow datasets to a synthetic directed graph lacks rigorous scientific justification at this stage. 

## 10. Justified Claims
Based on the experiments, we can justifiably claim:
1. **Implementation Validation**: We successfully constructed a rigorous, causally sound SDN simulation, forecasting, and evaluation pipeline.
2. **Forecasting Evidence**: Spatiotemporal GNNs (specifically Model B) can accurately forecast near-term link-level QoS metrics in a simulated directed network.
3. **Routing Evidence**: Equipping a GA router with proactive Model B forecasts yielded a functioning proactive routing system with zero flow failures, though it did not outperform the simple reactive or shortest-path baselines on aggregate QoS metrics.
4. **Security-Interface Validation**: We validated the interface by which a theoretical risk score can successfully penalize paths in a GA, though the security scores remain synthetic.

We **do not** claim production readiness, real-world deployment validation, statistical superiority over all baseline methods, or state-of-the-art performance. Model B was retained as the primary forecasting model for the final routing pipeline because Model D's small forecasting improvement did not translate into improved downstream routing performance under the controlled Phase 7 evaluation.
