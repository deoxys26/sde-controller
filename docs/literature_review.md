# Literature Review

## Paper 1
1. **Exact paper title**: DG-STMTL: A Novel Graph Convolutional Network for Multi-Task Spatio-Temporal Traffic Forecasting
2. **Problem addressed**: Spatio-temporal multi-task traffic forecasting (e.g., simultaneously predicting traffic flow and speed, or pick-up and drop-off demand).
3. **Research motivation**: Traditional GCNs struggle with static adjacency matrices (which introduce domain bias) or fully learnable matrices (which overfit). In Multi-Task Learning, there is "task interference" where different tasks have conflicting objectives, requiring a framework that efficiently shares information without negative interference.
4. **Dataset(s)**: PEMSD4, PEMSD8, NYCFHV (New York City For-Hire-Vehicle).
5. **Dataset size where reported**: PEMSD4 (307 road detectors, Jan-Feb 2018), PEMSD8 (170 detectors, Jul-Aug 2016), NYCFHV (69 regions, Jan-Dec 2022).
6. **Network/topology/environment**: Represented as spatio-temporal graphs where nodes are road segments or segmented city regions.
7. **Input features**: Historical sequence data spanning the past 12 time steps for multi-task targets (e.g., flow and speed).
8. **Model architecture**: DG-STMTL (Dynamic Group-wise Spatio-Temporal Multi-Task Learning). Includes Hybrid Adjacency Matrix Generation (HAMG), Cross-Task Knowledge Exchange (CTKE), and Group-wise Spatio-Temporal Graph Convolutional (GSTGC) modules.
9. **Prediction target**: Traffic attributes (speed and flow, or pick-up and drop-off demand) for the next time interval.
10. **Routing/optimization method, if any**: Optimization is performed using the Adam optimizer with an early stopping mechanism for model loss (Smooth L1). No network routing optimization is discussed.
11. **Evaluation methodology**: Evaluated against historical multi-task predictive baselines across standard metrics on the mentioned datasets.
12. **Evaluation metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Average Percentage Error (MAPE).
13. **Main reported findings/results**: The framework efficiently shares information among tasks, mitigating negative interference and adapting to data dynamics, outperforming baseline predictive models.
14. **Limitations explicitly stated by the authors**: The model's performance plateaus or collapses if the network depth or hidden dimension exceeds optimal thresholds (e.g., 3 layers, 64 dimensions), showing susceptibility to overfitting and oversmoothing. The framework lacks interpretability in decision-making.
15. **Limitations that can reasonably be inferred, clearly labelled as inference**: *(Inference)* The reliance on historical sequence lengths and dynamic graph generation may incur high computational inference times, potentially hindering extremely low-latency real-time applications.
16. **Relevance to our proposed project**: Provides the foundational architecture concept (spatial-temporal GNN) for accurately forecasting multi-dimensional network metrics (QoS) using dynamic graphs.

## Paper 2
1. **Exact paper title**: Network Traffic Prediction Based on Graph Neural Network and Anomaly Detection
2. **Problem addressed**: Traditional network traffic prediction and anomaly detection methods fail to effectively capture complex spatial dependencies, dynamic topology changes, and high-dimensional anomalies.
3. **Research motivation**: Breaking the single-dimensional limitation of traditional methods by using spatio-temporal integration to realize topology-aware anomaly detection, improve identification accuracy, and optimize resource allocation.
4. **Dataset(s)**: Three datasets were utilized (names not explicitly specified in the text): Dataset 1 (network traffic dataset), Dataset 2 (traffic matrices and topology), Dataset 3 (various anomaly types with 5% anomaly injection).
5. **Dataset size where reported**: 5% anomaly injection rate in Dataset 3; absolute sample sizes not explicitly stated.
6. **Network/topology/environment**: Network topology is modeled as a graph structure where nodes represent network devices, edges represent connections, and an adjacency matrix represents connection weights.
7. **Input features**: Node-level attributes including inbound traffic, outbound traffic, bandwidth utilization, packet loss rate, and latency.
8. **Model architecture**: Graph Neural Network (GNN)-based framework utilizing a Spatiotemporal Graph Convolutional Network (ST-GCN), Adaptive Graph Convolution (GCN), Optimized Causal Time Convolution Networks (TCN), and Spatio-temporal Attention Mechanisms.
9. **Prediction target**: Forecasting future multidimensional traffic features across multiple time horizons (0.25h, 0.5h, 2h), and a binary classification label indicating whether an anomaly is present.
10. **Routing/optimization method, if any**: Joint optimization of prediction and detection tasks to assist intelligent decision-making. No explicit network routing protocol optimization is discussed.
11. **Evaluation methodology**: Assessed the model's forecasting and binary classification capability against baselines.
12. **Evaluation metrics**: MAE, RMSE, MAPE, R² (Prediction); Precision, Recall, F1-score, AUC (Anomaly Detection).
13. **Main reported findings/results**: Spatio-temporal integration realizes better topology-aware anomaly detection and prediction accuracy compared to single-dimensional or traditional static baselines.
14. **Limitations explicitly stated by the authors**: Not explicitly identified in the available paper text.
15. **Limitations that can reasonably be inferred, clearly labelled as inference**: *(Inference)* Jointly optimizing for a continuous prediction target and a discrete anomaly target using a shared latent representation can suffer from task interference if not meticulously balanced.
16. **Relevance to our proposed project**: Demonstrates that GNNs can successfully process node-level QoS features (bandwidth, loss, latency) to output both future metrics and security/anomaly indicators, which forms the input for our GA optimizer.

## Paper 3
1. **Exact paper title**: QoS-based Routing Framework for Efficient Path Selection using Genetic Algorithm in Software Defined Networks
2. **Problem addressed**: Traditional shortest-path routing in SDN optimises only for hop count, ignoring crucial QoS parameters (bandwidth, delay, loss), making multi-objective routing an NP-hard problem.
3. **Research motivation**: The rise of bandwidth-demanding applications requires consistent QoS that traditional distributed routing protocols (OSPF, RIP) cannot handle dynamically.
4. **Dataset(s)**: Data was generated via 100 randomly selected host-pair experiments.
5. **Dataset size where reported**: 100 iterations of host-pair traffic generation.
6. **Network/topology/environment**: Mininet 2.3 network emulator, Floodlight OpenFlow 1.3 controller, OVS software switches. 12-switch, 20-host diamond topology.
7. **Input features**: Core QoS parameters used for fitness: Minimum Bandwidth (Mbps), Cumulative Delay/Latency (ms), Compound Packet Loss (%).
8. **Model architecture**: Genetic Algorithm (GA) utilizing tournament selection, single-point crossover, mutation, and elitism.
9. **Prediction target**: A multi-objective fitness score: `F(P) = 0.5 * BW_min(P) - 0.3 * D_sum(P) - 0.2 * L_comp(P)`.
10. **Routing/optimization method, if any**: Metaheuristic optimization (GA). Selected paths are enforced dynamically via the Floodlight Static Flow Pusher REST API.
11. **Evaluation methodology**: Compared against a baseline Dijkstra’s hop-count shortest-path algorithm using active measurement tools (iperf, ping).
12. **Evaluation metrics**: Throughput (Mbps), Latency (ms), Packet Delivery Ratio (PDR, %).
13. **Main reported findings/results**: GA found paths that successfully balanced bandwidth, latency, and packet loss, avoiding bottlenecks that Dijkstra's algorithm fell victim to.
14. **Limitations explicitly stated by the authors**: Significant REST API call overhead to set up flow rules (requires 5-second stabilization wait). Modest latency gains over Dijkstra due to the extra OVS forwarding step in the optimal 5-hop path versus the 4-hop path. Controller link-state entries timeout rapidly, requiring continuous high-priority LLDP polling.
15. **Limitations that can reasonably be inferred, clearly labelled as inference**: *(Inference)* The framework uses current/snapshot metrics to calculate fitness, making it reactive; it may still suffer from transient congestion before the GA is re-run and new rules are installed.
16. **Relevance to our proposed project**: Provides the structural foundation for the GA routing component, demonstrating how QoS metrics can be normalized and combined into a valid fitness function to program SDN controllers.

---

## Cross-Paper Comparison

| Aspect | Paper 1 (DG-STMTL) | Paper 2 (ST-GCN & Anomaly) | Paper 3 (QoS-GA in SDN) |
| :--- | :--- | :--- | :--- |
| **Main problem** | Spatio-temporal multi-task forecasting | Joint traffic prediction & anomaly detection | Multi-objective QoS route optimization |
| **Dataset** | PEMSD4, PEMSD8, NYCFHV (Traffic/Demand) | Unnamed (Traffic, matrices, anomalies) | Simulated (100 host-pair experiments) |
| **Network environment**| Segmented road/city graphs | Generalized device/connection graphs | Mininet, Floodlight controller, OVS |
| **Prediction** | Traffic attributes (next time interval) | Multidimensional traffic features (multi-horizon) | Multi-objective path fitness score |
| **Security/anomaly** | None | Binary classification of injected anomalies | None |
| **Routing** | None | None | SDN flow-rule enforcement (Floodlight REST API) |
| **Optimization** | Adam (model loss) | Joint optimization (model loss) | Genetic Algorithm (path finding) |
| **Metrics** | MAE, RMSE, MAPE | MAE, RMSE, MAPE, Precision, Recall, F1, AUC | Throughput, Latency, Packet Delivery Ratio |
| **Main limitation** | Overfitting/oversmoothing with depth | Not explicitly stated | Control-plane overhead, latency trade-offs, state timeouts |
| **Relevance** | Graph modeling for accurate forecasting | Node-level QoS feature processing & anomaly risk | GA formulation for path selection using QoS |
