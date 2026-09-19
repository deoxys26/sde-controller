# Research Gap and Novelty

## 1. Established from Literature
Based on the strict review of the core reference papers, the following concepts are established and validated in independent domains:
- **Graph-based Telemetry Forecasting**: Spatial-temporal GNNs (such as DG-STMTL and ST-GCN) are highly effective at modeling network topologies to predict future traffic features (flow, speed, delay, bandwidth utilization) and outperform traditional time-series methods (Paper 1, Paper 2).
- **Joint Prediction and Anomaly Detection**: Graph neural architectures can simultaneously predict continuous traffic metrics and identify the presence of discrete anomalies (Paper 2).
- **Multi-Objective Metaheuristic Routing**: Genetic Algorithms can successfully navigate the NP-hard problem of finding optimal SDN paths that balance conflicting QoS constraints (delay, bandwidth, packet loss), and these decisions can be successfully deployed to SDN controllers (Paper 3).

## 2. Apparent Gap
While the aforementioned components exist independently, there are distinct gaps in their combination:
- **Reactive Routing Shortfalls**: Current metaheuristic SDN routing frameworks (like the one in Paper 3) optimize paths based exclusively on *current* or *historical* telemetry snapshots. They react to congestion rather than preventing it.
- **Disconnected Forecasting**: GNN frameworks in literature (Papers 1 & 2) accurately forecast network states but stop at prediction. They do not feed these predictive outputs into a live routing decision engine.
- **Security as a Multi-Objective Routing Constraint**: While anomalies can be detected using graph data (Paper 2), security risk is rarely integrated directly into a mathematical multi-objective routing fitness function (Paper 3 focuses only on bandwidth, delay, and loss).

## 3. Proposed Contribution
This project intends to implement and experimentally test a framework that bridges these gaps through the following combination:
**GNN-based future QoS forecasting + security/anomaly-derived risk + GA multi-objective route optimization + proactive routing decisions**.

Specifically, this project proposes feeding the *predicted* QoS metrics and *anomaly risk scores* directly into the GA's fitness function, allowing an SDN controller (simulated) to enforce routes that avoid predicted congestion and compromised nodes *before* degradation occurs.

## 4. Novelty Status
Explicitly, the combination of GNN forecasting, anomaly detection, and GA optimization for proactive SDN routing is a **proposed integration** intended for experimental evaluation. 

**This combination is not yet a proven novel contribution.** It cannot be definitively claimed as globally novel without an exhaustive systematic review of the broader networking literature beyond the three core reference papers. The purpose of this project is to implement this integrated framework and test the hypothesis that this specific combination yields measurable improvements over reactive baseline methods.
