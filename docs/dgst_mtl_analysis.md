# DG-STMTL Technical Analysis for the SDN QoS Project

## Source and scope

This analysis is based on the local PDF *DG-STMTL: A Novel Graph Convolutional Network for Multi-Task Spatio-Temporal Traffic Forecasting* (Cui, Wang, Yin; April 2025 preprint). It forecasts related road-traffic tasks such as speed and flow (and pickup/drop-off demand), not SDN link QoS.

## Reference architecture

Input is a node-by-history-by-task-by-feature tensor. The paper's CTKE unit concatenates task inputs, adds learnable temporal-task and spatial-task embeddings, max-pools across history, projects the result, reshapes it, forms an all-pairs dot-product correlation matrix, and softmax-normalizes it into a dynamic adjacency. Its HAMG combines that dynamic matrix with a prior `3N x 3N` adjacency. The prior combines physical, temporal-self, and correlation-based spatio-temporal connectivity over three timesteps. A learnable task-specific gate modulates the combined adjacency for each task.

Each task has its own input projection and GSTGC module. GSTGC groups time into three-step blocks, performs three GCN aggregation stages per group, fuses input/intermediate states with normalized learned residual weights, crops the middle timestep, then applies overlapping feature grouping and max pooling. Task outputs are concatenated and passed through two shared output layers with a skip connection. The loss is a weighted sum of per-task Smooth L1 losses, with task coefficients and thresholds. The reported setup predicts the next interval from 12 historical steps, uses Adam/early stopping (patience 10, maximum 200 epochs), and reports MAE/RMSE/MAPE. The paper's datasets are PEMS road-detector flow/speed and NYC for-hire-vehicle pickup/drop-off demand, with chronological dataset splits described as 6:2:2 for PEMS.

The authors explicitly motivate the design around the trade-off between static priors (rigidity/domain bias) and fully learned adjacency (overfitting/instability), plus negative task interference. Their ablations report degradation without CTKE, static or dynamic adjacency components, task gate, task-specific inputs, residual fusion, or grouping. Scalability is limited by adjacency operations with an `N^2` contribution; they discuss sparse representations and future large-scale optimization.

## What transfers directly to SDN and what does not?

| Component | Classification | SDN interpretation |
|---|---|---|
| Multi-output forecasting and Smooth L1 | Directly reusable | Delay, utilization, loss, and jitter are related regression targets. |
| Historical input and chronological split | Directly reusable | Preserves the Phase 4 leakage rules. |
| Static plus dynamic adjacency | Reusable with modification | SDN is directed and QoS lives on links, not road-detector nodes. |
| Task-specific gates/projections | Reusable with modification | Retained as a compact link-QoS approximation. |
| Three-step grouping/residual GCN | Reusable with modification | History is configurable and may not be 12; partial final groups are pooled. |
| `3N x 3N` road synchronous adjacency | Unsuitable unchanged | Our prediction entities are directed links and no road spatial-correlation prior exists. |
| Road connectivity/correlation graph | Unsuitable unchanged | It must be replaced by topology-derived directed link relationships. |
| Security as a GNN task | Unsuitable for Phase 4.5 | UNSW-NB15 remains a separate downstream risk model. |

The paper does not train on SDN data, model security risk, or evaluate routing decisions. Its reported accuracy claims therefore do not transfer to this project.
