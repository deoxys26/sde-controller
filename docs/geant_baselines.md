# GEANT Non-Neural Baselines Evaluation

This report concerns **forecasting of computed link utilization from real GEANT traffic matrices**. The distance-shortest-path series is primary; hop-shortest-path results are robustness-only. It does not evaluate real-world QoS or routing improvement.

## Chronological protocol and distribution check

Each sample has five historical 15-minute utilization steps and one next-step target; every window is valid and wholly within its assigned train, validation, or test block. EWMA alpha is selected by validation MAE only. AR(5) and per-link slot means fit only training observations. Busy targets use raw float64 total demand >= the training p90 threshold.

- Train busy-target threshold: `59652.665088` Mbps

| Block | Demand mean | Demand median | Demand p95 | Utilization mean | Utilization p95 | Demand/train mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 45217.083530 | 43908.495255 | 62332.552163 | 0.031582 | 0.121249 | 1.000000 |
| val | 39687.033712 | 39009.201005 | 54248.354471 | 0.024098 | 0.092226 | 0.877700 |
| test | 37527.275414 | 36635.605386 | 52514.406075 | 0.022469 | 0.091090 | 0.829936 |

Validation and test demand means differ from train if the reported ratios differ from 1.0. This is descriptive evidence of a chronological distribution shift, not a causal or statistically confirmed finding.


## Series: DISTANCE ROUTING

- **Chosen EWMA alpha (val)**: 0.8
- **AR(5) Coefficients (x_0..x_4)**: `[-0.01057, 0.07411, 0.13645, 0.13489, 0.65767]`
- **AR(5) Intercept**: `0.000212`

### Block: VAL

- Mean target utilization (overall): `0.024095`
- Mean target utilization (busy targets): `0.036970`

| Baseline | Overall MAE | Overall RMSE | Overall Norm MAE | Busy MAE | Busy RMSE | Busy Norm MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **persistence** | 0.001969 | 0.005060 | 0.0817 | 0.002979 | 0.009170 | 0.0806 |
| **ewma** | 0.001942 | 0.004983 | 0.0806 | 0.003081 | 0.009932 | 0.0833 |
| **ar5** | 0.001996 | 0.005019 | 0.0828 | 0.003272 | 0.010734 | 0.0885 |
| **slot_mean** | 0.015995 | 0.029253 | 0.6638 | 0.020081 | 0.042484 | 0.5432 |
| **seasonal_naive_96** (cov=100.00%) | 0.006343 | 0.016731 | 0.2633 | 0.005952 | 0.019536 | 0.1610 |
| *persistence (same subset)* | 0.001969 | 0.005060 | 0.0817 | 0.002979 | 0.009170 | 0.0806 |

#### Per-link metrics
The complete 72-link MAE vector is retained in `data/processed/geant/baselines.json`; best/worst five are listed below.

Best five persistence links: it1.it->ch1.ch (0.000042), fr1.fr->ch1.ch (0.000066), it1.it->gr1.gr (0.000082), be1.be->lu1.lu (0.000092), cz1.cz->sk1.sk (0.000100)

Worst five persistence links: de1.de->se1.se (0.014361), pl1.pl->se1.se (0.011633), uk1.uk->ie1.ie (0.006445), at1.at->de1.de (0.006035), fr1.fr->be1.be (0.005513)

#### Persistence MAE for Top 10 Busiest Links (Train Mean Util)
| Rank | Link | Train Mean Util | Persistence MAE |
| :--- | :--- | :--- | :--- |
| 1 | at1.at->hu1.hu | 0.152436 | 0.001289 |
| 2 | il1.il->it1.it | 0.113866 | 0.001902 |
| 3 | ch1.ch->it1.it | 0.104009 | 0.002446 |
| 4 | at1.at->ny1.ny | 0.099595 | 0.002290 |
| 5 | ch1.ch->fr1.fr | 0.090652 | 0.002419 |
| 6 | ch1.ch->at1.at | 0.087754 | 0.004645 |
| 7 | be1.be->fr1.fr | 0.078753 | 0.002317 |
| 8 | de1.de->se1.se | 0.075222 | 0.014361 |
| 9 | fr1.fr->lu1.lu | 0.060943 | 0.001663 |
| 10 | gr1.gr->de1.de | 0.060224 | 0.001289 |

### Block: TEST

- Mean target utilization (overall): `0.022494`
- Mean target utilization (busy targets): `0.036642`

| Baseline | Overall MAE | Overall RMSE | Overall Norm MAE | Busy MAE | Busy RMSE | Busy Norm MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **persistence** | 0.001565 | 0.003494 | 0.0696 | 0.002339 | 0.005590 | 0.0638 |
| **ewma** | 0.001557 | 0.003477 | 0.0692 | 0.002343 | 0.005530 | 0.0639 |
| **ar5** | 0.001621 | 0.003543 | 0.0720 | 0.002485 | 0.005950 | 0.0678 |
| **slot_mean** | 0.014405 | 0.024826 | 0.6404 | 0.018279 | 0.035831 | 0.4988 |
| **seasonal_naive_96** (cov=100.00%) | 0.005189 | 0.011922 | 0.2307 | 0.007856 | 0.021496 | 0.2144 |
| *persistence (same subset)* | 0.001565 | 0.003494 | 0.0696 | 0.002339 | 0.005590 | 0.0638 |

#### Per-link metrics
The complete 72-link MAE vector is retained in `data/processed/geant/baselines.json`; best/worst five are listed below.

Best five persistence links: fr1.fr->ch1.ch (0.000001), cz1.cz->sk1.sk (0.000002), it1.it->gr1.gr (0.000041), it1.it->ch1.ch (0.000046), be1.be->lu1.lu (0.000108)

Worst five persistence links: de1.de->se1.se (0.006313), uk1.uk->ie1.ie (0.005941), pl1.pl->cz1.cz (0.005005), de1.de->nl1.nl (0.004991), fr1.fr->be1.be (0.004664)

#### Persistence MAE for Top 10 Busiest Links (Train Mean Util)
| Rank | Link | Train Mean Util | Persistence MAE |
| :--- | :--- | :--- | :--- |
| 1 | at1.at->hu1.hu | 0.152436 | 0.001190 |
| 2 | il1.il->it1.it | 0.113866 | 0.001727 |
| 3 | ch1.ch->it1.it | 0.104009 | 0.002191 |
| 4 | at1.at->ny1.ny | 0.099595 | 0.002361 |
| 5 | ch1.ch->fr1.fr | 0.090652 | 0.002689 |
| 6 | ch1.ch->at1.at | 0.087754 | 0.002295 |
| 7 | be1.be->fr1.fr | 0.078753 | 0.002189 |
| 8 | de1.de->se1.se | 0.075222 | 0.006313 |
| 9 | fr1.fr->lu1.lu | 0.060943 | 0.001400 |
| 10 | gr1.gr->de1.de | 0.060224 | 0.001289 |


## Series: HOP ROUTING

- **Chosen EWMA alpha (val)**: 0.8
- **AR(5) Coefficients (x_0..x_4)**: `[0.00638, 0.04851, 0.13248, 0.17698, 0.63031]`
- **AR(5) Intercept**: `0.000135`

### Block: VAL

- Mean target utilization (overall): `0.022312`
- Mean target utilization (busy targets): `0.034345`

| Baseline | Overall MAE | Overall RMSE | Overall Norm MAE | Busy MAE | Busy RMSE | Busy Norm MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **persistence** | 0.001774 | 0.005036 | 0.0795 | 0.002765 | 0.009265 | 0.0805 |
| **ewma** | 0.001752 | 0.004972 | 0.0785 | 0.002876 | 0.010082 | 0.0837 |
| **ar5** | 0.001809 | 0.005054 | 0.0811 | 0.003072 | 0.011041 | 0.0895 |
| **slot_mean** | 0.012987 | 0.026343 | 0.5821 | 0.015986 | 0.037459 | 0.4655 |
| **seasonal_naive_96** (cov=100.00%) | 0.005977 | 0.017632 | 0.2679 | 0.005535 | 0.019418 | 0.1612 |
| *persistence (same subset)* | 0.001774 | 0.005036 | 0.0795 | 0.002765 | 0.009265 | 0.0805 |

#### Per-link metrics
The complete 72-link MAE vector is retained in `data/processed/geant/baselines.json`; best/worst five are listed below.

Best five persistence links: de1.de->fr1.fr (0.000007), cz1.cz->sk1.sk (0.000020), it1.it->gr1.gr (0.000082), nl1.nl->de1.de (0.000085), it1.it->de1.de (0.000100)

Worst five persistence links: at1.at->de1.de (0.014767), at1.at->hu1.hu (0.012162), de1.de->it1.it (0.007444), be1.be->nl1.nl (0.006001), ny1.ny->at1.at (0.005848)

#### Persistence MAE for Top 10 Busiest Links (Train Mean Util)
| Rank | Link | Train Mean Util | Persistence MAE |
| :--- | :--- | :--- | :--- |
| 1 | at1.at->hu1.hu | 0.249378 | 0.012162 |
| 2 | be1.be->fr1.fr | 0.156089 | 0.003193 |
| 3 | ch1.ch->at1.at | 0.140196 | 0.005164 |
| 4 | il1.il->it1.it | 0.130851 | 0.002052 |
| 5 | at1.at->ny1.ny | 0.113938 | 0.002857 |
| 6 | at1.at->si1.si | 0.089667 | 0.002809 |
| 7 | at1.at->de1.de | 0.089508 | 0.014767 |
| 8 | gr1.gr->de1.de | 0.083827 | 0.002003 |
| 9 | ch1.ch->it1.it | 0.080407 | 0.002010 |
| 10 | ch1.ch->fr1.fr | 0.069028 | 0.002295 |

### Block: TEST

- Mean target utilization (overall): `0.020732`
- Mean target utilization (busy targets): `0.034126`

| Baseline | Overall MAE | Overall RMSE | Overall Norm MAE | Busy MAE | Busy RMSE | Busy Norm MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **persistence** | 0.001408 | 0.003474 | 0.0679 | 0.002095 | 0.005940 | 0.0614 |
| **ewma** | 0.001400 | 0.003468 | 0.0676 | 0.002119 | 0.005794 | 0.0621 |
| **ar5** | 0.001462 | 0.003569 | 0.0705 | 0.002239 | 0.005992 | 0.0656 |
| **slot_mean** | 0.012956 | 0.027198 | 0.6250 | 0.014553 | 0.030492 | 0.4264 |
| **seasonal_naive_96** (cov=100.00%) | 0.004714 | 0.012266 | 0.2274 | 0.006811 | 0.020055 | 0.1996 |
| *persistence (same subset)* | 0.001408 | 0.003474 | 0.0679 | 0.002095 | 0.005940 | 0.0614 |

#### Per-link metrics
The complete 72-link MAE vector is retained in `data/processed/geant/baselines.json`; best/worst five are listed below.

Best five persistence links: fr1.fr->ch1.ch (0.000001), cz1.cz->sk1.sk (0.000002), de1.de->fr1.fr (0.000008), it1.it->gr1.gr (0.000041), it1.it->es1.es (0.000087)

Worst five persistence links: be1.be->nl1.nl (0.005785), de1.de->it1.it (0.005629), ny1.ny->at1.at (0.004979), at1.at->ch1.ch (0.004833), at1.at->de1.de (0.004423)

#### Persistence MAE for Top 10 Busiest Links (Train Mean Util)
| Rank | Link | Train Mean Util | Persistence MAE |
| :--- | :--- | :--- | :--- |
| 1 | at1.at->hu1.hu | 0.249378 | 0.004066 |
| 2 | be1.be->fr1.fr | 0.156089 | 0.002807 |
| 3 | ch1.ch->at1.at | 0.140196 | 0.002969 |
| 4 | il1.il->it1.it | 0.130851 | 0.001738 |
| 5 | at1.at->ny1.ny | 0.113938 | 0.002818 |
| 6 | at1.at->si1.si | 0.089667 | 0.003294 |
| 7 | at1.at->de1.de | 0.089508 | 0.004423 |
| 8 | gr1.gr->de1.de | 0.083827 | 0.001703 |
| 9 | ch1.ch->it1.it | 0.080407 | 0.002026 |
| 10 | ch1.ch->fr1.fr | 0.069028 | 0.002420 |
