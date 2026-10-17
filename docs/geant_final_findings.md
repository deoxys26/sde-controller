# GEANT v2 one-step utilization forecasting — final findings

## Question and scope

Can a small B-U graph model or a per-link residual MLP improve one-step forecasting of computed GEANT directed-link utilization over fixed alpha-0.8 EWMA? The evaluated protocol used 7,344 training windows, 1,715 validation windows, 72 directed links, 900-second sampling, five historical inputs, and a one-step target. Utilization is computed from real GEANT traffic matrices under the project's topology/capacity assumptions; it is not measured live telemetry.

Training and validation arrays were bound to the train/validation-only provenance sidecar; scalers were fit on training data only, and validation loss alone selected checkpoints. The test split remained sealed and no test evaluation was performed. The `--splits train,val` builder did parse raw source matrices assigned to the test period to construct its full in-memory series/mask, but did not use those observations in training or validation windows and did not emit test arrays or test-window metadata. This is an implementation limitation, not a claim that test-period raw files were never read by the builder.

## Validation experiment

All methods below used the same 1,715 validation windows, 72 directed-edge columns, and 123,480 target-link pairs per seed. Metrics are dimensionless computed-utilization-ratio MAE/RMSE. Seeds are repetitions; their spread is descriptive, not a confidence interval.

| Seed | EWMA MAE / RMSE | Residual MLP MAE / RMSE | B-U MAE / RMSE |
|---:|---:|---:|---:|
| 42 | 0.0019424265 / 0.0049833796 | 0.0019399105 / 0.0049658327 | 0.0021622893 / 0.0050802591 |
| 123 | 0.0019424265 / 0.0049833796 | 0.0019365921 / 0.0049634832 | 0.0019609815 / 0.0050173820 |
| 2026 | 0.0019424265 / 0.0049833796 | 0.0019373443 / 0.0049569504 | 0.0020201566 / 0.0050369759 |
| Mean ± sample SD (`ddof=1`) | 0.0019424265 ± 0 / 0.0049833796 ± 0 | 0.0019379490 ± 0.0000017398 / 0.0049620888 ± 0.0000046024 | 0.0020478091 ± 0.0001034635 / 0.0050448723 ± 0.0000321737 |

Aggregate MAE/RMSE differences from EWMA (candidate minus EWMA, computed from the reported same-window aggregate metrics) were:

| Seed | Residual MLP ΔMAE / ΔRMSE | B-U ΔMAE / ΔRMSE |
|---:|---:|---:|
| 42 | -0.0000025161 / -0.0000175469 | +0.0002198628 / +0.0000968795 |
| 123 | -0.0000058344 / -0.0000198963 | +0.0000185550 / +0.0000340025 |
| 2026 | -0.0000050822 / -0.0000264291 | +0.0000777301 / +0.0000535963 |
| Mean | -0.0000044776 / -0.0000212908 | +0.0001053826 / +0.0000614928 |

Paired per-seed absolute-error differences (candidate minus EWMA) and 95% window-bootstrap percentile bounds were:

| Seed | Residual MLP ΔMAE [95% bounds] | B-U ΔMAE [95% bounds] |
|---:|---:|---:|
| 42 | -0.0000025161 [-0.0000053406, +0.0000003949] | +0.0002198628 [+0.0002116627, +0.0002283892] |
| 123 | -0.0000058344 [-0.0000090886, -0.0000022916] | +0.0000185550 [+0.0000128349, +0.0000242507] |
| 2026 | -0.0000050822 [-0.0000087026, -0.0000012246] | +0.0000777301 [+0.0000697022, +0.0000858097] |

The frozen residual usefulness gate **FAILED**: residual MAE was lower than EWMA in 3/3 seeds, but mean seed-level relative reduction was 0.2305%, below 1%. B-U was worse than EWMA in 3/3 seeds (mean MAE about 5.4% higher) and is contextual only. The evaluator used a paired window-level percentile bootstrap (2,000 resamples, 95% interval, RNG seed 20261017), retaining each sampled window's full link-error vector. Because the windows overlap, treating resampled windows as independent likely makes the intervals too narrow. Evaluator source SHA-256 recorded for this result: `023EFDBDD65D28DF11C7ADFF722E44085AA4C76EB8B244C500C179283DB7853E`.

On the Stage 9 chronological holdout, EWMA MAE was 0.00276 versus 0.00226 on all of train; the holdout period is noisier, and the previous-residual Ridge feature degraded held-out error. This is consistent with nonstationarity, but does not establish it as the cause.

## Training-only diagnostics

These are diagnostics on the training block, not validation/test results and not a change to frozen baselines.

- On 528,768 aligned training target-link pairs, last-value persistence MAE/RMSE was 0.00228508/0.00663146; five-step mean was 0.00285208/0.00712650. EWMA alpha MAE/RMSE: α=0.2, 0.00264637/0.00673034; α=0.4, 0.00245475/0.00641521; α=0.6, 0.00231775/0.00627888; α=0.8, 0.00226063/0.00635888; α=0.95, 0.00227106/0.00654627. Alpha 0.8 is retained; no tuning was performed.
- Target utilization: exact zeros 0.908%, near-zero (`|u|≤1e-6`) 0.911%; pooled p50 0.01696, p90 0.08471, p95 0.12116, p99 0.19851, max 0.55261. Per-link q50/q90/q95/q99 are in the machine-readable Stage 9 report.
- EWMA error concentration: top 1% of pairs account for 20.11% of absolute error; top 5%, 41.44%; per-link error Gini 0.4592. Highest error-share link indices: 18 (6.28%), 33 (4.72%), 16 (4.56%), 64 (4.00%), 47 (3.63%), 36 (3.37%), 17 (3.13%), 46 (3.11%), 35 (2.95%), 2 (2.70%).
- Mean per-link target autocorrelation was 0.927 at lag 1, 0.859 at lag 4, 0.656 at lag 96, and 0.523 at lag 192. EWMA residual autocorrelation was -0.095 (lag 1), 0.020 (lag 4), 0.084 (lag 96), 0.064 (lag 192), and 0.068 (lag 672). Pairs were restricted to eight reconstructed contiguous runs; no gap was bridged.
- Spatial check, with residual `r(q)=target−EWMA`, 20 shuffled neighbor assignments and seed 20261017: `corr(r(q), mean_neighbor(u(q−1)))=-0.03157`; shuffled 5th–95th band was [-0.03043,-0.02590]. The real-neighbor correlation lies slightly outside the band on the negative side; `r²≈0.1%`, negligible for forecasting. This is not “no difference from shuffled,” but it is not evidence of useful predictive gain. Same-window neighbor-residual correlation was 0.39191 versus shuffled mean 0.28381; that contemporaneous target-period statistic is unavailable at forecast time and is not predictive evidence.
- Fixed-alpha Ridge feature screen on a chronological 70/30 timeline split with a 96-step embargo: same 139,896 held-out pairs; EWMA reference MAE/RMSE 0.00275893/0.00871081. Seasonal-change only: 0.00275796/0.00871007, MAE/RMSE reductions +0.0352%/+0.0085%. Lag-96 residual: 0.00276936/0.00875381, -0.3782%/-0.4937%. Hour-of-day sin/cos: 0.00278791/0.00870654, -1.0506%/+0.0490%. Previous residual: 0.00286932/0.00893328, -4.0012%/-2.5540%. Combined: 0.00291916/0.00899399, -5.8077%/-3.2509%. MAE is primary; Ridge minimizes squared error, so an RMSE gain without an MAE gain is possible.

The predeclared diagnostic rule selected the best fixed feature set by MAE reduction: 0.0352% is below the 0.5% threshold. Candidate A is unsupported, Candidate B is not justified, and no v2 builder is proposed.

## Conclusion and limits

On GEANT 15-minute link utilization, B-U underperformed EWMA by about 5.4% MAE on average. A learned residual over EWMA improved MAE by 0.23%, below the predeclared 1% gate. Training-only diagnostics found no feature set meeting the practical threshold for one-step improvement; the causal spatial correlation was tiny and negative, and seasonal/time-of-day screens did not provide a useful MAE gain. These results do not establish that GNNs cannot work on this problem.

Scope is limited to one-step forecasts, MAE/RMSE objectives, this GEANT-derived dataset, and a linear signal screen plus one small per-link MLP residual model. All model-comparison evidence is validation-only; no test evaluation was performed. The bootstrap's overlapping-window independence assumption likely understates uncertainty.

Not tested in this protocol: longer forecast horizons; event/threshold objectives; probabilistic forecasts; larger or alternative neural architectures; and other datasets. These are out of scope, not negative findings.

Machine-readable training-only diagnostic: `C:\Users\Admin\DN_audit\stage9_train_only_feature_audit_complete.json`.
