# Phase 5: UNSW-NB15 Security / Anomaly Detection

## Implementation facts

UNSW-NB15 is used only for the binary security-risk model. It is not mixed into the synthetic simulator telemetry or the Phase 4 QoS forecaster. The local data contains 175,341 training rows and 82,332 test rows, 45 columns, and no missing values. It includes numerical flow features, categorical `proto`, `service`, and `state`, an `id` field, `attack_cat`, and binary `label`.

`label` is the primary task: zero is normal and one is attack. `id` is removed because it is a dataset-row identifier, `label` is removed because it is the target, and `attack_cat` is removed because it is an annotation that directly exposes attack status. No other source columns are discarded. Numeric columns receive median imputation and standard scaling; categorical columns receive most-frequent imputation and one-hot encoding. The fitted transformer is trained solely on the training portion and then reused unchanged for validation and the official test file. Infinite values are converted to missing values before imputation.

The configured 30,000-row seeded, unweighted stratified sample is drawn from the supplied training CSV for CPU-friendly baseline experiments; a seeded 15% validation split is then drawn only from that sample. The whole official test CSV remains untouched for final evaluation. It is never used for fitting preprocessing or model parameters.

Two binary models are implemented: Logistic Regression as an interpretable reference, and a 50-tree seeded Random Forest as a stronger tree baseline. Metrics are accuracy, precision, recall, F1, false-positive rate, confusion matrix, and Brier score. Attack-category output is descriptive: it reports support, mean risk, and predicted-attack rate per existing `attack_cat`; it is not a multiclass classifier evaluation.

## Risk score and calibration

Risk equals the classifier's `P(label=1)` and is clipped to `[0, 1]`. At the configured threshold (default 0.5), a risk at or above the threshold maps to an attack decision. This is a model score, not a claim of calibrated real-world attack probability. The Brier score is reported as a basic probability-quality diagnostic; no calibration method is fitted in this phase.

## Actual initial experiment results

Using seed 42, the configured 30,000-row training sample yielded 8,130 normal and 17,370 attack fit rows; the untouched test file contains 37,000 normal and 45,332 attack rows. At risk threshold 0.5, Logistic Regression achieved accuracy 0.8087, precision 0.7542, recall 0.9681, F1 0.8479, false-positive rate 0.3865, and Brier score 0.1130. Its confusion matrix was `[[22698, 14302], [1446, 43886]]` (rows actual normal/attack; columns predicted normal/attack).

The 50-tree Random Forest achieved accuracy 0.8698, precision 0.8161, recall 0.9857, F1 0.8929, false-positive rate 0.2722, and Brier score 0.0821. Its confusion matrix was `[[26929, 10071], [648, 44684]]`. It is the stronger of these two measured benchmark baselines, but these results do not establish real-world intrusion-detection performance or calibrated attack probabilities.

The category summary showed very high predicted-attack rates for Generic, DoS, Exploits, Reconnaissance, Backdoor, Shellcode and Worms in this test set. Fuzzers had lower rates (0.920 Logistic Regression, 0.903 Random Forest). The normal category's predicted-attack rate exactly equals each model's false-positive rate. This descriptive breakdown is not a multiclass evaluation.

## Limitations and research interpretation

UNSW-NB15 is a public benchmark and is not live traffic from the simulated SDN topology. Its risk output is deliberately not yet associated with routes or consumed by a Genetic Algorithm. Phase 5 establishes reproducible benchmarks; measured results must not be generalized to production networks, called novel, or used to assert model superiority without direct comparison.
