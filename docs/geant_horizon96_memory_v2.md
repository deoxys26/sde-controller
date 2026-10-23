# GEANT horizon-96 memory protocol v2

## Stage 24 finding

The v1 full runner uses a lazy `LazyH96Dataset`, `DataLoader(num_workers=0)`, and a per-batch `[B,672,72]` history slice; it does not stack all overlapping windows. The training epoch keeps only scalar loss/counts and a short list of step durations, but it left the just-finished parameter gradients allocated until the next batch. V2 clears gradients with `set_to_none=True` after each optimizer update and explicitly drops batch/autograd references before sampling memory.

V1 performs a startup RAM/disk preflight and records process memory before/after training, but it has no in-run resource monitor. Its validation function also retains predictions, targets, and four baseline prediction arrays on every epoch although checkpoint selection uses only the scalar h=96 validation residual SmoothL1. V2 uses an objective-only validation pass during epoch selection and materializes validation predictions/baselines only for the final strict checkpoint-reload evaluation.

The recorded Stage 23 interruption followed applying the 1 GiB startup-availability floor as an in-run stop condition. That is a category mismatch: OS-available RAM is system-wide, while process RSS/high-water is process-specific. The in-run reserve is the existing 384 MiB reserve; 983,699,456 bytes of system-available RAM is above that reserve. This does not prove the original process was healthy: its process peak was not captured in the available record, and its incomplete checkpoint/run directory was not inspected or changed for Stage 24. There is no evidence here establishing an actual tensor/graph leak as the cause.

## Bounded synthetic CPU benchmark

Benchmark command form (executed in four fresh Python processes, one per batch):

```powershell
python -c "import json,psutil; from phase9_geant.horizon96_experiment import run_synthetic_cpu_benchmark; a=psutil.virtual_memory().available; r=run_synthetic_cpu_benchmark(batch_size=4, measured_steps=4); b=psutil.virtual_memory().available; r['os_available_ram_before_bytes']=a; r['os_available_ram_after_bytes']=b; print(json.dumps(r, sort_keys=True))"
```

The same command was run with `batch_size=2`, `8`, and `16`. Each run used one warm-up and four measured CPU optimizer steps over synthetic `[batch,672,72]` input and `[batch,72]` target tensors, the existing 1,489-parameter model, and no GEANT files, arrays, manifests, or checkpoints.

| Batch | Step times (s, four measured) | Median (s) | Process high-water (bytes) | OS available before → after (bytes) |
|---:|---|---:|---:|---:|
| 2 | 0.247996, 0.254059, 0.258477, 0.263366 | 0.256268 | 387,284,992 | 1,840,734,208 → 1,704,370,176 |
| 4 | 0.339678, 0.342022, 0.332597, 0.343967 | 0.340850 | 462,741,504 | 1,789,853,696 → 1,602,981,888 |
| 8 | 0.541211, 0.610723, 0.554984, 0.501053 | 0.548098 | 613,597,184 | 1,873,006,592 → 1,603,489,792 |
| 16 | 0.971581, 0.956978, 1.012047, 1.118855 | 0.991814 | 1,039,900,672 | 1,854,824,448 → 1,610,416,128 |

The process high-water field is psutil's Windows peak working set when available; otherwise the platform peak RSS, with current RSS fallback. It includes the Python/PyTorch process baseline and is not a GPU-memory measurement. OS-available RAM is a system-wide point sample, can move because of unrelated processes, and is not attributed to the benchmark. Batch 8 is close to the existing 640 MiB process estimate; batch 16 exceeds it. Batch 4 was selected for the new protocol because its observed process peak left about 199 MiB under the 640 MiB limit. Batch 2 has faster individual steps but about twice the batches per epoch; linear extrapolation therefore predicts slower epoch throughput. For 3,924 windows, batch 4 is about 334 seconds of training batches per epoch and about 6,687 seconds for 20 epochs, excluding validation, data scans, and checkpoint I/O; these are estimates, not observed training durations. These short synthetic steps do not guarantee full-run memory or runtime.

## Versioned configuration and monitor

New protocol ID: `geant-horizon96-only-candidate-v2-batch4`.

New output root: `data/processed/geant_horizon96_streaming_experiment_v2/`. Existing output roots, including v1 and the interrupted Stage 23 run, are not reused, resumed, overwritten, or inspected.

Batch size is fixed at 4 for every future seed (42, 123, 2026) and adjacency mode (topology, identity, shuffled). This is an explicit optimization-protocol amendment: with 3,924 training windows, it yields 981 optimizer batches per epoch rather than 123 at batch 32. Adam update frequency therefore changes; v2 is not claimed equivalent to v1. History 672, horizon 96 only, EWMA alpha 0.8, SmoothL1 beta 0.001, Adam at 0.005, hidden size 16, 20-epoch maximum, patience 4, fixed adjacency shuffle seed, split boundaries, scaler procedure, validation-only checkpoint rule, and graph controls remain fixed.

The v2 runner retains the existing startup gate (at least 640 MiB estimated process capacity plus 384 MiB reserve, i.e. 1 GiB OS-available RAM, and 512 MiB free disk). During execution it samples process RSS/high-water and system-wide available RAM separately after each training and validation batch. It stops if the available process high-water measure exceeds 640 MiB or OS-available RAM falls below the 384 MiB reserve. On Windows this uses psutil's `peak_wset`; if a platform exposes neither `peak_wset` nor `peak_rss`, the monitor records and enforces the maximum sampled RSS instead, which can miss between-sample spikes. If psutil cannot supply a complete snapshot, execution fails closed. The monitor records sample count, maximum sampled RSS, maximum platform high-water when exposed, the measurement source/fallback, minimum observed OS-available RAM, thresholds, and the last sample in the run manifest. A resource stop is recorded as `failed`; the last atomically saved best checkpoint remains diagnostic only and cannot make the run complete. An abrupt OS/process termination can leave the durable status `incomplete`; no automatic retry or resume is supported.

Atomic checkpoint replacement is delegated to the existing same-directory temporary-file/fsync/replace helper; failure before replacement leaves the previous valid checkpoint intact. This is application-level protection, not a guarantee against power loss, OneDrive synchronization conflicts, or a hostile concurrent writer on Windows.

Future runs require separate explicit authorization. Example syntax only (not run in Stage 24):

```powershell
python -m phase9_geant.horizon96_streaming_experiment_v2 --run-training --authorize-horizon96-candidate --data-root data/processed/geant_horizon96_streaming_v1 --repository-root . --seed 42 --adjacency-mode topology --device cpu
```

The other eight fixed seed/mode combinations use the same command with only the seed and adjacency-mode arguments changed. No real-data training or evaluation was performed for this version in Stage 24.
