# Benchmark protocol

## Research question
Does source-reliability-aware, recency-weighted evidence memory improve online prediction on a synthetic stream of noisy, drifting categorical facts compared with simple memory baselines and a label-feedback control?

## Task generation
Each seed creates 36 keys across six synthetic domains. Each key has a latent value from alpha, beta, gamma. At each step one key is sampled; its latent value changes with probability 0.035. A source is sampled uniformly from four sources with fixed reliabilities (0.94, 0.78, 0.58, 0.38). The emitted observation is correct with the source reliability; otherwise one of the other values is emitted uniformly.

## Shared calibration
All systems receive the same first 180 observations without labels. The frozen prior is fitted on these observations and frozen. Latest-observation and frequency-memory baselines retain the same observations. NELM and the supervised truth-frequency baseline begin evaluation without labels.

## Evaluation
For each evaluation decision, every system predicts before the current observation or truth is revealed. Predictions are scored against latent truth. Only after all predictions are scored is the observation revealed. NELM receives the observation and ground-truth label to calibrate source reliability and retain evidence. The supervised truth-frequency baseline receives the same delayed label and updates per-key truth counts. Other baselines receive the observation without labels. The supervised baseline helps separate label-access benefit from NELM's source-aware weighting.

## Metrics
- Accuracy over all evaluation decisions.
- Accuracy in first and last evaluation quartiles.
- Mean and p95 prediction latency (microbenchmark only; noisy at sub-millisecond scale).
- Across-seed mean and sample standard deviation.

## Reproducibility
Run: python -m nelm.benchmark --seeds 10 --calibration-steps 180 --eval-steps 720

## Known limitations
This is a synthetic task, not a recognized public benchmark. The supervised truth-frequency baseline is a simple control, not a full suite of published continual-learning methods. Future studies should add replay learners, ablations, real datasets, confidence intervals, compute/memory accounting and pre-registered hyperparameters.
