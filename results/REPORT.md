# NELM synthetic benchmark — initial reproducible run

Configuration: 10 random seeds; 180 shared unlabeled calibration observations and 720 evaluation decisions per seed; 7,200 scored decisions per system.

## Results

| System | Mean accuracy | First quarter | Last quarter | Mean latency (ms) |
|---|---:|---:|---:|---:|
| Frozen calibration prior | 63.31% | 75.11% | 53.39% | 0.00070 |
| Latest observation | 65.21% | 65.44% | 65.50% | 0.00018 |
| Frequency memory | 78.11% | 81.28% | 71.94% | 0.00072 |
| Supervised truth frequency | **82.99%** | **87.11%** | 76.06% | 0.00064 |
| NELM | 80.32% | 80.61% | **80.06%** | 0.00645 |

Means across seeds. JSON contains sample standard deviations and per-seed outputs.

## Interpretation
- NELM exceeded unlabeled frequency memory by 2.21 percentage points in this synthetic setup.
- NELM underperformed the supervision-matched truth-frequency control by 2.67 points overall.
- NELM's final-quarter accuracy was more stable in this run, but this does not establish statistical significance.
- NELM was slower than the simplest baselines in this Python microbenchmark.

**Conclusion:** The prototype demonstrates evidence storage, contradiction counting, delayed source feedback and online prediction. This run does not show that NELM beats a simple baseline with equivalent label access. The next step is to investigate the supervised frequency result and run ablations.

## Limitations
Synthetic task, not a recognized public continual-learning benchmark. No commercial LLM APIs were called. This is not a comparison against GPT, Claude, Gemini or modern foundation models.
