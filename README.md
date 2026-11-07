# NELM — Never-Ending Learning Machine

**An open research prototype for evidence-aware continual learning.**

[![Tests](https://github.com/Anurag1/NELM/actions/workflows/tests.yml/badge.svg)](https://github.com/Anurag1/NELM/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](LICENSE)](LICENSE)

*Observe → Question → Discover → Experiment → Verify → Remember → Improve*

NELM explores a practical research question: **can an AI system improve its ability to learn from a changing stream of evidence, rather than merely retrieve its latest observation?**

This repository contains an inspectable Python implementation and a reproducible synthetic benchmark. It is a research starting point—not a claim of AGI, autonomous scientific discovery, or superiority to commercial foundation models.

## Implemented
- Evidence history with source and time metadata.
- Contradiction tracking.
- Beta-style source reliability calibration using delayed feedback.
- Recency-weighted evidence prediction.
- Five baselines: frozen prior, latest observation, unlabeled frequency memory, supervised truth-frequency control, and NELM.
- Prequential evaluation: predict first, score, then reveal.

## Quick start
Requires Python 3.10+; no third-party runtime dependencies.

    git clone https://github.com/Anurag1/NELM.git
    cd NELM
    python -m unittest discover -s tests -v
    python -m nelm.benchmark

Run a smoke benchmark:

    python -m nelm.benchmark --seeds 3 --calibration-steps 60 --eval-steps 120

Outputs: results/results.json and results/summary.csv.

## Architecture
Observation → Evidence memory → Source/time-aware prediction → Score → Delayed feedback → Update.

The prototype implements evidence memory, contradiction counting, recency weighting and source calibration. Question generation, autonomous experiment design, universal knowledge graphs and adaptive meta-learning are roadmap items—not current capabilities.

## Benchmark honesty
The task is synthetic and narrow. All systems receive the same unlabeled calibration observations. During evaluation all systems predict before the current truth is revealed. NELM and a supervised truth-frequency control then receive the same delayed ground-truth labels; other baselines receive only observations. This makes the supervision control explicit.

In the 10-seed run, NELM reached 80.32% mean accuracy; the supervised truth-frequency control reached 82.99%. NELM did **not** win against the strongest supervision-matched baseline. That negative result is important: the current evidence-weighting approach needs further work. No GPT, Claude, Gemini or other commercial model was queried. No public benchmark leaderboard result is claimed. See docs/PROTOCOL.md and results/REPORT.md.

## Roadmap
See docs/ROADMAP.md. Priorities: replay baselines, ablations, calibration metrics, public datasets, resource accounting and independently verifiable experiment records.

## Contributing
See CONTRIBUTING.md. Please do not submit unsupported superiority claims.

Created by [Anurag1](https://github.com/Anurag1).
