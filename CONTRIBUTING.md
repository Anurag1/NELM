# Contributing to NELM

Thanks for helping make continual-learning experiments more reproducible.

## Good first contributions
- Add a baseline with the same information access as NELM.
- Add a public dataset adapter with versioned download instructions.
- Add ablations for source calibration or recency weighting.
- Improve uncertainty and calibration metrics.
- Add memory and compute accounting.

## Development
Run:
    python -m unittest discover -s tests -v
    python -m nelm.benchmark --seeds 2 --calibration-steps 30 --eval-steps 60 --outdir /tmp/nelm-results

## Pull request checklist
- [ ] Tests pass.
- [ ] Random seeds and data protocol are documented.
- [ ] Baselines receive equivalent information unless the difference is the explicit subject of study.
- [ ] No test-set or future-label leakage.
- [ ] Claims match measured evidence.
- [ ] New dependencies are justified.
