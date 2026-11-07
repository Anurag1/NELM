"""Leakage-controlled synthetic online benchmark for NELM."""
from __future__ import annotations
import argparse, csv, json, random, statistics, time
from pathlib import Path
from .core import Observation, FrozenPrior, LatestMemory, FrequencyMemory, SupervisedFrequency, NELM

DOMAINS = ("materials", "biology", "astronomy", "computing", "geography", "energy")
VALUES = ("alpha", "beta", "gamma")
SOURCE_RELIABILITY = {"atlas": .94, "beacon": .78, "cipher": .58, "delta": .38}

def make_stream(seed, keys_count=36, calibration_steps=180, eval_steps=720):
    """Generate latent drifting facts and noisy observations for one seed."""
    rng = random.Random(seed)
    keys = [f"{DOMAINS[i % len(DOMAINS)]}.property_{i:02d}" for i in range(keys_count)]
    latent = {key: rng.choice(VALUES) for key in keys}
    stream = []
    for step in range(calibration_steps + eval_steps):
        key = rng.choice(keys)
        if rng.random() < .035:
            latent[key] = rng.choice([v for v in VALUES if v != latent[key]])
        source = rng.choice(tuple(SOURCE_RELIABILITY))
        truth = latent[key]
        observed = truth if rng.random() < SOURCE_RELIABILITY[source] else rng.choice([v for v in VALUES if v != truth])
        stream.append((Observation(key, observed, source, step), truth))
    return keys, stream

def metrics(correct, latencies):
    q = max(1, len(correct) // 4)
    return {
        "accuracy": statistics.mean(correct),
        "first_quarter_accuracy": statistics.mean(correct[:q]),
        "last_quarter_accuracy": statistics.mean(correct[-q:]),
        "mean_prediction_latency_ms": statistics.mean(latencies),
        "p95_prediction_latency_ms": sorted(latencies)[min(len(latencies)-1, int(.95*(len(latencies)-1)))],
    }

def run_seed(seed, calibration_steps=180, eval_steps=720):
    _, stream = make_stream(seed, calibration_steps=calibration_steps, eval_steps=eval_steps)
    models = {
        "Frozen calibration prior": FrozenPrior(),
        "Latest observation": LatestMemory(),
        "Frequency memory": FrequencyMemory(),
        "Supervised truth frequency": SupervisedFrequency(),
        "NELM": NELM(),
    }
    # Identical unlabeled calibration stream for every system.
    for obs, _truth in stream[:calibration_steps]:
        for model in models.values():
            model.update(obs)
    models["Frozen calibration prior"].freeze()
    scored = {name: [] for name in models}
    timings = {name: [] for name in models}
    for obs, truth in stream[calibration_steps:]:
        # Every system predicts before current observation and truth are revealed.
        for name, model in models.items():
            start = time.perf_counter_ns()
            pred = model.predict(obs.key)
            timings[name].append((time.perf_counter_ns()-start)/1e6)
            scored[name].append(int(pred == truth))
        # Reveal only after all predictions have been scored.
        for name, model in models.items():
            if name in ("NELM", "Supervised truth frequency"):
                model.update(obs, truth=truth)
            elif name != "Frozen calibration prior":
                model.update(obs)
    result = {name: metrics(scored[name], timings[name]) for name in models}
    result["NELM diagnostics"] = models["NELM"].diagnostics()
    result["evaluation_decisions"] = eval_steps
    return result

def aggregate(runs):
    names = ("Frozen calibration prior", "Latest observation", "Frequency memory", "Supervised truth frequency", "NELM")
    metric_names = ("accuracy", "first_quarter_accuracy", "last_quarter_accuracy", "mean_prediction_latency_ms", "p95_prediction_latency_ms")
    out = {}
    for name in names:
        out[name] = {}
        for metric in metric_names:
            values = [run[name][metric] for run in runs]
            out[name][metric + "_mean"] = statistics.mean(values)
            out[name][metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return out

def run_benchmark(seeds=range(10), calibration_steps=180, eval_steps=720, outdir="results"):
    seed_list = list(seeds)
    runs = [run_seed(seed, calibration_steps, eval_steps) for seed in seed_list]
    summary = aggregate(runs)
    payload = {
        "benchmark": "NELM synthetic drifting-facts benchmark",
        "protocol": {
            "seeds": seed_list, "calibration_steps_per_seed": calibration_steps,
            "evaluation_steps_per_seed": eval_steps, "domains": list(DOMAINS),
            "values": list(VALUES), "source_reliability": SOURCE_RELIABILITY,
            "evaluation": "prequential: all systems predict before current observation/truth is revealed",
            "feedback": "NELM and supervised truth-frequency receive the same ground-truth feedback after scoring; other systems receive observations without labels",
            "limitations": "synthetic task; not a public leaderboard and not a comparison to commercial LLMs",
        },
        "summary": summary, "seed_runs": runs,
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "accuracy_mean", "accuracy_sd", "first_quarter_mean", "last_quarter_mean", "mean_latency_ms"])
        for name, values in summary.items():
            writer.writerow([name, values["accuracy_mean"], values["accuracy_sd"], values["first_quarter_accuracy_mean"], values["last_quarter_accuracy_mean"], values["mean_prediction_latency_ms_mean"]])
    return payload

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--calibration-steps", type=int, default=180)
    parser.add_argument("--eval-steps", type=int, default=720)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    payload = run_benchmark(range(args.seeds), args.calibration_steps, args.eval_steps, args.outdir)
    print(f"NELM benchmark | {args.seeds} seeds | {args.eval_steps} evaluation decisions/seed")
    print(f"{'System':<28} {'Accuracy':>10} {'First ¼':>10} {'Last ¼':>10} {'Mean ms':>10}")
    for name, row in payload["summary"].items():
        print(f"{name:<28} {row['accuracy_mean']*100:>9.2f}% {row['first_quarter_accuracy_mean']*100:>9.2f}% {row['last_quarter_accuracy_mean']*100:>9.2f}% {row['mean_prediction_latency_ms_mean']:>10.5f}")
    print("\nNELM seed-0 diagnostics:", payload["seed_runs"][0]["NELM diagnostics"])
    print(f"\nWrote {args.outdir}/results.json and {args.outdir}/summary.csv")

if __name__ == "__main__":
    main()
