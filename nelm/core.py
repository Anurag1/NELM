"""Small, auditable evidence-aware online learning components."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass

@dataclass(frozen=True)
class Observation:
    key: str
    value: str
    source: str
    step: int

class FrozenPrior:
    """Per-key majority model fitted only on the shared calibration stream."""
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.fitted = False
    def update(self, obs, truth=None):
        if not self.fitted:
            self.counts[obs.key][obs.value] += 1
    def freeze(self):
        self.fitted = True
    def predict(self, key):
        counts = self.counts.get(key)
        return sorted(counts, key=lambda v: (-counts[v], v))[0] if counts else "unknown"

class LatestMemory:
    """Minimal retrieval baseline: latest observed value for a key."""
    def __init__(self):
        self.latest = {}
    def update(self, obs, truth=None):
        self.latest[obs.key] = obs.value
    def predict(self, key):
        return self.latest.get(key, "unknown")

class FrequencyMemory:
    """Unlabeled evidence memory using all observed values equally."""
    def __init__(self):
        self.counts = defaultdict(Counter)
    def update(self, obs, truth=None):
        self.counts[obs.key][obs.value] += 1
    def predict(self, key):
        counts = self.counts.get(key)
        return sorted(counts, key=lambda v: (-counts[v], v))[0] if counts else "unknown"

class SupervisedFrequency:
    """Per-key frequency learner updated with the same delayed labels as NELM."""
    def __init__(self):
        self.counts = defaultdict(Counter)
    def update(self, obs, truth=None):
        self.counts[obs.key][truth if truth is not None else obs.value] += 1
    def predict(self, key):
        counts = self.counts.get(key)
        return sorted(counts, key=lambda v: (-counts[v], v))[0] if counts else "unknown"

class NELM:
    """Evidence-weighted learner with delayed-feedback source calibration.

    Feedback is provided only after a prediction has been scored. It cannot
    change the prediction that was just evaluated.
    """
    def __init__(self, decay=0.985, beta_prior=(1.0, 1.0)):
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.alpha0, self.beta0 = beta_prior
        self.rows = defaultdict(list)
        self.source_success = defaultdict(lambda: self.alpha0)
        self.source_failure = defaultdict(lambda: self.beta0)
        self.contradictions = 0
        self.feedback_updates = 0
    def predict(self, key, now=None):
        rows = self.rows.get(key, [])
        if not rows:
            return "unknown"
        current_step = max(r.step for r in rows) if now is None else now
        scores = defaultdict(float)
        latest = defaultdict(lambda: -1)
        for row in rows:
            age = max(0, current_step - row.step)
            success = self.source_success[row.source]
            failure = self.source_failure[row.source]
            reliability = success / (success + failure)
            scores[row.value] += max(0.05, reliability) * (self.decay ** age)
            latest[row.value] = max(latest[row.value], row.step)
        return sorted(scores, key=lambda v: (-scores[v], -latest[v], v))[0]
    def update(self, obs, truth=None):
        seen = {row.value for row in self.rows[obs.key]}
        if seen and obs.value not in seen:
            self.contradictions += 1
        self.rows[obs.key].append(obs)
        if truth is not None:
            if obs.value == truth:
                self.source_success[obs.source] += 1.0
            else:
                self.source_failure[obs.source] += 1.0
            self.feedback_updates += 1
    def diagnostics(self):
        return {
            "observations_stored": sum(len(rows) for rows in self.rows.values()),
            "contradictory_observations": self.contradictions,
            "source_feedback_updates": self.feedback_updates,
            "sources_calibrated": len(self.source_success),
        }
