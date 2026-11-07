import unittest
from nelm.core import Observation, FrozenPrior, LatestMemory, FrequencyMemory, NELM

class CoreTests(unittest.TestCase):
    def test_frozen_prior_stops_learning_after_freeze(self):
        model = FrozenPrior()
        model.update(Observation("k", "a", "s", 0))
        model.freeze()
        model.update(Observation("k", "b", "s", 1))
        self.assertEqual(model.predict("k"), "a")
    def test_latest_memory_returns_latest(self):
        model = LatestMemory()
        model.update(Observation("k", "a", "s", 0))
        model.update(Observation("k", "b", "s", 1))
        self.assertEqual(model.predict("k"), "b")
    def test_frequency_memory_counts_observations(self):
        model = FrequencyMemory()
        for i, value in enumerate(("a", "b", "a")):
            model.update(Observation("k", value, "s", i))
        self.assertEqual(model.predict("k"), "a")
    def test_nelm_feedback_is_delayed(self):
        model = NELM()
        model.update(Observation("k", "a", "source", 0))
        self.assertEqual(model.feedback_updates, 0)
        model.update(Observation("k", "b", "source", 1), truth="a")
        self.assertEqual(model.feedback_updates, 1)
        self.assertEqual(model.source_failure["source"], 2.0)
    def test_nelm_detects_conflicts(self):
        model = NELM()
        model.update(Observation("k", "a", "s", 0))
        model.update(Observation("k", "b", "s", 1))
        self.assertEqual(model.diagnostics()["contradictory_observations"], 1)
    def test_invalid_decay_rejected(self):
        with self.assertRaises(ValueError):
            NELM(decay=0)

if __name__ == "__main__":
    unittest.main()
