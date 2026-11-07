import os, tempfile, unittest
from nelm.benchmark import run_seed, run_benchmark

class BenchmarkTests(unittest.TestCase):
    def test_seed_accuracy_is_reproducible(self):
        a, b = run_seed(7, 20, 40), run_seed(7, 20, 40)
        for name in ("Frozen calibration prior", "Latest observation", "Frequency memory", "Supervised truth frequency", "NELM"):
            self.assertEqual(a[name]["accuracy"], b[name]["accuracy"])
    def test_benchmark_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = run_benchmark(range(2), 10, 20, directory)
            self.assertEqual(len(payload["seed_runs"]), 2)
            self.assertTrue(os.path.exists(os.path.join(directory, "results.json")))
            self.assertTrue(os.path.exists(os.path.join(directory, "summary.csv")))

if __name__ == "__main__":
    unittest.main()
