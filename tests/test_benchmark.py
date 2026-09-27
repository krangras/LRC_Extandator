import unittest

from benchmark import score_lines


class BenchmarkMetricTest(unittest.TestCase):
    def test_known_100ms_shift(self):
        ref = [{
            "start": 1.0,
            "text": "one two",
            "words": [
                {"word": "one", "start": 1.0, "end": 1.3},
                {"word": "two", "start": 1.5, "end": 1.8},
            ],
        }]
        cand = [{
            "start": 1.1,
            "text": "one two",
            "words": [
                {"word": "one", "start": 1.1, "end": 1.4},
                {"word": "two", "start": 1.6, "end": 1.9},
            ],
        }]
        metrics = score_lines(ref, cand)
        self.assertAlmostEqual(metrics["wordOnsetMaeMs"], 100.0, places=3)
        self.assertEqual(metrics["matchedWords"], 2)
        self.assertAlmostEqual(metrics["within100ms"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
