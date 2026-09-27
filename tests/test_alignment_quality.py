import unittest

from alignment_quality import quality_report


class AlignmentQualityTest(unittest.TestCase):
    def test_good_alignment_scores_above_interpolated(self):
        good = [{
            "start": 1,
            "end": 2,
            "text": "a b",
            "confidence": 0.95,
            "words": [
                {"word": "a", "start": 1, "end": 1.4, "confidence": 0.95, "origin": "aligned"},
                {"word": "b", "start": 1.5, "end": 2, "confidence": 0.9, "origin": "aligned"},
            ],
        }]
        bad = [{
            "start": 1,
            "end": 2,
            "text": "a b",
            "confidence": 0.2,
            "words": [
                {"word": "a", "start": 1, "end": 1.5, "confidence": 0.12, "origin": "interpolated"},
                {"word": "b", "start": 1.5, "end": 2, "confidence": 0.12, "origin": "interpolated"},
            ],
        }]
        self.assertGreater(quality_report(good)["score"], quality_report(bad)["score"])


    def test_adaptive_and_line_switch_diagnostics_are_counted(self):
        report = quality_report([{
            "start": 1.0, "display_start": 1.12, "end": 2.0, "text": "a b",
            "confidence": 0.6, "adaptive_rescue_used": True,
            "cross_line_reconciled": True, "line_switch_delayed_for_previous": True,
            "words": [
                {"word": "a", "start": 1.0, "end": 1.4, "confidence": 0.8, "origin": "adaptive_consensus"},
                {"word": "b", "start": 1.5, "end": 2.0, "confidence": 0.35, "origin": "tempo_energy_rescue"},
            ],
            "tempo_reconstructed_words": 1,
        }])
        self.assertEqual(report["adaptiveRescues"], 1)
        self.assertEqual(report["tempoRescuedWords"], 1)
        self.assertEqual(report["lineSwitchReconciliations"], 1)
        self.assertEqual(report["lineSwitchDelays"], 1)

    def test_structural_error_is_not_publishable(self):
        report = quality_report([{
            "start": 1,
            "end": 2,
            "text": "x",
            "confidence": 1,
            "words": [{"word": "x", "start": 1.5, "end": 1.4, "confidence": 1, "origin": "aligned"}],
        }])
        self.assertFalse(report["publishable"])
        self.assertGreater(report["invalidWordDurations"], 0)


if __name__ == "__main__":
    unittest.main()
