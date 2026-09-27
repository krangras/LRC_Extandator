import unittest

from lrc_formats import build_outputs, format_time, parse_lyrics, repair_lines


class LrcFormatsTest(unittest.TestCase):
    def test_format_rounds_across_minute(self):
        self.assertEqual(format_time(59.9996, 3), "01:00.000")
        self.assertEqual(format_time(1.239, 2), "00:01.24")

    def test_standard_lrc_keeps_line_anchor_without_fake_words(self):
        parsed = parse_lyrics("[ar:A]\n[00:01.250]First line")
        self.assertEqual(parsed["lines"][0]["start"], 1.25)
        self.assertEqual(parsed["lines"][0]["anchor_start"], 1.25)
        self.assertEqual(parsed["lines"][0]["words"], [])

    def test_elrc_round_trip_keeps_word_times(self):
        source = [{
            "start": 2.001,
            "end": 3.456,
            "text": "one two",
            "words": [
                {"word": "one", "start": 2.001, "end": 2.555},
                {"word": "two", "start": 2.700, "end": 3.456},
            ],
        }]
        output = build_outputs(source, {"ar": "A"})["elrc"]
        imported = parse_lyrics(output)["lines"][0]
        self.assertEqual([w["start"] for w in imported["words"]], [2.001, 2.7])
        self.assertEqual(imported["words"][-1]["end"], 3.456)

    def test_repair_preserves_confidence_and_origin(self):
        fixed, _ = repair_lines([{
            "start": 1.0,
            "text": "hello",
            "confidence": 0.8,
            "method": "test",
            "words": [{"word": "hello", "start": 1.0, "end": 1.5, "confidence": 0.9, "origin": "aligned"}],
        }])
        self.assertEqual(fixed[0]["method"], "test")
        self.assertEqual(fixed[0]["words"][0]["origin"], "aligned")
        self.assertAlmostEqual(fixed[0]["words"][0]["confidence"], 0.9)

    def test_offset_baked_into_anchor(self):
        parsed = parse_lyrics("[offset:-100]\n[00:01.000]word")
        self.assertEqual(parsed["lines"][0]["start"], 0.9)
        self.assertEqual(parsed["lines"][0]["anchor_start"], 0.9)


if __name__ == "__main__":
    unittest.main()
