import unittest

from lrc_formats import build_outputs, format_time, parse_lyrics, repair_lines


class LrcFormatsTest(unittest.TestCase):
    def test_format_rounds_across_minute(self):
        self.assertEqual(format_time(59.9996, 3), "01:00.000")
        self.assertEqual(format_time(1.239, 2), "00:01.24")

    def test_standard_lrc_and_metadata(self):
        parsed = parse_lyrics("[ar:Исполнитель]\n[ti:Песня]\n[00:01.25]Первая строка")
        self.assertEqual(parsed["metadata"]["ar"], "Исполнитель")
        self.assertEqual(parsed["lines"][0]["text"], "Первая строка")
        self.assertEqual(parsed["lines"][0]["start"], 1.25)

    def test_enhanced_lrc_uses_word_starts_and_terminal_end(self):
        parsed = parse_lyrics("[00:01.000]<00:01.100>Hello <00:01.600>world <00:02.250>")
        words = parsed["lines"][0]["words"]
        self.assertEqual([word["start"] for word in words], [1.1, 1.6])
        self.assertEqual(words[-1]["end"], 2.25)

    def test_foobar_inline_square_timestamps(self):
        parsed = parse_lyrics("[00:05.000]hello [00:06.000]world")
        self.assertEqual([word["word"] for word in parsed["lines"][0]["words"]], ["hello", "world"])
        self.assertEqual(parsed["lines"][0]["words"][1]["start"], 6.0)

    def test_offset_is_baked_into_all_cues(self):
        parsed = parse_lyrics("[offset:-100]\n[00:01.000]<00:01.000>word <00:01.500>")
        self.assertEqual(parsed["lines"][0]["start"], 0.9)
        self.assertEqual(parsed["lines"][0]["words"][0]["end"], 1.4)
        self.assertNotIn("offset", parsed["metadata"])

    def test_multiple_line_timestamps(self):
        parsed = parse_lyrics("[00:01.00][00:02.00]repeat")
        self.assertEqual([line["start"] for line in parsed["lines"]], [1.0, 2.0])

    def test_repair_fills_missing_word_times(self):
        fixed, _ = repair_lines([{"start": 1, "text": "a b", "words": [{"word": "a"}, {"word": "b"}]}])
        self.assertGreater(fixed[0]["words"][0]["end"], fixed[0]["words"][0]["start"])
        self.assertGreaterEqual(fixed[0]["words"][1]["start"], fixed[0]["words"][0]["start"])

    def test_exact_elrc_round_trip(self):
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
        self.assertEqual(imported["words"], source[0]["words"])


if __name__ == "__main__":
    unittest.main()
