import unittest

from lrc_maker import _alignment_quality, lines_from_alignment


class AlignmentQualityTest(unittest.TestCase):
    def test_reports_interpolated_words(self):
        payload = {
            "lines": [{
                "raw_text": "hello world",
                "assigned_time": 1.0,
                "end_time": 2.0,
                "aligned_units": [
                    {
                        "start_time": 1.1,
                        "end_time": 1.4,
                        "metadata": {"source_word": "hello"},
                    },
                ],
            }],
        }

        lines = lines_from_alignment(payload)

        self.assertEqual(lines[0]["matched_words"], 1)
        self.assertEqual(lines[0]["interpolated_words"], 1)
        self.assertEqual(lines[0]["alignment_quality"], 0.5)
        self.assertEqual(_alignment_quality(lines), 0.5)


if __name__ == "__main__":
    unittest.main()
