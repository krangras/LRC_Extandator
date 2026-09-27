import unittest
from unittest.mock import patch
import tempfile

import numpy as np

from alignment_engine import ctc_viterbi_align, _window_bounds, normalize_words
from lrc_maker import generate_elrc


class FakeRomanizer:
    def romanize_string(self, value, lcode=None):
        mapping = {"привет": "privet", "мир": "mir"}
        return mapping.get(value, value)


class ForcedAlignmentCoreTest(unittest.TestCase):
    def test_ctc_viterbi_finds_known_token_spans(self):
        # vocab: blank=0, a=1, b=2. Desired path is blank,a,a,blank,b,b,blank.
        probs = np.array([
            [0.98, 0.01, 0.01],
            [0.02, 0.97, 0.01],
            [0.02, 0.97, 0.01],
            [0.98, 0.01, 0.01],
            [0.02, 0.01, 0.97],
            [0.02, 0.01, 0.97],
            [0.98, 0.01, 0.01],
        ], dtype=np.float32)
        spans = ctc_viterbi_align(np.log(probs), [1, 2])
        self.assertEqual((spans[0]["start_frame"], spans[0]["end_frame"]), (1, 3))
        self.assertEqual((spans[1]["start_frame"], spans[1]["end_frame"]), (4, 6))
        self.assertGreater(spans[0]["confidence"], 0.9)

    def test_repeated_ctc_token_requires_blank_between(self):
        probs = np.array([
            [0.98, 0.02],
            [0.02, 0.98],
            [0.98, 0.02],
            [0.02, 0.98],
            [0.98, 0.02],
        ], dtype=np.float32)
        spans = ctc_viterbi_align(np.log(probs), [1, 1])
        self.assertLess(spans[0]["end_frame"], spans[1]["start_frame"])

    def test_lrc_anchor_makes_local_window(self):
        start, end = _window_bounds(
            42.1, 46.3, 240.0,
            pre_margin=0.95, post_margin=0.35, last_line_window=11.0,
        )
        self.assertAlmostEqual(start, 41.15)
        self.assertAlmostEqual(end, 46.65)

    def test_known_words_are_romanized_not_recognized(self):
        words, target = normalize_words("Привет мир", "rus", FakeRomanizer())
        self.assertEqual(target, "privetmir")
        self.assertEqual([w.display for w in words], ["Привет", "мир"])
        self.assertEqual([w.chars for w in words], ["privet", "mir"])

    def test_last_line_window_is_bounded(self):
        start, end = _window_bounds(120.0, None, 180.0, pre_margin=1.0, post_margin=0.3, last_line_window=9.0)
        self.assertAlmostEqual(start, 119.0)
        self.assertAlmostEqual(end, 129.0)

    def test_censored_known_word_is_normalized_for_alignment(self):
        words, target = normalize_words("sh!t", "eng", FakeRomanizer())
        self.assertEqual(target, "shit")
        self.assertEqual(words[0].display, "sh!t")

    def test_timed_lrc_reaches_forced_alignment_engine(self):
        aligned_line = {
            "start": 1.0, "end": 1.8, "anchor_start": 1.0, "anchor_delta_ms": 0.0,
            "text": "hello world", "confidence": 0.9, "method": "MMS_FA/custom-CTC-viterbi/local",
            "words": [
                {"word": "hello", "start": 1.0, "end": 1.35, "confidence": 0.92, "origin": "aligned"},
                {"word": "world", "start": 1.4, "end": 1.8, "confidence": 0.88, "origin": "aligned"},
            ],
        }
        fake_engine = unittest.mock.MagicMock()
        fake_engine.align.return_value = {
            "lines": [aligned_line],
            "quality": {"grade": "good", "score": 0.9},
            "backend": "MMS_FA/custom-CTC-viterbi", "device": "cuda", "gpuName": "GPU",
            "runtimeSec": 0.1, "cacheHit": False,
            "selectedCandidate": {"label": "mix/local-ctc", "use_demucs": False},
            "candidates": [],
        }
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            with patch("lrc_maker._extract_metadata", return_value={}), patch("lrc_maker._engine", return_value=fake_engine):
                result = generate_elrc(
                    audio.name, artist="Artist", title="Song",
                    lrc_text="[00:01.000]hello world", use_demucs=False, language="eng",
                )
        self.assertTrue(result["success"])
        self.assertIn("<00:01.000>hello", result["elrc"])
        fake_engine.align.assert_called_once()

    def test_plain_lyrics_fail_instead_of_invoking_asr(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            with patch("lrc_maker._extract_metadata", return_value={}), patch("lrc_maker.search_lrc", return_value=None), patch("lrc_maker.logger.exception"):

                result = generate_elrc(
                    audio.name,
                    artist="Artist",
                    title="Song",
                    lrc_text="known words but no timestamps",
                    use_demucs=False,
                )
        self.assertFalse(result["success"])
        self.assertIn("Нужен обычный синхронизированный LRC", result["error"])


if __name__ == "__main__":
    unittest.main()
