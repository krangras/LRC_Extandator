import unittest
from unittest.mock import patch
import tempfile

import numpy as np

from alignment_engine import (
    ctc_viterbi_align, _window_bounds, normalize_words, _align_line_once,
    _last_word_candidate_score, _boundary_is_suspicious, _refine_last_word_boundary, get_profile,
)
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


    def test_next_line_lookahead_constrains_boundary_but_is_not_returned(self):
        # vocab: blank,a,b,c,d. Current line is "ab", next-line context is "cd".
        probs = np.array([
            [0.98, 0.005, 0.005, 0.005, 0.005],
            [0.01, 0.96, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.96, 0.01, 0.01],
            [0.96, 0.01, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.96, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.96],
            [0.96, 0.01, 0.01, 0.01, 0.01],
        ], dtype=np.float32)
        source = {"start": 1.0, "anchor_start": 1.0, "text": "ab"}
        next_line = {"start": 2.0, "anchor_start": 2.0, "text": "cd"}
        token_dict = {"-": 0, "a": 1, "b": 2, "c": 3, "d": 4}
        waveform = np.zeros(4 * 16000, dtype=np.float32)
        with patch("alignment_engine._emission_for_window", return_value=np.log(probs)):
            result = _align_line_once(
                waveform, source, 2.0,
                next_line=next_line, lookahead_words=1,
                language="eng", model=None, token_dict=token_dict, device="cpu",
                pre_margin=0.5, post_margin=1.0, last_line_window=5.0,
                method_suffix="test", romanizer=FakeRomanizer(),
            )
        self.assertEqual([w["word"] for w in result["words"]], ["ab"])
        self.assertEqual(result["lookahead_text"], "cd")
        self.assertEqual(result["lookahead_words_used"], 1)
        self.assertIsNotNone(result["lookahead_start"])
        self.assertLess(result["words"][-1]["start"], result["lookahead_start"])

    def test_lookahead_backoff_does_not_break_current_line(self):
        # Current b + three lookahead 'a' tokens cannot fit in four frames because
        # repeated CTC tokens require blanks. b + two 'a' tokens can fit exactly.
        probs = np.array([
            [0.01, 0.01, 0.98],
            [0.01, 0.98, 0.01],
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
        ], dtype=np.float32)
        source = {"start": 1.0, "anchor_start": 1.0, "text": "b"}
        next_line = {"start": 2.0, "anchor_start": 2.0, "text": "a a a"}
        token_dict = {"-": 0, "a": 1, "b": 2}
        waveform = np.zeros(4 * 16000, dtype=np.float32)
        with patch("alignment_engine._emission_for_window", return_value=np.log(probs)):
            result = _align_line_once(
                waveform, source, 2.0,
                next_line=next_line, lookahead_words=3,
                language="eng", model=None, token_dict=token_dict, device="cpu",
                pre_margin=0.5, post_margin=1.0, last_line_window=5.0,
                method_suffix="test", romanizer=FakeRomanizer(),
            )
        self.assertEqual(result["lookahead_words_requested"], 3)
        self.assertEqual(result["lookahead_words_used"], 2)
        self.assertEqual(result["words"][0]["word"], "b")

    def test_boundary_guard_flags_word_stuck_to_next_anchor(self):
        profile = get_profile("max")
        line = {
            "window_end": 11.2,
            "lookahead_start": 10.15,
            "words": [{"word": "last", "start": 9.94, "end": 10.08, "confidence": 0.91, "origin": "aligned"}],
        }
        self.assertTrue(_boundary_is_suspicious(line, next_anchor=10.0, profile=profile))

    def test_boundary_candidate_prefers_clean_separation_from_next_line(self):
        stuck = {"start": 9.99, "end": 10.05, "confidence": 0.82, "origin": "aligned"}
        rescued = {"start": 9.55, "end": 9.91, "confidence": 0.80, "origin": "aligned"}
        stuck_score = _last_word_candidate_score(stuck, next_anchor=10.0, lookahead_start=10.08)
        rescued_score = _last_word_candidate_score(rescued, next_anchor=10.0, lookahead_start=10.08)
        self.assertGreater(rescued_score, stuck_score)


    def test_boundary_rescue_replaces_last_word_when_primary_is_stuck(self):
        current = {
            "start": 8.8, "end": 10.04, "confidence": 0.86,
            "method": "primary", "window_end": 11.2,
            "lookahead_start": 10.10, "lookahead_confidence": 0.9,
            "words": [
                {"word": "one", "start": 8.9, "end": 9.3, "confidence": 0.92, "origin": "aligned"},
                {"word": "last", "start": 9.98, "end": 10.04, "confidence": 0.82, "origin": "aligned"},
            ],
        }
        rescue = {
            "lookahead_start": 10.08, "lookahead_confidence": 0.91,
            "words": [
                {"word": "one", "start": 8.91, "end": 9.31, "confidence": 0.90, "origin": "aligned"},
                {"word": "last", "start": 9.54, "end": 9.92, "confidence": 0.80, "origin": "aligned"},
            ],
        }
        source = {"start": 8.8, "anchor_start": 8.8, "text": "one last"}
        next_line = {"start": 10.0, "anchor_start": 10.0, "text": "next line"}
        with patch("alignment_engine._align_line_once", return_value=rescue):
            result = _refine_last_word_boundary(
                np.zeros(12 * 16000, dtype=np.float32), source, next_line, 10.0, current,
                language="eng", profile=get_profile("max"), model=None,
                token_dict={"-": 0}, device="cpu", romanizer=FakeRomanizer(),
            )
        self.assertTrue(result["boundary_rescue_used"])
        self.assertEqual(result["words"][-1]["origin"], "aligned_boundary_rescue")
        self.assertAlmostEqual(result["words"][-1]["start"], 9.54)
        self.assertLess(result["boundary_rescue_delta_ms"], -400)

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
