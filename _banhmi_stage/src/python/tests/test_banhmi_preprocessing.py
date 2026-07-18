import tempfile
import unittest
from pathlib import Path

from banhmi_tts.preprocessing.audio import AudioRejected, cache_key, continuous_log_f0
from banhmi_tts.preprocessing.config import PreprocessConfig
from banhmi_tts.preprocessing.manifest import (
    ProcessedRecord,
    assign_splits,
    load_ljspeech,
)
from banhmi_tts.preprocessing.text import (
    build_vocabulary,
    encode_phonemes,
    find_espeak,
    ipa_to_symbols,
    phonemize_batch,
)


def make_record(index: int) -> ProcessedRecord:
    return ProcessedRecord(
        schema_version=1,
        utterance_id=f"utt-{index:04d}",
        text=f"Text {index}",
        phonemes=["t"],
        phoneme_ids=[1, 2, 3],
        audio_path=f"artifacts/audio/{index}.pt",
        spectrogram_path=f"artifacts/spectrogram/{index}.pt",
        pitch_path=f"artifacts/pitch/{index}.pt",
        sample_rate=22_050,
        num_samples=22_050,
        num_spectrogram_frames=86,
        duration_seconds=1.0,
        source_sha256="a" * 64,
        cache_key="b" * 64,
        quality={},
        pitch={
            "voiced_frames": 50.0,
            "voiced_ratio": 0.5,
            "log_f0_sum": 1.0,
            "log_f0_squared_sum": 1.0,
        },
    )


class ConfigTests(unittest.TestCase):
    def test_fingerprint_is_stable(self):
        first = PreprocessConfig()
        second = PreprocessConfig.from_dict(first.to_dict())
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_cache_key_covers_text_and_config(self):
        base = cache_key("source", "id", "text", "config")
        self.assertNotEqual(base, cache_key("source", "id", "changed", "config"))
        self.assertNotEqual(base, cache_key("source", "id", "text", "changed"))

    def test_continuous_log_f0_interpolates_unvoiced_gap(self):
        import numpy as np

        f0 = np.array([100.0, 0.0, 0.0, 200.0], dtype=np.float64)
        times = np.arange(4, dtype=np.float64)
        result = continuous_log_f0(f0, times, times)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(float(result[0]), float(np.log(100.0)), places=5)
        self.assertAlmostEqual(float(result[-1]), float(np.log(200.0)), places=5)
        self.assertGreater(float(result[1]), float(result[0]))
        self.assertLess(float(result[2]), float(result[-1]))

    def test_continuous_log_f0_rejects_fully_unvoiced_audio(self):
        import numpy as np

        values = np.zeros(3, dtype=np.float64)
        with self.assertRaises(AudioRejected):
            continuous_log_f0(values, np.arange(3), np.arange(3))

    def test_vocabulary_and_blank_insertion(self):
        vocabulary = build_vocabulary([["a", "b"], ["b", "c"]])
        ids = encode_phonemes(["a", "b"], vocabulary)
        self.assertEqual(ids[0], vocabulary["<bos>"])
        self.assertEqual(ids[-1], vocabulary["<eos>"])
        self.assertEqual(ids[2], vocabulary["<pad>"])
        self.assertEqual(ids[4], vocabulary["<pad>"])

    def test_ipa_formatting_joiner_is_not_a_phoneme(self):
        symbols = ipa_to_symbols("t\u200dʃ o\u200dʊ ˈɑː")
        self.assertNotIn("\u200d", symbols)
        self.assertIn(" ", symbols)
        self.assertIn("ˈ", symbols)
        self.assertIn("ː", symbols)

        vocabulary = build_vocabulary([symbols])
        self.assertNotIn("\u200d", vocabulary)
        self.assertEqual(len(encode_phonemes(symbols, vocabulary)), 2 * len(symbols) + 2)

    def test_espeak_clause_wrapping_preserves_records(self):
        try:
            executable = find_espeak()
        except RuntimeError:
            self.skipTest("eSpeak-ng is not installed")
        sequences, backend = phonemize_batch(
            [
                "First clause, followed by a second clause.",
                "Another record, with several, comma-separated clauses.",
            ],
            PreprocessConfig().text,
            executable,
        )
        self.assertEqual(len(sequences), 2)
        self.assertTrue(all(sequences))
        self.assertTrue(all("\u200d" not in sequence for sequence in sequences))
        self.assertEqual(backend["backend"], "espeak-ng-cli")


class ManifestTests(unittest.TestCase):
    def test_ljspeech_parser_disables_quote_handling(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "wavs").mkdir()
            (root / "metadata.csv").write_text(
                'a|A sentence with an "unbalanced quote.\n', encoding="utf-8"
            )
            records = load_ljspeech(root)
            self.assertEqual(len(records), 1)
            self.assertIn('"unbalanced', records[0].text)

    def test_split_is_deterministic_and_disjoint(self):
        records = [make_record(index) for index in range(20)]
        first = assign_splits(records, num_validation=3, num_test=4, seed=1234)
        second = assign_splits(records, num_validation=3, num_test=4, seed=1234)
        self.assertEqual(
            [(r.utterance_id, r.split) for r in first],
            [(r.utterance_id, r.split) for r in second],
        )
        counts = {split: sum(r.split == split for r in first) for split in ("train", "validation", "test")}
        self.assertEqual(counts, {"train": 13, "validation": 3, "test": 4})

    def test_split_must_leave_training_data(self):
        records = [make_record(index) for index in range(4)]
        with self.assertRaises(ValueError):
            assign_splits(records, num_validation=2, num_test=2, seed=1234)


if __name__ == "__main__":
    unittest.main()
