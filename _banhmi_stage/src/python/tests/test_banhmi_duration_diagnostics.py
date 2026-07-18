import unittest

from banhmi_tts.duration_diagnostics import build_token_rows, summarize_samples


class DurationDiagnosticsTest(unittest.TestCase):
    def test_build_token_rows_labels_blank_and_accumulates_time(self):
        rows = build_token_rows(
            [1, 17, 0, 2],
            {0: "<pad>", 1: "<bos>", 2: "<eos>", 17: "s"},
            [0.0, 1.0, 0.5, 0.0],
            [1, 3, 2, 1],
            milliseconds_per_frame=10.0,
            maximum_duration_frames=500,
        )
        self.assertEqual(rows[2]["symbol"], "<blank>")
        self.assertEqual(rows[2]["kind"], "blank")
        self.assertEqual(rows[1]["milliseconds"], 30.0)
        self.assertEqual(rows[-1]["end_milliseconds"], 70.0)

    def test_summarize_samples_reports_duration_variation(self):
        token = lambda symbol, kind, frames: {
            "symbol": symbol,
            "kind": kind,
            "frames": frames,
        }
        samples = [
            {
                "total_frames": 6,
                "duration_seconds": 0.06,
                "blank_frame_ratio": 2 / 6,
                "word_boundary_frame_ratio": 0.0,
                "tokens": [token("s", "phoneme", 4), token("<blank>", "blank", 2)],
            },
            {
                "total_frames": 10,
                "duration_seconds": 0.10,
                "blank_frame_ratio": 2 / 10,
                "word_boundary_frame_ratio": 0.0,
                "tokens": [token("s", "phoneme", 8), token("<blank>", "blank", 2)],
            },
        ]
        summary = summarize_samples(samples)
        self.assertEqual(summary["total_frames"]["mean"], 8.0)
        self.assertEqual(summary["tokens_by_duration_variation"][0]["symbol"], "s")
        self.assertEqual(summary["tokens_by_duration_variation"][0]["std_frames"], 2.0)


if __name__ == "__main__":
    unittest.main()
