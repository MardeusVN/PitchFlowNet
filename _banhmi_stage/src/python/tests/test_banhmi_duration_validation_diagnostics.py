import unittest

from banhmi_tts.duration_validation_diagnostics import (
    grouped_summaries,
    summarize_comparisons,
    summarize_utterances,
)


class DurationValidationDiagnosticsTest(unittest.TestCase):
    def test_summarize_comparisons_reports_signed_and_absolute_error(self):
        rows = [
            {"kind": "phoneme", "target_frames": 2, "predicted_frames": 4},
            {"kind": "blank", "target_frames": 4, "predicted_frames": 3},
        ]
        summary = summarize_comparisons(rows)
        self.assertEqual(summary["mean_error_frames"], 0.5)
        self.assertEqual(summary["mae_frames"], 1.5)
        self.assertAlmostEqual(summary["rmse_frames"], (2.5) ** 0.5)

        groups = grouped_summaries(rows, "kind")
        self.assertEqual(groups["blank"]["mean_error_frames"], -1.0)
        self.assertEqual(groups["phoneme"]["mean_error_frames"], 2.0)

    def test_summarize_utterances_reports_total_ratio(self):
        rows = [
            {"predicted_to_target_ratio": 0.8, "duration_error_seconds": -0.2},
            {"predicted_to_target_ratio": 1.2, "duration_error_seconds": 0.3},
        ]
        summary = summarize_utterances(rows)
        self.assertEqual(summary["predicted_to_target_ratio_mean"], 1.0)
        self.assertEqual(summary["duration_error_seconds_mae"], 0.25)


if __name__ == "__main__":
    unittest.main()
