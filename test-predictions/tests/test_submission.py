import json
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TEST_DIR.parent
REPO_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import reconstruct_from_tsv
import submission


class SubmissionTests(unittest.TestCase):
    def test_unlabelled_reconstruction_and_exact_tsv_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "test_unlabelled.tsv"
            source.write_text("Hello\n\\\"\n\nworld\n", encoding="utf-8")

            sentences = reconstruct_from_tsv.build(source, unlabelled=True)
            self.assertEqual([["Hello", '"'], ["world"]], [u["tokens"] for u in sentences])
            self.assertTrue(all("labels" not in unit for unit in sentences))

            data = [
                [{"word_id": 0, "probability": 0.9},
                 {"word_id": 1, "probability": 0.1}],
                [{"word_id": 0, "probability": 0.5}],
            ]
            data_path = root / "data.json"
            data_path.write_text(json.dumps(data), encoding="utf-8")
            predictions = submission.probabilities_to_predictions(data_path, sentences, 0.5)
            out = root / "prediction.tsv"
            submission.write_submission(source, sentences, predictions, out)

            self.assertEqual("Hello\tc\n\\\"\ti\n\nworld\tc\n", out.read_text(encoding="utf-8"))
            self.assertEqual(
                {"lines": 4, "sentences": 2, "tokens": 3},
                submission.validate_submission(source, out),
            )

    def test_writer_replaces_existing_gold_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dev.tsv"
            source.write_text("token\ti\n", encoding="utf-8")
            sentences = reconstruct_from_tsv.build(source)
            predictions = [{"unit_id": 0, "tokens": ["token"], "predicted_labels": ["c"]}]
            out = root / "prediction.tsv"
            submission.write_submission(source, sentences, predictions, out)
            self.assertEqual("token\tc\n", out.read_text(encoding="utf-8"))
            submission.validate_submission(source, out)

    def test_validator_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.tsv"
            hyp = root / "hyp.tsv"
            source.write_text("one\n\ntwo\n", encoding="utf-8")
            hyp.write_text("one\tc\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line-count mismatch"):
                submission.validate_submission(source, hyp)

    def test_out_of_range_word_id_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.json"
            data_path.write_text(
                json.dumps([[{"word_id": 2, "probability": 0.1}]]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "out-of-range word_id"):
                submission.probabilities_to_predictions(
                    data_path, [{"tokens": ["one"]}], threshold=0.5
                )

    def test_threshold_is_read_from_standard_eval_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            result = {
                "meta": {
                    "dataset": "unrelated-name-is-not-interpreted",
                    "source": "/some/dev/data.json",
                    "threshold": 0.2,
                },
                "global": {"f0.5": 0.75},
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")

            threshold, loaded = submission.threshold_from_result(result_path)

            self.assertEqual(0.2, threshold)
            self.assertEqual(result, loaded)

    def test_threshold_result_requires_numeric_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps({
                "meta": {"threshold": "auto"},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing numeric meta.threshold"):
                submission.threshold_from_result(result_path)


if __name__ == "__main__":
    unittest.main()
