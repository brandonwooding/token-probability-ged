import unittest

import torch

from objectives import auxiliary_losses


class ObjectivesTest(unittest.TestCase):
    def test_default_skips_empty_contrast_but_trains_unlikelihood(self):
        source = [[torch.tensor(-.1, requires_grad=True), torch.tensor(-2., requires_grad=True)]]
        target = [[torch.tensor(-.2), torch.tensor(-.2), torch.tensor(-.2)]]
        record = {"labels": ["c", "i"], "edits": [{
            "source_start": 1, "source_end": 1, "target_start": 1, "target_end": 2,
            "kind": "insertion", "label_indices": [1], "sentence_final_missing": False}]}
        total, parts, counts = auxiliary_losses(
            source, target, [record], empty_policy="skip",
            contrast_weight=1., unlikelihood_weight=1.)
        self.assertEqual(float(parts["contrast"].detach()), 0.)
        self.assertGreater(float(parts["unlikelihood"].detach()), 0.)
        self.assertEqual(counts["skipped_empty_edits"], 1)
        total.backward()
        self.assertIsNotNone(source[0][1].grad)

    def test_direct_contrast_uses_labelled_source_words_only(self):
        source = [[torch.tensor(-.2), torch.tensor(-.1, requires_grad=True)]]
        target = [[torch.tensor(-.05)]]
        record = {"labels": ["c", "i"], "edits": [{
            "source_start": 0, "source_end": 2, "target_start": 0, "target_end": 1,
            "kind": "replacement", "label_indices": [1], "sentence_final_missing": False}]}
        total, _, _ = auxiliary_losses(source, target, [record], contrast_weight=1.)
        total.backward()
        self.assertIsNotNone(source[0][1].grad)

    def test_complete_json_punctuation_edit_is_provenance_only(self):
        source = [[torch.tensor(-.2, requires_grad=True), torch.tensor(-.1)]]
        target = [[torch.tensor(-.05), torch.tensor(-.05)]]
        record = {"labels": ["c", "c"], "edits": [{
            "source_start": 0, "source_end": 1, "target_start": 0, "target_end": 0,
            "kind": "deletion", "label_indices": [], "sentence_final_missing": False,
            "error_type": "U:PUNCT", "included_in_ged": False}]}
        total, parts, counts = auxiliary_losses(
            source, target, [record], contrast_weight=1., unlikelihood_weight=1.)
        self.assertEqual(float(total.detach()), 0.)
        self.assertEqual(float(parts["contrast"].detach()), 0.)
        self.assertEqual(counts["skipped_empty_edits"], 0)


if __name__ == "__main__":
    unittest.main()
