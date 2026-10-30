"""Objective arithmetic on hand-built tensors."""
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import objectives  # noqa: E402

import unittest


def _batch(examples, row_of):
    return {"examples": examples, "row_of": row_of}

def _ul_example(labels, edits):
    return {"record": {"labels": labels, "edits": edits}}

class ObjectivesTest(unittest.TestCase):
    def test_slot_is_zero_when_the_target_clears_the_margin(self):
        log_probs = torch.log(torch.tensor([[0.01, 0.99]]))
        example = {"slots": [{"position": 3, "source_id": 0, "target_id": 1,
                              "pair": "1:1"}]}
        value, counts = objectives.slot_loss(log_probs, _batch([example], {(0, 3): 0}))
        assert value.item() == 0.0
        assert counts == {"1:1": 1, "n:m": 0}

    def test_slot_penalises_a_wrong_token_scoring_above_the_right_one(self):
        log_probs = torch.log(torch.tensor([[0.9, 0.1]]))
        example = {"slots": [{"position": 3, "source_id": 0, "target_id": 1,
                              "pair": "1:1"}]}
        value, _ = objectives.slot_loss(log_probs, _batch([example], {(0, 3): 0}))
        expected = 1.0 + math.log(0.9) - math.log(0.1)
        self.assertAlmostEqual(value.item(), max(0.0, expected), places=5)
        assert value.item() > 0

    def test_slot_is_invariant_to_flattening_the_distribution(self):
        """A uniform log-shift cancels, so collapse cannot satisfy this term."""
        example = {"slots": [{"position": 3, "source_id": 0, "target_id": 1,
                              "pair": "1:1"}]}
        batch = _batch([example], {(0, 3): 0})
        sharp = torch.log(torch.tensor([[0.6, 0.2]]))
        flat = sharp - 2.0
        assert torch.allclose(objectives.slot_loss(sharp, batch)[0],
                              objectives.slot_loss(flat, batch)[0])

    def test_unlikelihood_rises_as_the_error_probability_rises(self):
        edits = [{"included_in_ged": True, "label_indices": [0],
                  "sentence_final_missing": False}]
        batch = _batch([_ul_example(["i"], edits)], {})
        low = objectives.unlikelihood_loss([[torch.log(torch.tensor(0.01))]], batch)[0]
        high = objectives.unlikelihood_loss([[torch.log(torch.tensor(0.9))]], batch)[0]
        assert high.item() > low.item()

    def test_unlikelihood_covers_insertions_via_the_anchor_word(self):
        insertion = {"included_in_ged": True, "label_indices": [1],
                     "sentence_final_missing": False, "operation": "M"}
        batch = _batch([_ul_example(["c", "i"], [insertion])], {})
        scores = [[torch.log(torch.tensor(0.5)), torch.log(torch.tensor(0.5))]]
        assert objectives.unlikelihood_loss(scores, batch)[0] is not None

    def test_unlikelihood_skips_sentence_final_insertions(self):
        edit = {"included_in_ged": True, "label_indices": [],
                "sentence_final_missing": True}
        batch = _batch([_ul_example(["c"], [edit])], {})
        value, skipped = objectives.unlikelihood_loss(
            [[torch.log(torch.tensor(0.5))]], batch)
        assert value is None and skipped == 1

    def test_unlikelihood_ignores_edits_outside_the_ged_policy(self):
        edit = {"included_in_ged": False, "label_indices": [0],
                "sentence_final_missing": False}
        batch = _batch([_ul_example(["i"], [edit])], {})
        assert objectives.unlikelihood_loss([[torch.log(torch.tensor(0.5))]], batch)[0] is None

    def test_keep_is_a_flat_mean_over_correct_words_only(self):
        example = {"record": {"labels": ["c", "i", "c"], "edits": []}}
        scores = [[torch.log(torch.tensor(0.5)), torch.log(torch.tensor(0.01)),
                   torch.log(torch.tensor(0.5))]]
        value = objectives.keep_loss(scores, _batch([example], {}))
        assert abs(value.item() - (-math.log(0.5))) < 1e-5

    def test_keep_gradient_is_flat_in_log_probability_space(self):
        """The self-concentration is NOT visible here -- it is d/d(logP) = -1/N."""
        example = {"record": {"labels": ["c", "c"], "edits": []}}
        easy = torch.tensor(math.log(0.9), requires_grad=True)
        hard = torch.tensor(math.log(0.05), requires_grad=True)
        objectives.keep_loss([[easy, hard]], _batch([example], {})).backward()
        self.assertAlmostEqual(easy.grad.item(), hard.grad.item(), places=6)

    def test_keep_concentrates_gradient_on_the_damaged_word_at_the_logits(self):
        """d(-log p_t)/d(logit_t) = p_t - 1, so a badly-scoring word dominates."""
        example = {"record": {"labels": ["c", "c"], "edits": []}}
        logits = torch.zeros(2, 100)
        logits[0, 0] = math.log(0.9 / 0.1 * 99)    # word 0 scores ~.9
        logits[1, 0] = math.log(0.05 / 0.95 * 99)  # word 1 scores ~.05
        logits.requires_grad_(True)
        log_probs = torch.log_softmax(logits, dim=-1)
        objectives.keep_loss([[log_probs[0, 0], log_probs[1, 0]]],
                             _batch([example], {})).backward()
        easy, hard = abs(logits.grad[0, 0].item()), abs(logits.grad[1, 0].item())
        assert hard > 5 * easy, (easy, hard)

    def test_combine_treats_absent_components_as_zero(self):
        reference = torch.tensor(3.0)
        total, resolved = objectives.combine(
            {"ul": torch.tensor(2.0)}, {"slot": .5, "ul": .5, "keep": .5}, reference)
        assert total.item() == 1.0
        assert resolved["slot"].item() == 0.0


if __name__ == "__main__":
    unittest.main()
