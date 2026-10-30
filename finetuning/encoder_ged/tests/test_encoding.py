"""Divergence-point and batching logic, with no model or tokenizer involved."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import encoding  # noqa: E402

import unittest


def _record(edits):
    return {"edits": edits}

def _side(ids, word_positions):
    return {"input_ids": ids, "word_positions": word_positions}

class EncodingTest(unittest.TestCase):
    def test_common_prefix_length(self):
        assert encoding.common_prefix_length([1, 2, 3], [1, 2, 9]) == 2
        assert encoding.common_prefix_length([1], [2]) == 0
        assert encoding.common_prefix_length([1, 2], [1, 2]) == 2
        assert encoding.common_prefix_length([], [1]) == 0

    def test_ordered_unique_keeps_first_occurrence(self):
        assert encoding.ordered_unique([3, 1, 3, 2, 1]) == [3, 1, 2]

    def test_slot_picks_divergence_not_the_first_position(self):
        # shared stem: [stem, wrong] vs [stem, right] -> compare at index 1
        source = _side([0, 10, 11, 1], [[1, 2]])
        target = _side([0, 10, 22, 1], [[1, 2]])
        edit = {"operation": "R", "included_in_ged": True, "source_start": 0,
                "source_end": 1, "target_start": 0, "target_end": 1}
        slots, skipped = encoding._slot_targets(_record([edit]), source, target)
        assert len(slots) == 1
        assert slots[0]["position"] == 2          # the divergent sub-token
        assert slots[0]["source_id"] == 11
        assert slots[0]["target_id"] == 22
        assert slots[0]["pair"] == "1:1"
        assert skipped == {"empty_side": 0, "strict_prefix": 0}

    def test_slot_picks_index_zero_when_the_difference_is_at_the_front(self):
        # shared suffix: last-sub-token comparison would be degenerate here
        source = _side([0, 10, 30, 1], [[1, 2]])
        target = _side([0, 20, 30, 1], [[1, 2]])
        edit = {"operation": "R", "included_in_ged": True, "source_start": 0,
                "source_end": 1, "target_start": 0, "target_end": 1}
        slots, _ = encoding._slot_targets(_record([edit]), source, target)
        assert slots[0]["position"] == 1
        assert (slots[0]["source_id"], slots[0]["target_id"]) == (10, 20)

    def test_slot_never_compares_a_token_against_itself(self):
        source = _side([0, 10, 11, 12, 1], [[1, 2, 3]])
        target = _side([0, 10, 11, 99, 1], [[1, 2, 3]])
        edit = {"operation": "R", "included_in_ged": True, "source_start": 0,
                "source_end": 1, "target_start": 0, "target_end": 1}
        slots, _ = encoding._slot_targets(_record([edit]), source, target)
        assert slots[0]["source_id"] != slots[0]["target_id"]

    def test_slot_skips_strict_prefix(self):
        source = _side([0, 10, 1], [[1]])
        target = _side([0, 10, 11, 1], [[1, 2]])
        edit = {"operation": "R", "included_in_ged": True, "source_start": 0,
                "source_end": 1, "target_start": 0, "target_end": 1}
        slots, skipped = encoding._slot_targets(_record([edit]), source, target)
        assert slots == []
        assert skipped["strict_prefix"] == 1

    def test_slot_skips_insertions_and_deletions(self):
        source = _side([0, 10, 11, 1], [[1], [2]])
        target = _side([0, 10, 11, 1], [[1], [2]])
        insertion = {"operation": "M", "included_in_ged": True, "source_start": 1,
                     "source_end": 1, "target_start": 0, "target_end": 1}
        deletion = {"operation": "U", "included_in_ged": True, "source_start": 0,
                    "source_end": 1, "target_start": 1, "target_end": 1}
        slots, skipped = encoding._slot_targets(
            _record([insertion, deletion]), source, target)
        assert slots == []
        # non-R edits never reach the empty-side branch; they are filtered first
        assert skipped == {"empty_side": 0, "strict_prefix": 0}

    def test_slot_handles_n_to_m_spans(self):
        source = _side([0, 10, 11, 1], [[1, 2]])
        target = _side([0, 20, 21, 22, 1], [[1], [2, 3]])
        edit = {"operation": "R", "included_in_ged": True, "source_start": 0,
                "source_end": 1, "target_start": 0, "target_end": 2}
        slots, _ = encoding._slot_targets(_record([edit]), source, target)
        assert slots[0]["pair"] == "n:m"
        assert (slots[0]["source_id"], slots[0]["target_id"]) == (10, 20)

    def test_slot_ignores_edits_outside_the_ged_policy(self):
        source = _side([0, 10, 1], [[1]])
        target = _side([0, 20, 1], [[1]])
        edit = {"operation": "R", "included_in_ged": False, "source_start": 0,
                "source_end": 1, "target_start": 0, "target_end": 1}
        slots, _ = encoding._slot_targets(_record([edit]), source, target)
        assert slots == []

    def test_iter_batches_budgets_rows_times_width(self):
        # 4 sentences of 30 sub-tokens: rows grow 30/60/90, width stays 30,
        # so 2 fit under 2000 (60*30) but 3 do not (90*30).
        examples = [{"n_copies": 30, "width": 30} for _ in range(4)]
        batches = list(encoding.iter_batches(examples, token_budget=2000))
        assert [len(b) for b in batches] == [2, 2]

    def test_iter_batches_accounts_for_the_widest_member(self):
        """A long sentence raises the padded width for everything beside it."""
        short = {"n_copies": 10, "width": 10}
        long = {"n_copies": 100, "width": 100}
        # short alone: 10*10=100. Adding long: 110 rows * 100 width = 11000.
        batches = list(encoding.iter_batches([short, long], token_budget=5000))
        assert [len(b) for b in batches] == [1, 1]

    def test_iter_batches_never_drops_an_oversized_example(self):
        examples = [{"n_copies": 192, "width": 192}, {"n_copies": 10, "width": 10}]
        batches = list(encoding.iter_batches(examples, token_budget=1000))
        assert sum(len(b) for b in batches) == 2


if __name__ == "__main__":
    unittest.main()
