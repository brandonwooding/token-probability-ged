#!/usr/bin/env python3
"""Fast structural tests for filtering and rich-record projection."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("c4_merrant_build", HERE.parent / "build.py")
BUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


class OutputPolicyTest(unittest.TestCase):
    def test_excluded_types(self):
        for label in ("UNK", "R:UNK", "U:PUNCT", "M:PUNCT", "R:PUNCT", "noop"):
            self.assertTrue(BUILD.is_excluded(label), label)
        for label in ("U:DET", "M:DET", "R:VERB:TENSE", "U:OTHER"):
            self.assertFalse(BUILD.is_excluded(label), label)

    def test_edit_projection(self):
        class Span:
            def __init__(self, start, end):
                self.start, self.end = start, end

        class Edit:
            def __init__(self, ss, se, ts, te, label):
                self.source_span, self.target_span = Span(ss, se), Span(ts, te)
                self.label = label

            def get_label(self):
                return self.label

            def is_deletion(self):
                return self.target_span.start == self.target_span.end

            def is_insertion(self):
                return self.source_span.start == self.source_span.end

        self.assertEqual(BUILD.edit_record(Edit(1, 3, 1, 2, "R:VERB"), 5)["label_indices"],
                         [1, 2])
        self.assertEqual(BUILD.edit_record(Edit(2, 2, 2, 3, "M:ADV"), 5)["label_indices"],
                         [2])
        final = BUILD.edit_record(Edit(5, 5, 4, 5, "M:NOUN"), 5)
        self.assertEqual(final["label_indices"], [])
        self.assertTrue(final["sentence_final_missing"])
        punct = BUILD.edit_record(Edit(2, 3, 2, 2, "U:PUNCT"), 5)
        self.assertFalse(punct["included_in_ged"])
        self.assertEqual(punct["label_indices"], [])


if __name__ == "__main__":
    unittest.main()
