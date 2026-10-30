"""Compare repeated GED-by-instruction runs of the same (dataset, model).

Greedy decoding (`temperature=0`) is *not* on its own a guarantee of a
reproducible run. vLLM batches continuously, so the set of sequences sharing a
forward pass differs between runs; that changes the shapes fed to the GEMMs and
hence the floating-point reduction order, and two logits within one ULP of each
other can swap places under argmax. The constrained-decoding grammar then
amplifies any single flipped label into the rest of that sentence's mask. So
"we set temperature to 0" is a claim about the sampler, and this script is what
turns it into a claim about the output.

Runs are grouped by (dataset stem, model id, everything but the tag/timestamp)
and compared against the OLDEST run in each group -- the original result the
repeats are meant to reproduce. Reported per pair:

  sent diff   sentences whose full label array differs (the unit a reader cares
              about: one flipped token makes the sentence's prediction differ)
  tok diff    differing token labels / all token labels, since a single flip in
              a 4,000-sentence corpus and a systematic shift look identical at
              sentence granularity otherwise
  F0.5        each run's scored F0.5 from standard-eval/results/, when scored.
              Identical labels must give identical F0.5; non-identical labels
              giving identical F0.5 is a coincidence worth seeing, not evidence.

A unit whose predicted_labels is null (a parse failure) compares equal only to
another null in the same position -- a run that failed to parse where the
baseline succeeded is a difference, not a missing value.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
import paths  # noqa: E402

OUTPUTS = paths.APPROACH_DIRS["instruction"] / "outputs"

# A run folder is <stem>__<model>[__<tag>]__<timestamp>. The tag is what
# distinguishes concurrent repeats, so it is stripped alongside the timestamp:
# rep1 and rep2 of one system belong in the same comparison group.
_TIMESTAMP = re.compile(r"__\d{8}-\d{6}$")
_TAG = re.compile(r"__rep\d+$")


def _system_key(run_dir: Path):
    """Run folder -> (stem, system) with tag and timestamp removed, or None."""
    name = _TIMESTAMP.sub("", run_dir.name)
    if name == run_dir.name:
        return None  # not a timestamped run folder
    name = _TAG.sub("", name)
    stem, _, system = name.partition("__")
    return (stem, system) if system else None


def _labels(run_dir: Path):
    """[(labels or None) per unit] for one run, plus its meta."""
    data = json.loads((run_dir / "data.json").read_text())
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return [s.get("predicted_labels") for s in data], meta


def _f05(run_dir: Path):
    """Scored F0.5 for this run from standard-eval/results/, or None."""
    path = paths.RESULTS_DIR / f"instruction__{run_dir.name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["global"]["f0.5"]


def _diff(base, other):
    """(sentences differing, tokens differing, tokens compared)."""
    sent_diff = tok_diff = tok_total = 0
    for a, b in zip(base, other):
        if a == b:
            tok_total += len(a) if a is not None else 0
            continue
        sent_diff += 1
        # Lengths are fixed by the dataset, so they only disagree when one side
        # is a parse failure; count the whole sentence as differing there rather
        # than zipping a null against a list.
        if a is None or b is None or len(a) != len(b):
            n = len(a) if a is not None else (len(b) if b is not None else 0)
            tok_total += n
            tok_diff += n
            continue
        tok_total += len(a)
        tok_diff += sum(1 for x, y in zip(a, b) if x != y)
    return sent_diff, tok_diff, tok_total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None,
                        help="only groups whose system contains this, e.g. tiny-aya-global.")
    parser.add_argument("--dataset", default=None,
                        help="only this dataset stem, e.g. en_fce_dev.")
    args = parser.parse_args()

    groups = defaultdict(list)
    for run_dir in sorted(OUTPUTS.iterdir()):
        if not (run_dir / "data.json").exists():
            continue
        key = _system_key(run_dir)
        if key is None:
            continue
        if args.dataset and key[0] != args.dataset:
            continue
        if args.model and args.model not in key[1]:
            continue
        groups[key].append(run_dir)

    if not groups:
        print("no matching runs found under", OUTPUTS)
        return

    all_identical = True
    compared_any = False
    for (stem, system), runs in sorted(groups.items()):
        print(f"\n=== {stem}  /  {system}  ({len(runs)} run(s)) ===")
        if len(runs) < 2:
            print("  only one run -- nothing to compare")
            continue
        compared_any = True
        base_dir = runs[0]
        base, base_meta = _labels(base_dir)
        b_f05 = _f05(base_dir)
        print(f"  baseline  {base_dir.name}   units={len(base)}"
              + (f"  F0.5={b_f05*100:.2f}" if b_f05 is not None else "  F0.5=(unscored)"))
        for run_dir in runs[1:]:
            other, meta = _labels(run_dir)
            if len(other) != len(base):
                print(f"  {run_dir.name}: UNIT COUNT MISMATCH "
                      f"({len(other)} vs {len(base)}) -- not the same corpus")
                all_identical = False
                continue
            sent_diff, tok_diff, tok_total = _diff(base, other)
            f05 = _f05(run_dir)
            verdict = "IDENTICAL" if sent_diff == 0 else "DIFFERS"
            if sent_diff:
                all_identical = False
            fails = sum(meta.get(k, 0) for k in
                        ("json_failures", "schema_failures", "length_failures"))
            base_fails = sum(base_meta.get(k, 0) for k in
                             ("json_failures", "schema_failures", "length_failures"))
            print(f"  {verdict:9s} {run_dir.name}")
            print(f"            sent diff {sent_diff}/{len(base)}"
                  f"   tok diff {tok_diff}/{tok_total}"
                  f"   parse failures {fails} (baseline {base_fails})"
                  + (f"   F0.5={f05*100:.2f}" if f05 is not None else "   F0.5=(unscored)"))

    if compared_any:
        print("\n" + ("ALL COMPARED RUNS ARE LABEL-IDENTICAL."
                      if all_identical else
                      "AT LEAST ONE REPEAT DIFFERS -- greedy decoding is not "
                      "bit-reproducible here."))


if __name__ == "__main__":
    main()
