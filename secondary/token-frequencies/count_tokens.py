"""Count unigram token frequencies over a frozen sample. Offline, CPU only.

One table per (model, language). Ten models over five languages is fifty
tables, four of which duplicate another four because gemma-4-12B and its -it
variant share a vocabulary, as do Qwen3.5-9B and its Base. That duplication is
deliberate: a scoring run names a model, and a lookup that has to know which
models are secretly the same model is a lookup that will eventually get it
wrong. Each table costs about a minute, so the redundancy is free.

Three things are done differently from the reference implementation this method
comes from (jennhu/probability-grammaticality, src/estimate_token_freqs.py):

  Keyed by token id, not token string. That repo saves `{token_string: count}`
  via convert_ids_to_tokens, which is lossy for byte-fallback pieces and forces
  a re-tokenisation to look anything up. Every data.json this project has
  written already carries `token_id` per sub-token, so ids join directly. The
  counts are stored as a dense array indexed by id -- shorter than a dict of
  the same data, and it gzips well because the tail is zeros.

  A vocab hash is stored. mmBERT carries a Gemma-2 vocabulary and gemma-4-12B
  its own; "1" is id 235274 in one and 236770 in the other. Same family name,
  different table, and normalising with the wrong one would fail silently
  rather than loudly. Anything reading a table should check the hash against
  the tokenizer the run actually used.

  `total_tokens` is the sum of the counts. The reference implementation writes
  `total = sum(ctr)`, which sums a Counter's *keys* -- the token ids -- so its
  published frequencies are off by a large arbitrary factor. Harmless for the
  rank-based analyses that repo does, but fatal for anything comparing an
  absolute score across languages.

No smoothing is applied. A zero count is recorded as a zero; how to treat an
unseen token is a scoring-time decision, and baking a floor into the table
would hide the coverage number that says whether the sample was big enough.

    python count_tokens.py --all
    python count_tokens.py --model google/gemma-4-12B --language german
"""

import argparse
import gzip
import json
import hashlib
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from sources import MODELS, LANGUAGES, model_id  # noqa: E402

SAMPLES_DIR = SCRIPT_DIR / "samples"
TABLES_DIR = SCRIPT_DIR / "tables"
BATCH_DOCS = 500


def vocab_fingerprint(tokenizer):
    """(size, hash) identifying this exact vocabulary.

    Size is the largest id the tokenizer can emit, not len(vocab): added special
    tokens routinely sit above the base vocabulary, and a counts array shorter
    than the highest id would drop them.
    """
    vocab = tokenizer.get_vocab()
    size = max(len(tokenizer), max(vocab.values()) + 1)
    payload = "\n".join(f"{i}\t{t}" for t, i in sorted(vocab.items(), key=lambda kv: kv[1]))
    return size, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_docs(sample_path, batch_docs=BATCH_DOCS):
    """Yield batches of document texts from a frozen sample."""
    batch = []
    with gzip.open(sample_path, "rt", encoding="utf-8") as f:
        for line in f:
            batch.append(json.loads(line)["text"])
            if len(batch) >= batch_docs:
                yield batch
                batch = []
    if batch:
        yield batch


def count_language(tokenizer, size, sample_path, label):
    import numpy as np

    counts = np.zeros(size, dtype=np.int64)
    t0 = time.time()
    n_docs = 0
    for batch in read_docs(sample_path):
        # add_special_tokens=False because the scored runs skip special tokens
        # too -- a BOS counted here would be a frequency no data.json ever asks
        # for, and would shift every other token's share of the total.
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        flat = np.fromiter((i for ids in encoded for i in ids), dtype=np.int64)
        if flat.size:
            if flat.max() >= counts.size:
                counts = np.concatenate(
                    [counts, np.zeros(flat.max() + 1 - counts.size, dtype=np.int64)])
            counts += np.bincount(flat, minlength=counts.size)
        n_docs += len(batch)
        if n_docs % (BATCH_DOCS * 40) == 0:
            rate = counts.sum() / max(time.time() - t0, 1e-9)
            print(f"  {label}: {n_docs} docs, {counts.sum() / 1e6:.0f}M tokens, "
                  f"{rate / 1e6:.1f}M tok/s", flush=True)
    return counts, n_docs, time.time() - t0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", dest="models", help="repeat for several")
    parser.add_argument("--language", action="append", dest="languages", choices=LANGUAGES)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--samples", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--out", type=Path, default=TABLES_DIR)
    parser.add_argument("--force", action="store_true", help="recount existing tables")
    args = parser.parse_args()

    models = args.models or list(MODELS)
    languages = args.languages or list(LANGUAGES)

    from transformers import AutoTokenizer, logging as hf_logging
    hf_logging.set_verbosity_error()   # silences the >max_length notice; we never truncate

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    # Model outer, language inner: the tokenizer is the expensive thing to
    # construct, so it is built once and reused across all five languages.
    for model in models:
        mid = model_id(model)
        pending = [lang for lang in languages
                   if args.force or not (args.out / f"{mid}__{lang}.freqs.json.gz").exists()]
        if not pending:
            print(f"[{mid}] all tables present, skipping", flush=True)
            continue

        print(f"[{mid}] loading tokenizer", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model)
        size, vhash = vocab_fingerprint(tokenizer)
        print(f"[{mid}] vocab={size} hash={vhash[:12]}", flush=True)

        for language in pending:
            sample_path = args.samples / f"{language}.jsonl.gz"
            manifest_path = args.samples / f"{language}.manifest.json"
            if not sample_path.exists():
                raise SystemExit(f"no sample for {language}: run sample_fineweb.py first")
            manifest = json.loads(manifest_path.read_text())

            counts, n_docs, secs = count_language(tokenizer, size, sample_path,
                                                  f"{mid}/{language}")
            total = int(counts.sum())
            unique = int((counts > 0).sum())
            table = {
                "model": model,
                "model_id": mid,
                "language": language,
                "vocab_size": int(counts.size),
                "vocab_hash": vhash,
                "corpus": {k: manifest[k] for k in
                           ("repo", "revision", "prefix", "n_docs", "n_chars",
                            "shards_sampled", "doc_weighting", "selection")},
                "n_docs": n_docs,
                "total_tokens": total,
                "unique_tokens": unique,
                "vocab_coverage": round(unique / counts.size, 4),
                "smoothing": "none (raw counts; zeros are genuine zeros)",
                "special_tokens_counted": False,
                # Dense, indexed by token id. counts[i] is the number of times
                # token id i occurred; sum(counts) == total_tokens.
                "counts": counts.tolist(),
            }
            out_path = args.out / f"{mid}__{language}.freqs.json.gz"
            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                json.dump(table, f)
            written.append(out_path)
            print(f"[{mid}/{language}] {total / 1e6:.0f}M tokens, {unique} of "
                  f"{counts.size} vocab seen ({unique / counts.size:.1%}), "
                  f"{secs:.0f}s -> {out_path.name} "
                  f"({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

        del tokenizer

    print(f"\nwrote {len(written)} tables to {args.out}", flush=True)


if __name__ == "__main__":
    main()
