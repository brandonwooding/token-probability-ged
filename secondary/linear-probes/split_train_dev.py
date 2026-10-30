import argparse
import random
from pathlib import Path


def read_sentence_blocks(path):
    """Group raw lines into sentence blocks separated by blank lines.

    Lines are preserved verbatim (including escaped quotes) so the split
    output is byte-identical to the source apart from block ordering.
    """
    blocks = []
    current = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                current.append(line.rstrip("\n"))
            elif current:
                blocks.append(current)
                current = []
        if current:
            blocks.append(current)
    return blocks


def write_blocks(path, blocks):
    with open(path, "w", encoding="utf-8") as f:
        for block in blocks:
            for line in block:
                f.write(line + "\n")
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    blocks = read_sentence_blocks(args.input)
    grouped = {}
    for block in blocks:
        grouped.setdefault(tuple(block), []).append(block)

    groups = list(grouped.values())
    rng = random.Random(args.seed)
    rng.shuffle(groups)

    n_val = round(len(blocks) * args.val_frac)
    train_blocks, val_blocks = [], []
    for group in groups:
        add_to_val = (
            len(val_blocks) < n_val
            and abs(n_val - len(val_blocks) - len(group)) <= abs(n_val - len(val_blocks))
        )
        (val_blocks if add_to_val else train_blocks).extend(group)

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    write_blocks(args.train_out, train_blocks)
    write_blocks(args.val_out, val_blocks)

    print(f"input_sentences={len(blocks)}")
    print(f"train_sentences={len(train_blocks)} -> {args.train_out}")
    print(f"val_sentences={len(val_blocks)} -> {args.val_out}")


if __name__ == "__main__":
    main()
