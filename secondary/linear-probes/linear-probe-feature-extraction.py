import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import (
    choose_device,
    conll_to_dataset,
    load_tokenizer_and_model,
    login_to_huggingface,
    tsv_to_dataset,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reconstruct_from_tsv import detokenise, positions_for_words  # noqa: E402


LABEL2ID = {"c": 0, "i": 1}
DEFAULT_MODEL_ID = "answerdotai/ModernBERT-base"
DEFAULT_BATCH_SIZE = 32


def build_collate_fn(tokenizer):
    def collate_fn(batch):
        tokens = [item["tokens"] for item in batch]
        labels = [item["labels"] for item in batch]

        sentence_texts = []
        word_spans = []

        for sentence_tokens in tokens:
            sentence_text, sentence_word_spans = detokenise(sentence_tokens)
            sentence_texts.append(sentence_text)
            word_spans.append(sentence_word_spans)

        encoding = tokenizer(
            sentence_texts,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        encoding["word_labels"] = labels
        encoding["word_spans"] = word_spans
        return encoding

    return collate_fn


def extract_features(
    data_path,
    feature_dir,
    tokenizer,
    model,
    device,
    model_id,
    batch_size=DEFAULT_BATCH_SIZE,
    label2id=LABEL2ID,
):
    data_path = Path(data_path)
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    print(f"extracting data_path={data_path}")
    print(f"feature_dir={feature_dir}")

    if data_path.suffix == ".tsv":
        data = tsv_to_dataset(data_path)
    else:
        data = conll_to_dataset(data_path)
    data = data.map(lambda x: {"labels": [label2id[l] for l in x["labels"]]})

    max_length = tokenizer.model_max_length
    kept_indices = []
    skipped_words = 0
    for index, item in enumerate(data):
        text, _ = detokenise(item["tokens"])
        length = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        if length <= max_length:
            kept_indices.append(index)
        else:
            skipped_words += len(item["labels"])
    skipped_sentences = len(data) - len(kept_indices)
    if skipped_sentences:
        print(
            f"skipping {skipped_sentences} sentences ({skipped_words} words) "
            f"over model_max_length={max_length}"
        )
        data = data.select(kept_indices)

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        collate_fn=build_collate_fn(tokenizer),
    )
    total_batches = len(dataloader)
    total_words = sum(len(item["labels"]) for item in data)

    hidden_size = model.config.hidden_size
    num_hidden_states = model.config.num_hidden_layers + 1

    layer_arrays = [
        np.lib.format.open_memmap(
            feature_dir / f"layer_{layer_idx:02d}.npy",
            mode="w+",
            dtype=np.float16,
            shape=(total_words, hidden_size),
        )
        for layer_idx in range(num_hidden_states)
    ]

    labels_array = np.lib.format.open_memmap(
        feature_dir / "labels.npy",
        mode="w+",
        dtype=np.int64,
        shape=(total_words,),
    )

    write_pos = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            print(f"batch {batch_idx}/{total_batches}")
            labels = batch.pop("word_labels")
            word_spans = batch.pop("word_spans")
            offset_mapping = batch.pop("offset_mapping")

            batch = batch.to(device)

            output = model(**batch, output_hidden_states=True)

            batch_word_labels = []
            for sentence_labels in labels:
                batch_word_labels.extend(sentence_labels)

            batch_word_count = len(batch_word_labels)
            batch_start = write_pos
            batch_end = batch_start + batch_word_count

            labels_array[batch_start:batch_end] = np.array(
                batch_word_labels,
                dtype=np.int64,
            )

            for layer_idx, layer in enumerate(output.hidden_states):
                layer_write_pos = batch_start

                for i in range(layer.shape[0]):
                    word_positions = positions_for_words(
                        offsets=offset_mapping[i].tolist(),
                        word_spans=word_spans[i],
                    )
                    if any(not positions for positions in word_positions):
                        raise ValueError("a dataset word has no overlapping model token")
                    word_vectors = torch.stack([
                        layer[i, positions[-1]] for positions in word_positions
                    ])

                    assert len(labels[i]) == word_vectors.shape[0]

                    n_words = word_vectors.shape[0]
                    layer_arrays[layer_idx][
                        layer_write_pos:layer_write_pos + n_words
                    ] = word_vectors.detach().cpu().half().numpy()

                    layer_write_pos += n_words

                assert layer_write_pos == batch_end

            write_pos = batch_end

            del output
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    assert write_pos == total_words

    for layer_array in layer_arrays:
        layer_array.flush()
    labels_array.flush()

    metadata = {
        "label2id": label2id,
        "model_id": model_id,
        "source_path": str(data_path),
        "total_words": total_words,
        "hidden_size": hidden_size,
        "num_hidden_states": num_hidden_states,
        "feature_dtype": "float16",
        "label_dtype": "int64",
        "skipped_overlength_sentences": skipped_sentences,
        "skipped_overlength_words": skipped_words,
    }

    torch.save(metadata, feature_dir / "metadata.pt")
    print(f"saved metadata={feature_dir / 'metadata.pt'}")


def parse_args():
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data-path",
        type=Path,
        help="Path to one CoNLL file to extract.",
    )
    input_group.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing CoNLL files to extract per file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output feature directory.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Extraction batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = login_to_huggingface(Path(".env"))

    device = choose_device(args.device)
    print(f"device={device}")
    print(f"model_id={args.model_id}")

    tokenizer, model = load_tokenizer_and_model(
        model_id=args.model_id,
        local_files_only=False,
        token=hf_token,
    )
    model = model.to(device)
    model.eval()

    if args.data_path:
        extract_features(
            data_path=args.data_path,
            feature_dir=args.output_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            model_id=args.model_id,
            batch_size=args.batch_size,
        )
        return

    conll_paths = sorted(args.data_dir.glob("*.conll"))
    if not conll_paths:
        raise FileNotFoundError(f"No .conll files found in {args.data_dir}")

    for conll_path in conll_paths:
        extract_features(
            data_path=conll_path,
            feature_dir=args.output_dir / conll_path.stem,
            tokenizer=tokenizer,
            model=model,
            device=device,
            model_id=args.model_id,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
