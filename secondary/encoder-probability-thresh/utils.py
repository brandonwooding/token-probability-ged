from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
from pathlib import Path
import json
from datasets import Dataset
import csv

def login_to_huggingface(env_file: Path) -> str | None:
    if env_file.exists():
        load_dotenv(env_file)
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("huggingface_login=skipped token_env_missing")
        return None
    try:
        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        print(f"huggingface_login=failed {type(exc).__name__}: {exc}")
        return token
    print("huggingface_login=ok")
    return token


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device

def jsonl_to_list(file_path):
    all_data = []

    with open(file_path, 'r') as file:
        for line in file:

            if line.strip(): 
                row_dict = json.loads(line)
                all_data.append(row_dict)

    return [row["text"].split('\n\n') for row in all_data]

def tsv_to_dataset(file_path):
    """
    Inputs: 
        file_path (string): path to .tsv file with labelled tokens, paragraphs separated by blank line.
    Outputs:
        ds (Hugging Face dataset object)
    """
    with open(file_path, "r") as f:
        my_data = csv.reader(f, delimiter="\t", quotechar='"', escapechar="\\",)
        data = []
        sentence = []
        for line in my_data:
            if line and line[0].strip():
                sentence.append(line)
            else:
                if sentence:
                    data.append(sentence)
                sentence = []
        if sentence:
            data.append(sentence)

    dataset = []
    for para in data:
        tokens, labels = zip(*para)
        dataset.append({"tokens": list(tokens), "labels": list(labels)})

    ds = Dataset.from_list(dataset)
    return ds

def get_custom_word_ids(input_ids, offset_mapping, tsv_spans, special_ids):
    word_ids = []
    for token_id, (char_start, char_end) in zip(input_ids, offset_mapping):
        if token_id.item() in special_ids:
            word_ids.append(None)
            continue
        
        char_start, char_end = char_start.item(), char_end.item()
        
        if char_start == char_end:
            word_ids.append(None)
            continue
        
        matched = None
        for i, (s, e) in enumerate(tsv_spans):
            if char_start < e and char_end > s:
                matched = i
                break
        word_ids.append(matched)
    return word_ids