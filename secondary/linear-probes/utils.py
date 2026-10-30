import argparse
import json
import os
import csv
from dotenv import load_dotenv
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import login
from sklearn.metrics import accuracy_score, fbeta_score, precision_recall_fscore_support
from torch import nn
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer


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

def conll_to_dataset(file_path):
    """
    Read a CoNLL-style token-label file into a Hugging Face Dataset.

    Blank lines separate sentences. The first column is treated as the token
    and the last column is treated as the label.
    """
    sentences = []
    tokens = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                if tokens:
                    sentences.append({
                        "tokens": tokens,
                        "labels": labels
                    })
                    tokens = []
                    labels = []
                continue

            columns = line.split()
            tokens.append(columns[0])
            labels.append(columns[-1].upper())

    if tokens:
        sentences.append({
            "tokens": tokens,
            "labels": labels
        })
    
    ds = Dataset.from_list(sentences)
    return ds


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


def login_to_huggingface(env_file: Path) -> str | None:
    if env_file.exists():
        load_dotenv(env_file)
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("huggingface_login=skipped token_env_missing")
        return None
    login(token=token, add_to_git_credential=False)
    print("huggingface_login=ok")
    return token

def load_tokenizer_and_model(
    model_id: str, local_files_only: bool, token: str | None
) -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, add_prefix_space=True, local_files_only=local_files_only, token=token
    )
    model = AutoModel.from_pretrained(
        model_id, local_files_only=local_files_only, token=token
    )
    return tokenizer, model

def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )

    f05 = fbeta_score(
        y_true,
        y_pred,
        beta=0.5,
        pos_label=1,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0.5": f05,
    }
