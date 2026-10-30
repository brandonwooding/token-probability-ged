#!/usr/bin/env python3
"""Upload a finished fine-tuning run to a private Hugging Face repo.

Deliberately separate from training. Training writes to RDS and never touches
the network mid-job: a multi-hour LoRA run should not die on a transient upload
failure, and resume-after-walltime needs the checkpoints local anyway. Publish
when a run is worth keeping, not while it is still being produced.

Repos are created **private** by default. The training data is C4_200M-derived
and the checkpoints are research artefacts; making them public is a decision to
take explicitly with --public, not by omission.

    python push_to_hub.py --model models/<run> --repo <user>/<name>
    python push_to_hub.py --model models/<run> --repo <user>/<name> --public
"""
import argparse
import json
import os

import ft_data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a run dir under models/")
    ap.add_argument("--repo", required=True, help="target repo id, e.g. brandonwooding/token-probability-ged")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public (default: private)")
    ap.add_argument("--message", default=None, help="commit message")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model)
    if not os.path.isdir(model_dir):
        raise SystemExit("not a directory: %s" % model_dir)

    ft_data.hf_login()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("huggingface_hub is required: pip install huggingface_hub")

    meta_path = os.path.join(model_dir, "train_meta.json")
    summary = ""
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        summary = " (%s, langs=%s, n_train=%s)" % (
            meta.get("kind"), ",".join(meta.get("languages", [])), meta.get("n_train"))

    api = HfApi()
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=model_dir,
        repo_id=args.repo,
        commit_message=args.message or ("upload %s%s" % (os.path.basename(model_dir), summary)),
    )
    visibility = "public" if args.public else "private"
    print("uploaded %s -> https://huggingface.co/%s (%s)" % (model_dir, args.repo, visibility))


if __name__ == "__main__":
    main()
