import argparse
import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
import paths  # noqa: E402

MODEL_NAME = "google/gemma-4-12B-it"

# The only labels the schema admits. A tuple rather than a set on purpose: the
# membership test in run_ged runs over whatever the model actually returned, and
# an unhashable element (a nested list) has to compare False there, not raise.
LABELS = ("c", "i")

# vLLM sizes the KV cache to hold max_model_len tokens per sequence and refuses
# to start when it cannot. Left unset it takes the model's advertised context --
# 262,144 for gemma-4-12B-it and Qwen3.5-9B -- which needs 8-16 GB of KV beside
# the weights. That fits a 48 GB L40S by luck, but not a 40 GB A100 (gemma: 13 GB
# left after weights, 16 GB needed) or a 24 GB RTX6000. The actual requirement is
# tiny: the longest MultiGED dev sentence is 133 tokens, so a prompt is under 1k
# tokens and 8192 clears prompt + max_tokens with room to spare.
MAX_MODEL_LEN = 8192

# The corpus language is named in the system prompt, so the same prompt works
# across the multilingual MultiGED datasets; it comes from the dataset id's
# language folder (english/ -> "English", german/ -> "German").
SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert annotator for grammatical error detection (GED) on {language} "
    "learner text. You will be given a list of spaCy-tokenized units from a "
    "sentence, in order. For each token, decide whether it is part of an error "
    "(spelling, grammar, wrong word choice, or a missing/extra word signalled by "
    "the surrounding tokens) or not.\n\n"
    "When a word is missing from the text, label the token immediately AFTER "
    "the gap as 'i' (the token before which the missing word should be inserted).\n\n"
    "Respond with a JSON array of exactly the same length as the token list, "
    'containing only the strings "c" (correct) or "i" (incorrect), in the same '
    "order as the tokens. Do not add, remove, merge, or reorder any tokens."
)


def build_messages(text: str, tokens: list[str], language: str) -> list[dict]:
    user_prompt = (
        f"Sentence text:\n{text}\n\n"
        f"Tokens ({len(tokens)} total): {json.dumps(tokens)}\n\n"
        "Return the JSON array of labels now."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(language=language)},
        {"role": "user", "content": user_prompt},
    ]


def build_sampling_params(num_tokens: int) -> SamplingParams:
    schema = {
        "type": "array",
        "items": {"enum": list(LABELS)},
        "minItems": num_tokens,
        "maxItems": num_tokens,
    }
    return SamplingParams(
        temperature=0,
        max_tokens=4096,
        structured_outputs=StructuredOutputsParams(json=schema),
    )


def run_ged(dataset: str, model_name: str, tag: str | None = None):
    model_id = model_name.split("/")[-1]

    sentences = json.loads(paths.sentences(dataset).read_text())
    language = paths.language(dataset).capitalize()

    conversations = [build_messages(s["text"], s["tokens"], language) for s in sentences]
    sampling_params = [build_sampling_params(len(s["tokens"])) for s in sentences]

    # minItems/maxItems makes the grammar mask out "]" until the array holds
    # num_tokens elements -- but JSON permits whitespace between elements, so a
    # model that believes it has finished still has a legal token to emit, and
    # emits it forever: the array never closes and the rest of the generation
    # budget goes on newlines. That was 9-18% of sentences on the gemma-4-12B-it
    # runs, and in 100% of those the unlabelled token was sentence-final
    # punctuation the model declined to label. Dropping whitespace from the
    # grammar leaves "," plus the next label as the only legal continuation, so
    # the array always closes. disable_any_whitespace is rejected unless the
    # backend is named explicitly, hence pinning xgrammar rather than "auto".
    model = LLM(
        model=model_name,
        max_model_len=MAX_MODEL_LEN,
        structured_outputs_config={
            "backend": "xgrammar",
            "disable_any_whitespace": True,
        },
    )
    outputs = model.chat(
        conversations,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )

    # Track the failure modes separately so we can tell whether a model is
    # producing invalid JSON (grammar/template problem), valid JSON of the wrong
    # shape (not an array, or labels outside LABELS), or valid JSON of the wrong
    # length (the array length constraint not being enforced by the
    # structured-output backend). On any failure we keep the raw model text and,
    # for length mismatches, the produced length, so the failures can be
    # inspected or recovered later rather than silently discarded.
    #
    # Shape is checked before length because len() is only meaningful once we
    # know we have a list: a bare `null`, `3` or `true` would otherwise raise
    # TypeError here and take down the whole run, which is only written below.
    # The label-value check matters for the same reason -- ["C","I"] is a list
    # of the right length, so without it the run would be scored as if the model
    # had found no errors at all, and reported as a clean run.
    json_failures = 0
    schema_failures = 0
    length_failures = 0
    for sent, output in zip(sentences, outputs):
        raw_text = output.outputs[0].text
        try:
            predicted = json.loads(raw_text)
        except json.JSONDecodeError:
            predicted = None
            json_failures += 1
            sent["failure_mode"] = "json_decode"
            sent["raw_output"] = raw_text
        else:
            if not isinstance(predicted, list) or not all(x in LABELS for x in predicted):
                schema_failures += 1
                sent["failure_mode"] = "schema_mismatch"
                sent["raw_output"] = raw_text
                predicted = None
            elif len(predicted) != len(sent["tokens"]):
                length_failures += 1
                sent["failure_mode"] = "length_mismatch"
                sent["raw_output"] = raw_text
                sent["predicted_length"] = len(predicted)
                predicted = None
        sent["predicted_labels"] = predicted

    failures = json_failures + schema_failures + length_failures
    run_dir = paths.run_dir("instruction", dataset, model_id, tag=tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data.json").write_text(json.dumps(sentences, indent=2))
    (run_dir / "meta.json").write_text(json.dumps({
        "model_name": model_name,
        "model_id": model_id,
        "dataset": dataset,
        "tag": tag,
        "num_sentences": len(sentences),
        "max_model_len": MAX_MODEL_LEN,
        "json_failures": json_failures,
        "schema_failures": schema_failures,
        "length_failures": length_failures,
    }, indent=2))
    print(
        f"Wrote {len(sentences)} sentences to {run_dir} "
        f"({failures} failed: {json_failures} json-decode, "
        f"{schema_failures} schema-mismatch, {length_failures} length-mismatch)."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="dataset id, e.g. english/en_fce_dev (or bare stem).",
    )
    # Two runs of the same (dataset, model) are distinguished only by the run
    # folder's timestamp, which is second-resolution -- so two jobs launched
    # together on identical hardware can in principle land on the same name and
    # one silently overwrites the other. A tag makes the folders distinct by
    # construction. It affects naming only: nothing about inference reads it,
    # so a tagged run is bit-for-bit the same computation as an untagged one.
    parser.add_argument(
        "--tag",
        default=None,
        help="label folded into the run folder name, e.g. rep1. Naming only.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Model name/path to run. Repeat to queue multiple models.",
    )
    args = parser.parse_args()

    models = args.models or [MODEL_NAME]
    for model_name in models:
        run_ged(args.dataset, model_name, tag=args.tag)


if __name__ == "__main__":
    main()
