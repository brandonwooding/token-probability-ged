"""Probe head architectures, shared by the trainer, the evaluator and the adapter.

Three separate places have to build the same module: `linear-probe-predictor.py`
creates it to train, `linear-probe-eval.py` and
`standard-eval/adapt_linear_probes.py` rebuild it to load a checkpoint. A
`state_dict` does not describe its own architecture, so if those three ever
disagree the load raises -- or, worse, quietly succeeds with a different dropout
and reports numbers that are not the ones that were trained.

So the head is written into the experiment's `config.json` when it is trained,
and read back from there whenever a checkpoint is rebuilt. One source, three
readers.

Experiments trained before the mlp head existed have no "head" key in their
config; `read_spec` returns the linear default for them, which is exactly what
they are, so `fce_binary`, `c4200_en` and `c4200_de` keep loading untouched.

## Why an mlp head at all

The linear probe asks whether error information is *linearly* readable from a
frozen representation. When it underperforms, that is two hypotheses at once:
the information may be absent from the representation, or present but not
linearly separable. An mlp on the same cached features separates them, and it
costs nothing extra to find out -- the features are already on disk, so this is
a training job rather than another pass over the encoder. If the mlp does not
beat the linear head, the representation is the ceiling and only unfreezing the
encoder will move it.
"""
import json

import torch

HEAD_LINEAR = "linear"
HEAD_MLP = "mlp"
HEADS = (HEAD_LINEAR, HEAD_MLP)

# Defaults are the linear probe exactly as it was before this module existed:
# `Sequential(Dropout(0.1), Linear(hidden_size, 2))`, whose state_dict keys are
# `1.weight` / `1.bias`. Existing checkpoints must keep loading.
DEFAULT_HEAD = HEAD_LINEAR
DEFAULT_MLP_HIDDEN = 256
DEFAULT_DROPOUT = 0.1


def build_probe(hidden_size, head=DEFAULT_HEAD, mlp_hidden=DEFAULT_MLP_HIDDEN,
                dropout=DEFAULT_DROPOUT, device=None):
    """The probe module for one layer's features.

    `mlp` is deliberately small (one hidden layer, 768->256->2 by default, ~197k
    parameters against the linear head's 1.5k). It is still a probe: the point
    is to test for non-linearly-readable signal, not to train the best possible
    classifier, and a large head would start answering the second question.
    """
    if head == HEAD_LINEAR:
        module = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, 2),
        )
    elif head == HEAD_MLP:
        module = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, mlp_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_hidden, 2),
        )
    else:
        raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
    return module.to(device) if device is not None else module


def spec_from_config(config):
    """Head keyword arguments from a config dict, defaulting to linear."""
    return {
        "head": config.get("head", DEFAULT_HEAD),
        "mlp_hidden": config.get("mlp_hidden", DEFAULT_MLP_HIDDEN),
        "dropout": config.get("dropout", DEFAULT_DROPOUT),
    }


def read_spec(experiment_dir):
    """Head keyword arguments recorded by the run that trained `experiment_dir`.

    A missing config.json is not an error: probes predating this module are
    linear, and refusing to load them would break the archived runs the
    comparison table is built from.
    """
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        return spec_from_config({})
    with open(config_path) as f:
        return spec_from_config(json.load(f))


def describe(spec):
    """One-line human description, for logs and metadata."""
    if spec["head"] == HEAD_MLP:
        return f"mlp (hidden={spec['mlp_hidden']}, dropout={spec['dropout']})"
    return f"linear (dropout={spec['dropout']})"
