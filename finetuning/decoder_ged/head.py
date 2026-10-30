"""Linear decoder GED head and word-state extraction."""
from __future__ import annotations

import torch
from torch import nn


class LinearGEDHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, hidden_states):
        return self.classifier(hidden_states.float())


def decoder_last_hidden(model, input_ids, attention_mask):
    """Run the decoder backbone without materialising 256k-vocabulary logits."""
    causal_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(causal_model, "model", None)
    if backbone is None:
        raise TypeError(f"cannot locate decoder backbone inside {type(model).__name__}")
    output = backbone(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=False, return_dict=True)
    return output.last_hidden_state


def final_word_states(last_hidden, word_positions, records):
    states, labels, languages, sentence_indices, word_indices = [], [], [], [], []
    for batch_index, (positions, record) in enumerate(zip(word_positions, records)):
        for word_index, token_positions in enumerate(positions):
            if not token_positions:
                continue
            states.append(last_hidden[batch_index, token_positions[-1]])
            labels.append(int(record["labels"][word_index] == "i"))
            languages.append(record["lang"])
            sentence_indices.append(batch_index)
            word_indices.append(word_index)
    if not states:
        raise ValueError("batch contains no aligned word states")
    return (torch.stack(states), torch.tensor(labels, device=last_hidden.device),
            languages, sentence_indices, word_indices)
