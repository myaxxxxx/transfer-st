#!/usr/bin/env python3
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    LayerNorm,
)
from fairseq.utils import get_activation_fn

logger = logging.getLogger(__name__)


def get_adapter_keys(lang_pairs, lang):
    if lang == "src":
        adapter_keys = [p.split("-")[0] for p in lang_pairs.split(",")]
    elif lang == "tgt":
        adapter_keys = [p.split("-")[1] for p in lang_pairs.split(",")]
    elif lang == "pair":
        adapter_keys = lang_pairs.split(",")
    else:
        raise ValueError
    # ensure consistent order!
    adapter_keys = sorted(list(set(adapter_keys)))
    return adapter_keys

# We want to use the same init as in transformer.py, but to avoid circular imports
# we temporarily copy the Linear() function here
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class Adapter(nn.Module):
    def __init__(
            self,
            input_size: int,
            bottleneck_size: int,
            activation_fn: str,
            static_layernorm: bool,
    ):
        """
        Implements an Adapter layer following the architecture of
        Bapna and Firat 2019 - Simple, Scalable Adaptation for Neural Machine Translation
        https://aclanthology.org/D19-1165/

        Args:
            input_size (int): the dimensionality of the input feature vector
            bottleneck_size (int): the dimensionality of the bottleneck vector
            activation_fn (str): the activation function used after the down-projection
            static_layernorm (bool): use LayerNorm without trainable parameters
        """
        super().__init__()

        # reuse the transformer Linear layer to have consistent init with the rest of the model
        self.down = Linear(input_size, bottleneck_size)
        self.up = Linear(bottleneck_size, input_size)
        self.layer_norm = LayerNorm(input_size,
                                    elementwise_affine=not static_layernorm)
        self.activation_fn = get_activation_fn(activation_fn)

        for n, p in self.named_parameters():
            p.adapter = True
            p.label = n

    def forward(self, x: torch.Tensor):
        shortcut = x

        x = self.layer_norm(x)
        x = self.down(x)
        x = self.activation_fn(x)
        x = self.up(x)

        return x + shortcut