"""Query-priority constraints shared by object attention and token selection."""

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


module_path = Path(__file__).resolve().parents[1] / "mask2former/modeling/object_decoder.py"
spec = importlib.util.spec_from_file_location("object_decoder_under_test", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ObjectDecoder = module.ObjectDecoder


class RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, query, key, value, attn_mask, need_weights):
        self.mask = attn_mask
        return query.new_zeros(query.shape), None


def inputs():
    embeddings = torch.zeros(1, 3, 4)
    features = [torch.zeros(1, 4, 4) for _ in range(3)]
    masks = torch.full((1, 3, 2, 2), -1.0)
    return embeddings, features, masks, [(2, 2)] * 3


def test_query_bias_masks_attention_and_token_logits_and_detaches_source():
    decoder = ObjectDecoder(4, 4, 3, 1, 1, 8, bias_order_constraint=True)
    for parameter in decoder.parameters():
        nn.init.zeros_(parameter)
    decoder.query_bias_scale.data.fill_(1)
    attention = RecordingAttention()
    decoder.vocabulary_attention[0] = attention
    embeddings, features, masks, sizes = inputs()
    bias = torch.tensor([[0.0, 2.0, 1.0]], requires_grad=True)
    exclusions = decoder.prepare_vocabulary_exclusions(masks, bias)
    assert exclusions[0].tolist() == [
        [True, True, True], [False, True, False], [False, True, True],
    ]

    regions = decoder.prepare_regions(masks, sizes)
    logits = decoder(embeddings, features, regions, torch.tensor([[-1, 2]]),
                     exclusions, bias)
    torch.testing.assert_close(logits[0, 0], torch.tensor([0., 2., 1., 0.]))
    assert torch.isneginf(logits[0, 1, 1:3]).all()
    torch.testing.assert_close(logits[0, 1, [0, 3]], torch.zeros(2))
    torch.testing.assert_close(attention.mask[0, 0], logits[0, 0])
    assert torch.isneginf(attention.mask[0, 1, 1:3]).all()

    F.cross_entropy(logits[:, 0], torch.tensor([1])).backward()
    assert bias.grad is None
    assert decoder.query_bias_scale.grad is not None
    assert decoder.query_bias_offset.grad is not None


def test_generation_never_returns_to_higher_query_bias():
    decoder = ObjectDecoder(4, 4, 3, 1, 1, 8, bias_order_constraint=True)
    for parameter in decoder.parameters():
        nn.init.zeros_(parameter)
    embeddings, features, masks, sizes = inputs()
    bias = torch.tensor([[0., 2., 1.]])
    # Equal token scores choose query zero first; higher-bias queries then
    # become unavailable to both vocabulary attention and token selection.
    generated = decoder.generate(embeddings, features, masks, sizes, bias)
    assert generated[0][0].tolist() == [0]


def test_sampled_rollout_and_rescoring_share_query_bias_constraints():
    torch.manual_seed(7)
    decoder = ObjectDecoder(4, 4, 3, 1, 1, 8, bias_order_constraint=True)
    embeddings, features, masks, sizes = inputs()
    bias = torch.tensor([[0., 2., 1.]])
    with torch.no_grad():
        orders, sampled_log_probability, actions = decoder.rollout(
            embeddings, features, masks, sizes, sample=True, max_steps=4,
            return_actions=True, query_bias_logits=bias,
        )
    ranks = {1: 0, 2: 1, 0: 2}
    assert [ranks[query] for query in orders[0]] == sorted(
        ranks[query] for query in orders[0]
    )
    rescored = decoder.autoregressive_log_probability(
        embeddings, features, masks, sizes, actions, query_bias_logits=bias,
    )
    torch.testing.assert_close(rescored, sampled_log_probability)
    invalid = decoder.autoregressive_log_probability(
        embeddings, features, masks, sizes, [[0, 1, 3]],
        query_bias_logits=bias,
    )
    assert torch.isneginf(invalid).all()
