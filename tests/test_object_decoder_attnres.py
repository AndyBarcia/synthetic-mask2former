"""Causality and packed-prefix consistency for experimental decoder variants."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "object_decoder", Path(__file__).resolve().parents[1] / "mask2former/modeling/object_decoder.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ObjectDecoderAttentionResidualsTest(unittest.TestCase):
    variants = ("baseline", "attnres")

    def make_decoder(self, variant):
        return module.ObjectDecoder(8, 8, 5, 2, 2, 16, variant=variant).eval()


    def test_depth_attention_starts_as_mean(self):
        attention = module.DepthAttention(8)
        history = [torch.randn(2, 3, 8) for _ in range(4)]
        torch.testing.assert_close(attention(history), torch.stack(history).mean(0))


    def test_tree_forward_matches_independent_paths_and_has_finite_gradients(self):
        paths = [[0, 1, 2], [0, 3, 4]]
        embeddings = torch.randn(1, 5, 8)
        features = [torch.randn(1, 4, 8) for _ in range(3)]
        regions = [torch.zeros(1, 5, 4, dtype=torch.bool) for _ in range(3)]
        exclusions = torch.eye(5, dtype=torch.bool)[None]
        previous, positions, ancestors, weights = module.ObjectDecoder.pack_prefix_trees(
            [paths], 5, torch.device("cpu")
        )
        for variant in self.variants:
            with self.subTest(variant=variant):
                decoder = self.make_decoder(variant)
                tree_logits = decoder(embeddings, features, regions, previous, exclusions,
                                      positions=positions, ancestors=ancestors)
                self.assertTrue(torch.isfinite(tree_logits).all())
                for path in paths:
                    tokens = torch.tensor([[-1] + path])
                    path_logits = decoder(embeddings, features, regions, tokens, exclusions)
                    for depth in range(len(path) + 1):
                        # Identify this prefix in the packed trie by its tokens.
                        for node in range(previous.shape[1]):
                            branch = previous[0, ancestors[0, node]].tolist()
                            if branch == tokens[0, :depth + 1].tolist():
                                torch.testing.assert_close(tree_logits[:, node], path_logits[:, depth],
                                                           atol=2e-6, rtol=2e-5)
                                break
                        else:
                            self.fail("Prefix missing from trie")
                loss = -(tree_logits.log_softmax(-1) * weights).sum() / weights.sum()
                loss.backward()
                for name, parameter in decoder.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if decoder.use_attnres:
                    self.assertIsNotNone(decoder.depth_attention[-1].query.grad)


    def test_future_tokens_do_not_change_prefix_logits(self):
        embeddings = torch.randn(1, 5, 8)
        features = [torch.randn(1, 4, 8) for _ in range(3)]
        regions = [torch.zeros(1, 5, 4, dtype=torch.bool) for _ in range(3)]
        exclusions = torch.eye(5, dtype=torch.bool)[None]
        for variant in self.variants:
            decoder = self.make_decoder(variant)
            first = decoder(embeddings, features, regions, torch.tensor([[-1, 0, 1, 2]]), exclusions)
            changed = decoder(embeddings, features, regions, torch.tensor([[-1, 0, 3, 4]]), exclusions)
            torch.testing.assert_close(first[:, :2], changed[:, :2], atol=1e-6, rtol=1e-5)


    def test_initialization_preserves_common_weights_and_rng(self):
        torch.manual_seed(2026)
        baseline = self.make_decoder("baseline").state_dict()
        rng = torch.get_rng_state()
        torch.manual_seed(2026)
        decoder = self.make_decoder("attnres")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in baseline.items():
            torch.testing.assert_close(value, decoder.state_dict()[name])

if __name__ == "__main__":
    unittest.main()
