import unittest
import torch
from mask2former.modeling.meta_arch.yolox_head import ProposalTokenAdapter
from mask2former.modeling.yolox.object_decoder import ProposalObjectDecoder
from tests.test_native_yolox import model, batch


class ConcatTokensTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_identical_embeddings_different_kernels_produce_distinct_tokens(self):
        adapter = ProposalTokenAdapter(4, 256, fusion='concat')
        embedding = torch.randn(1, 256).expand(2, -1).clone().requires_grad_()
        kernels = torch.zeros(2, 169, requires_grad=True)
        with torch.no_grad():
            kernels[1, 10] = 1.
        proposal = dict(embeddings=embedding, dynamic_parameters=kernels,
                        centers=torch.zeros(2, 2), strides=torch.ones(2, 2),
                        bias=torch.zeros(2, 1), scores=torch.ones(2), labels=torch.zeros(2, dtype=torch.long))
        tokens = adapter(proposal, (128, 128))
        self.assertEqual(adapter.output_dim, 435)
        torch.testing.assert_close(tokens[:, :256], embedding)
        torch.testing.assert_close(tokens[:, 256:425], kernels)
        self.assertFalse(torch.equal(tokens[0], tokens[1]))
        decoder = ProposalObjectDecoder(435, 32, 2, 1, 4, 64)
        (decoder.input_mask_projection(tokens).square().sum() +
         decoder.output_mask_projection(tokens).square().sum()).backward()
        self.assertGreater(float(embedding.grad.abs().sum()), 0.)
        self.assertTrue(bool((kernels.grad.abs().sum(1) > 0).all()))

    def make_model(self, rl=False):
        m = model(rl=rl)
        adapter = ProposalTokenAdapter(4, 256, fusion='concat')
        m.sem_seg_head.token_adapter = adapter
        m.criterion.object_decoder = ProposalObjectDecoder(adapter.output_dim, 32, 86, 1, 4, 64, prefix_tree_branches=2)
        m.num_queries = m.sem_seg_head.num_queries = 86
        m.criterion.object_dec_supervision = 'dynamic_k_grouped'
        m.criterion.matcher.ensure_gt_coverage = True
        return m.train()

    def test_object_loss_reaches_dynamic_kernels_immediately(self):
        m = self.make_model()
        m(batch())['loss_object_decoder'].backward()
        for module in [m.sem_seg_head.predictor.kernel, m.sem_seg_head.predictor.dynamic]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_empty_scene_oracle_and_decoder_only_rl(self):
        m = self.make_model()
        sum(m(batch(empty=True)).values()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        m.eval()
        inputs = batch()
        inputs[0]['panoptic_ground_truth'] = (torch.zeros(71, 79), [])
        with torch.no_grad():
            result = m(inputs)[0]
        self.assertEqual(tuple(result['panoptic_proposals'].shape), (86, 71, 79))
        m = self.make_model(rl=True)
        m(batch())['loss_object_rl'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))


if __name__ == '__main__':
    unittest.main()
