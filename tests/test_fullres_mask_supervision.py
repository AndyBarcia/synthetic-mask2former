import unittest
from unittest.mock import patch
import torch
from tests.test_native_yolox import model, batch
from mask2former.modeling.yolox.criterion import mask_losses


class FullResolutionMaskTest(unittest.TestCase):
    def test_single_pixel_object_survives_mask_supervision_and_backward(self):
        torch.set_num_threads(1)
        m = model().train()
        inputs = batch()
        inst = inputs[0]['instances']
        inst = inst[:1]
        inst.gt_masks.zero_()
        inst.gt_masks[0, 9, 9] = True  # Erased by the original stride-4 nearest target.
        inputs[0]['instances'] = inst
        seen = []

        def capture(logits, targets):
            seen.append(targets.detach().clone())
            return mask_losses(logits, targets)

        with patch('mask2former.modeling.yolox.criterion.mask_losses', side_effect=capture):
            losses = m(inputs)
        self.assertEqual(seen[0].shape[1], 64 * 64)
        self.assertTrue(bool((seen[0].sum(1) == 1).all()))
        self.assertGreater(len(seen[0]), 0)
        sum(losses.values()).backward()
        grad = m.sem_seg_head.predictor.kernel[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(bool(torch.isfinite(grad).all()))


if __name__ == '__main__':
    unittest.main()
