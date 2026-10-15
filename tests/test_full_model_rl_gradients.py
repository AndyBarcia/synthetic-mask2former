import torch
from torch import nn

from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.transformer_decoder.gt_query_bias import GTQueryBias


def test_full_model_rl_preserves_policy_input_gradients():
    criterion = SetCriterion.__new__(SetCriterion)
    nn.Module.__init__(criterion)
    upstream = torch.randn(3, requires_grad=True)
    assert not criterion._rl_input(upstream).requires_grad
    criterion.object_rl_detach_inputs = False
    criterion._rl_input(upstream).sum().backward()
    torch.testing.assert_close(upstream.grad, torch.ones_like(upstream))


def test_full_model_query_bias_trains_masks_and_image_features():
    torch.manual_seed(10)
    head = GTQueryBias(4, 8, 3)
    nn.init.normal_(head.query_logits.weight)
    head.detach_predicted_masks = False
    features = torch.randn(1, 4, 4, 4, requires_grad=True)
    masks = torch.randn(1, 3, 8, 8, requires_grad=True)
    head.predicted_query_logits(features, masks).sum().backward()
    for value in (features, masks):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_hybrid_excludes_object_teacher_forcing():
    from unittest.mock import Mock

    criterion = SetCriterion.__new__(SetCriterion)
    nn.Module.__init__(criterion)
    criterion.fused_thing_masks = False
    criterion.losses = []
    criterion.object_decoder = nn.Identity()
    criterion.object_rl_weight = 1.0
    criterion.matcher = Mock(return_value=[])
    criterion._mask_targets = Mock(side_effect=lambda targets, **kwargs: targets)
    criterion._object_indices = Mock(return_value=[])
    criterion.loss_object_decoder = Mock(return_value={'loss_object_decoder': torch.tensor(2.)})
    criterion.loss_object_rl = Mock(return_value={'loss_object_rl': torch.tensor(1.)})
    outputs = {'pred_masks': torch.zeros(1, 1, 2, 2)}
    targets = [{'labels': torch.zeros(0, dtype=torch.long)}]
    assert set(criterion(outputs, targets)) == {'loss_object_rl'}
    criterion.loss_object_decoder.assert_not_called()
    criterion.object_rl_weight = 0.0
    assert set(criterion(outputs, targets)) == {'loss_object_decoder'}
    criterion.loss_object_decoder.assert_called_once()
