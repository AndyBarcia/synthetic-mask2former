import torch
from mask2former.modeling.selected_panoptic_supervision import selected_segments, segment_predictions
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher


def test_hard_paint_merges_stuff_and_drops_occluded_and_unselected_queries():
    masks = torch.tensor([[[8., -8., -8.]], [[8., -8., -8.]],
                          [[-8., 8., -8.]], [[-8., -8., 8.]], [[8., 8., 8.]]])
    classes = torch.tensor([[0., 8., 0., 0.], [0., 8., 0., 0.],
                            [8., 0., 0., 0.], [8., 0., 0., 0.], [0., 0., 8., 0.]])
    order, groups, hard = selected_segments(masks, classes, [0, 1, 2, 3], (1, 2), (1, 3))
    assert groups == [[1], [2, 3]]
    torch.testing.assert_close(hard, torch.tensor([[[1., 0., 0.]], [[0., 1., 1.]]]))
    _, reverse_groups, _ = selected_segments(masks, classes, [0, 1], (1, 2), (1, 3), reverse=True)
    assert reverse_groups == [[0]]
    _, repeated_groups, _ = selected_segments(masks, classes, [0, 1, 0], (1, 2), (1, 3))
    assert repeated_groups == [[0]]


def test_selected_losses_train_mask_class_and_bias_but_not_unused_proposals():
    criterion = SetCriterion(3, HungarianMatcher(mask_loss_type='block'), {}, .1,
                             ['labels', 'masks', 'query_bias'], 8, 3, .75,
                             mask_loss_type='block', fused_thing_masks=True, thing_class_ids=(1, 2))
    criterion.hybrid_selected_panoptic_supervision = True
    criterion.object_rl_paint_order = 'forward'
    masks = torch.tensor([[[[8., -8., -8.]], [[-8., 8., -8.]],
                           [[-8., -8., 8.]], [[8., 8., 8.]]]], requires_grad=True)
    classes = torch.tensor([[[0., 8., 0., 0.], [8., 0., 0., 0.],
                             [8., 0., 0., 0.], [0., 0., 8., 0.]]], requires_grad=True)
    bias = torch.zeros(2, 4, requires_grad=True)
    outputs = {'pred_masks': masks, 'pred_logits': classes, 'gt_query_bias_logits': [bias]}
    aux_masks = masks.detach().clone().requires_grad_()
    aux_classes = classes.detach().clone().requires_grad_()
    outputs['aux_outputs'] = [{'pred_masks': aux_masks, 'pred_logits': aux_classes,
                              'gt_query_bias_logits': [bias]}]
    def rl(outputs, targets):
        outputs['selected_panoptic_orders'] = [[0, 1, 2]]
        return {'loss_object_rl': masks.sum() * 0}
    criterion.loss_object_rl = rl
    targets = [{'labels': torch.tensor([1, 0]),
                'masks': torch.tensor([[[1., 0., 0.]], [[0., 1., 1.]]])}]
    losses = criterion(outputs, targets)
    assert 'loss_object_decoder' not in losses
    assert 'loss_query_bias_0' in losses
    sum(losses.values()).backward()
    for value in [masks, classes, aux_masks, aux_classes]:
        assert torch.isfinite(value.grad).all()
        assert value.grad[0, :3].abs().sum() > 0
        assert value.grad[0, 3].abs().sum() == 0
    assert bias.grad[:, :3].abs().sum() > 0
    assert bias.grad[:, 3].abs().sum() == 0


def test_empty_selection_has_finite_zero_supervised_losses():
    criterion = SetCriterion(3, HungarianMatcher(mask_loss_type='block'), {}, .1,
                             ['labels', 'masks', 'query_bias'], 8, 3, .75,
                             mask_loss_type='block', thing_class_ids=(1, 2))
    criterion.hybrid_selected_panoptic_supervision = True
    criterion.object_rl_paint_order = 'forward'
    outputs = {'pred_masks': torch.randn(1, 2, 2, 2, requires_grad=True),
               'pred_logits': torch.randn(1, 2, 4, requires_grad=True),
               'gt_query_bias_logits': [torch.zeros(1, 2, requires_grad=True)]}
    def rl(outputs, targets):
        outputs['selected_panoptic_orders'] = [[]]
        return {'loss_object_rl': outputs['pred_masks'].sum() * 0}
    criterion.loss_object_rl = rl
    losses = criterion(outputs, [{'labels': torch.tensor([1]), 'masks': torch.ones(1, 2, 2)}])
    assert all(v.item() == 0 for v in losses.values())
    sum(losses.values()).backward()


def test_best_orders_reuse_eof_samples_and_include_greedy():
    class Decoder(torch.nn.Module):
        num_queries = 3
        def __init__(self):
            super().__init__()
            self.bos = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0
        def rollout(self, *args, sample, **kwargs):
            self.calls += 1
            orders = [[0], [1], [2], [0, 1]] if sample else [[0, 2], [1, 2]]
            return orders, None, [order + [3] for order in orders]
        def autoregressive_log_probability(self, *args, **kwargs):
            return self.bos.expand(len(args[4]))
    decoder = Decoder()
    criterion = SetCriterion(3, HungarianMatcher(mask_loss_type='block'), {}, .1,
                             ['labels', 'masks', 'query_bias'], 8, 3, .75,
                             object_decoder=decoder, object_rl_weight=1,
                             object_rl_num_samples=2, object_rl_max_steps=4)
    criterion.hybrid_selected_panoptic_supervision = True
    criterion._object_rollout_rewards = lambda *args: (torch.tensor([[.2, -.1], [.4, -.2]]), torch.zeros(2))
    outputs = {'mask_embeddings': torch.randn(2, 3, 4),
               'object_decoder_image_features': [torch.randn(2, 4, 4)],
               'pred_masks': torch.randn(2, 3, 2, 2), 'object_decoder_image_sizes': [(2, 2)]}
    criterion.loss_object_rl_eof(outputs, [])
    assert outputs['selected_panoptic_orders'] == [[2], [1, 2]]
    assert decoder.calls == 2
