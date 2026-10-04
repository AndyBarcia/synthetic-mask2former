import torch

from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher


def make_criterion():
    return SetCriterion(
        3, HungarianMatcher(mask_loss_type="block"), {}, .1,
        ["labels", "masks", "query_bias"], 8, 3, .75,
        mask_loss_type="block", fused_thing_masks=True, thing_class_ids=(1, 2),
    )


def test_unions_only_for_repeated_things_and_preserve_targets():
    criterion = make_criterion()
    labels = torch.tensor([0, 0, 1, 1, 2])
    masks = torch.eye(5).reshape(5, 1, 5)
    targets = [{"labels": labels, "masks": masks}]
    augmented = criterion._mask_targets(targets)[0]
    assert augmented["labels"].tolist() == [0, 0, 1, 1, 2, 1]
    torch.testing.assert_close(augmented["masks"][-1], masks[2:4].bool().any(0).float())
    assert targets[0]["masks"] is masks
    assert criterion._mask_targets([{"labels": labels[:0], "masks": masks[:0]}])[0]["masks"].shape == (0, 1, 5)


def test_fused_match_has_mask_gradients_only_and_is_excluded_from_objects():
    criterion = make_criterion()
    targets = [{"labels": torch.tensor([1, 1]),
                "masks": torch.tensor([[[1., 0.]], [[0., 1.]]])}]
    outputs = {
        "pred_masks": torch.tensor([[[[8., -8.]], [[-8., 8.]], [[8., 8.]]]], requires_grad=True),
        "pred_logits": torch.zeros(1, 3, 4, requires_grad=True),
        "gt_query_bias_logits": [torch.zeros(2, 3, requires_grad=True)],
    }
    augmented = criterion._mask_targets(targets)
    indices = criterion.matcher(outputs, augmented)
    assert list(zip(*[x.tolist() for x in indices[0]])) == [(0, 0), (1, 1), (2, 2)]
    assert criterion._object_indices(indices, targets)[0][0].tolist() == [0, 1]
    captured = []
    criterion.object_decoder = torch.nn.Identity()
    criterion.loss_object_decoder = lambda outputs, indices: captured.append(indices) or {}
    outputs["aux_outputs"] = [{k: v for k, v in outputs.items()}]
    losses = criterion(outputs, targets)
    assert captured[0][0][0].tolist() == [0, 1]
    assert "loss_mask_0" in losses
    sum(losses.values()).backward()
    assert outputs["pred_masks"].grad[0, 2].abs().sum() > 0
    assert outputs["pred_logits"].grad[0, 2].abs().sum() == 0


def test_union_matching_ignores_class_and_query_prior():
    criterion = make_criterion()
    targets = [{"labels": torch.tensor([1, 1]), "masks": torch.ones(2, 1, 2)}]
    augmented = criterion._mask_targets(targets)
    outputs = {"pred_masks": torch.ones(1, 3, 1, 2),
               "pred_logits": torch.tensor([[[0., 9., 0., 0.], [0., 9., 0., 0.], [9., 0., 0., 0.]]]),
               "gt_query_bias_logits": [torch.tensor([[9., 9., -9.], [9., 9., -9.]])]}
    src, tgt = criterion.matcher(outputs, augmented)[0]
    assert src[tgt == 2].item() == 2


def test_disabled_and_insufficient_queries():
    criterion = make_criterion()
    targets = [{"labels": torch.tensor([1, 1]), "masks": torch.ones(2, 1, 2)}]
    criterion.fused_thing_masks = False
    assert criterion._mask_targets(targets) is targets
    criterion.fused_thing_masks = True
    try:
        criterion({"pred_masks": torch.zeros(1, 2, 1, 2)}, targets)
    except ValueError as error:
        assert "NUM_OBJECT_QUERIES" in str(error)
    else:
        raise AssertionError("Insufficient queries must not silently drop targets")
