"""Frozen original terminal-sequence PQ reward for API parity tests."""
import torch
import torch.nn.functional as F

@torch.no_grad()
def object_reward(self, orders, outputs, targets):
    """Per-image mean class PQ with the inference paint and evaluation rules.

    A reward size of zero renders at the ground-truth mask resolution.
    """
    classes = outputs["pred_logits"][..., :-1].argmax(-1)
    rewards = []
    for batch_index, order in enumerate(orders):
        target_masks = targets[batch_index]["masks"]
        size = (self.object_rl_reward_size,) * 2 if self.object_rl_reward_size else target_masks.shape[-2:]
        masks = F.interpolate(
            outputs["pred_masks"][batch_index:batch_index + 1].float(),
            size=size, mode="bilinear", align_corners=False,
        )[0] > 0
        painted = torch.zeros(size, dtype=torch.long, device=masks.device)
        segments = []
        stuff_id = None
        paint_order = reversed(order) if self.object_rl_paint_order == "reverse" else order
        for query in paint_order:
            if not masks[query].any():
                continue
            category = int(classes[batch_index, query])
            if category == 0 and stuff_id is not None:
                segment_id = stuff_id
            else:
                segment_id = len(segments) + 1
                segments.append(category)
                if category == 0:
                    stuff_id = segment_id
            painted[masks[query]] = segment_id
        gt_masks = (
            F.interpolate(target_masks[:, None].float(), size=size, mode="nearest")[:, 0] > 0
            if len(target_masks) else masks.new_zeros((0, *size))
        )
        gt_classes = targets[batch_index]["labels"]
        gt_map = torch.zeros(size, dtype=torch.long, device=masks.device)
        for target_index, target_mask in enumerate(gt_masks, start=1):
            gt_map[target_mask] = target_index

        num_gt = len(gt_masks)
        num_pred = len(segments)
        pair_base = num_pred + 1
        intersections = torch.bincount(
            (gt_map * pair_base + painted).flatten(),
            minlength=(num_gt + 1) * pair_base,
        ).reshape(num_gt + 1, pair_base)
        pred_area = intersections.sum(0)[1:]
        gt_area = intersections.sum(1)[1:]
        void_overlap = intersections[0, 1:]
        pred_classes = gt_classes.new_tensor(segments)
        if num_gt and num_pred:
            overlap = intersections[1:, 1:].float()
            union = gt_area[:, None] + pred_area[None, :] - overlap - void_overlap[None, :]
            iou = overlap / union.clamp_min(1)
            matches = (iou > 0.5) & (gt_classes[:, None] == pred_classes[None, :])
            matched_gt, matched_pred = matches.nonzero(as_tuple=True)
            iou_sum = torch.bincount(
                gt_classes[matched_gt], weights=iou[matched_gt, matched_pred],
                minlength=self.num_classes,
            )
            matched_predictions = matches.any(0)
        else:
            iou_sum = masks.new_zeros(self.num_classes, dtype=torch.float)
            matched_predictions = torch.zeros(num_pred, dtype=torch.bool, device=masks.device)
        counted_predictions = (pred_area > 0) & (
            matched_predictions | (void_overlap.float() / pred_area.clamp_min(1) <= 0.5)
        )
        gt_count = torch.bincount(gt_classes, minlength=self.num_classes)
        pred_count = torch.bincount(
            pred_classes[counted_predictions], minlength=self.num_classes,
        )
        denominator = 0.5 * (gt_count + pred_count)
        active_classes = denominator > 0
        class_pq = iou_sum / denominator.clamp_min(1)
        rewards.append(class_pq[active_classes].mean() if active_classes.any()
                       else iou_sum.new_zeros(()))
    return torch.stack(rewards)
