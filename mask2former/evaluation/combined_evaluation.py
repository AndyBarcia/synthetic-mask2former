"""Adapters for combined segmentation metrics and distributed AP accumulation."""
import torch


def instance_batch(outputs, batch_type, device, class_mapping=None):
    masks, classes, scores = [], [], []
    for output in outputs:
        instances = output["instances"].to(device)
        labels = instances.pred_classes.long()
        if class_mapping is not None:
            labels = torch.tensor([class_mapping[int(label)] for label in labels.cpu()],
                                  dtype=torch.int64, device=device)
        binary = instances.pred_masks.bool()
        keep = binary.flatten(1).any(1)
        masks.append(binary[keep])
        classes.append(labels[keep])
        scores.append(instances.scores[keep])
    return batch_type(masks, classes, scores=scores)


def instance_state(evaluator):
    # The backend retains only scored match records, never image masks. Keep
    # its internal-state dependency here until a public merge API is available.
    return ([type(record)(*(field.cpu() for field in record))
             for record in evaluator._records], evaluator._counts.cpu())


def summarize_instances(states, num_classes, device="cpu"):
    from panoptic_evaluator import InstanceEvaluator
    evaluator = InstanceEvaluator(num_classes, device=device)
    for records, counts in states:
        evaluator._commit([type(record)(*(field.to(device) for field in record))
                           for record in records], counts.to(device))
    values = evaluator.compute()
    names = {"ap": "AP", "ap50": "AP50", "ap75": "AP75",
             "ap_small": "APs", "ap_medium": "APm", "ap_large": "APl",
             "ar1": "AR1", "ar10": "AR10", "ar100": "AR100",
             "ar_small": "ARs", "ar_medium": "ARm", "ar_large": "ARl"}
    return {name: 100 * values[key].item() if values[key] >= 0 else float("nan")
            for key, name in names.items()}
