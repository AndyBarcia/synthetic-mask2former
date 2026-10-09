"""Render only selected queries; keep dense oracle proposals lazy."""
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from detectron2.modeling.postprocessing import sem_seg_postprocess


def render_selected_masks(embeddings, features, orders, padded_size):
    """One batched projection and resize, padded to the longest selected order."""
    indices = pad_sequence(orders, batch_first=True, padding_value=0)
    if indices.shape[1] == 0:
        return features.new_empty((features.shape[0], 0, *padded_size))
    selected = embeddings.gather(1, indices[:, :, None].expand(-1, -1, embeddings.shape[-1]))
    masks = torch.einsum('bqc,bchw->bqhw', selected, features)
    return F.interpolate(masks, size=padded_size, mode='bilinear', align_corners=False)


class DeferredMaskProposals:
    """All-query masks are materialized only when an oracle evaluator needs them."""
    def __init__(self, embeddings, features, padded_size, image_size, output_size, postprocess):
        self.embeddings = embeddings
        self.features = features
        self.padded_size = padded_size
        self.image_size = image_size
        self.output_size = output_size
        self.postprocess = postprocess

    @torch.no_grad()
    def materialize(self):
        masks = torch.einsum('qc,chw->qhw', self.embeddings, self.features)
        masks = F.interpolate(masks[None], size=self.padded_size,
                              mode='bilinear', align_corners=False)[0]
        if self.postprocess:
            masks = sem_seg_postprocess(masks, self.image_size, *self.output_size)
        return masks

    def __gt__(self, value):
        # Keep analysis tools that threshold all proposals compatible.
        return self.materialize() > value

    def __getitem__(self, indices):
        # A diagnostic asking for one proposal need not render the whole vocabulary.
        embeddings = self.embeddings[indices]
        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings[None]
        masks = torch.einsum('qc,chw->qhw', embeddings, self.features)
        if masks.shape[0]:
            masks = F.interpolate(masks[None], size=self.padded_size,
                                  mode='bilinear', align_corners=False)[0]
            if self.postprocess:
                masks = sem_seg_postprocess(masks, self.image_size, *self.output_size)
        else:
            size = self.output_size if self.postprocess else self.padded_size
            masks = self.features.new_empty((0, *size))
        return masks[0] if single else masks

    @property
    def shape(self):
        size = self.output_size if self.postprocess else self.padded_size
        return (self.embeddings.shape[0], *size)

    def detach(self):
        return self.materialize().detach()
