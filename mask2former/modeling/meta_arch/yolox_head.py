"""Semantic head for dense mask proposals, parallel to MaskFormerHead."""
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..pixel_decoder.fpn import build_pixel_decoder
from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..yolox.proposals import YOLOXMaskProposals


class ImageMemories(nn.Module):
    def __init__(self, channels, hidden_dim, enforce_projection=False):
        super().__init__()
        self.input_proj = nn.ModuleList([
            nn.Conv2d(channels, hidden_dim, 1) if channels != hidden_dim or enforce_projection else nn.Sequential()
            for _ in range(3)
        ])
        self.level_embed = nn.Embedding(3, hidden_dim)
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

    def forward(self, multi):
        return [(self.input_proj[i](x).flatten(2) + self.level_embed.weight[i][None, :, None]
                 + self.pe_layer(x, None).flatten(2)).transpose(1, 2) for i, x in enumerate(multi)]


class ProposalTokenAdapter(nn.Sequential):
    def __init__(self, num_classes, mask_dim, fusion='residual'):
        if fusion not in ('residual', 'concat'):
            raise ValueError('Unknown proposal token fusion')
        context_dim = 175 + num_classes
        layers = ([nn.LayerNorm(context_dim), nn.Linear(context_dim, mask_dim)]
                  if fusion == 'residual' else [])
        super().__init__(*layers)
        if fusion == 'residual':
            nn.init.zeros_(self[-1].weight)
            nn.init.zeros_(self[-1].bias)
        self.num_classes = num_classes
        self.fusion = fusion
        self.output_dim = mask_dim + context_dim if fusion == 'concat' else mask_dim

    def forward(self, proposal, padded_size):
        scale = proposal['centers'].new_tensor((padded_size[1], padded_size[0]))
        context = torch.cat((proposal['dynamic_parameters'], proposal['centers'] / scale,
                             proposal['strides'] / scale, proposal['bias'], proposal['scores'][:, None],
                             F.one_hot(proposal['labels'], self.num_classes).float()), -1)
        if self.fusion == 'concat':
            return torch.cat((proposal['embeddings'], context), dim=-1)
        return proposal['embeddings'] + super().forward(context)


@SEM_SEG_HEADS_REGISTRY.register()
class YOLOXHead(nn.Module):
    """Pixel decoder + dense predictor, returning the standard prediction dictionary."""

    @configurable
    def __init__(self, *, pixel_decoder, predictor, memory, token_adapter, num_classes, num_queries):
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = predictor
        self.memory = memory
        self.token_adapter = token_adapter
        self.num_classes = num_classes
        self.num_queries = num_queries

    @classmethod
    def from_config(cls, cfg, input_shape):
        y, m = cfg.MODEL.YOLOX, cfg.MODEL.MASK_FORMER
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        if channels != cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM:
            raise ValueError('Pyramid and mask feature dimensions must match')
        if not y.DYNAMIC_MASKS:
            raise ValueError('YOLOXHead requires dynamic-mask tokens')
        things = metadata.thing_dataset_id_to_contiguous_id.values()
        if not 0 < len(things) < classes:
            raise ValueError('YOLOXHead requires thing and stuff classes')
        return dict(
            pixel_decoder=build_pixel_decoder(cfg, input_shape),
            predictor=YOLOXMaskProposals(channels, classes, things, y.WIDTH, dynamic_masks=True,
                                        box_free=y.BOX_FREE, soft_labels=y.SOFT_LABELS, full_mask_loss=y.FULL_MASK_LOSS,
                                        large_kernel=y.LARGE_KERNEL),
            memory=ImageMemories(channels, m.HIDDEN_DIM, m.ENFORCE_INPUT_PROJ),
            token_adapter=ProposalTokenAdapter(classes, channels, fusion=y.TOKEN_FUSION), num_classes=classes, num_queries=y.TOPK)

    def forward(self, features, mask=None, *, padded_size):
        return self.layers(features, mask, padded_size=padded_size)

    def layers(self, features, mask=None, *, padded_size):
        dense, _, multi = self.pixel_decoder.forward_features(features)
        raw = self.predictor(multi, dense, padded_size)
        proposals = [self.predictor.proposals(raw, dense, index=i, topk=self.num_queries) for i in range(len(dense))]
        if any(len(p['masks']) != self.num_queries for p in proposals):
            raise ValueError('Proposal budget exceeds the candidates available at this resolution')
        return {
            'pred_masks': torch.stack([p['masks'] for p in proposals]),
            'mask_embeddings': torch.stack([self.token_adapter(p, padded_size) for p in proposals]),
            'object_decoder_image_features': self.memory(multi),
            'object_decoder_image_sizes': [tuple(x.shape[-2:]) for x in multi],
            'dense_predictions': raw,
            'mask_features': dense,
            'proposals': proposals,
            'padded_size': padded_size,
        }
