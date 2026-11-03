"""Object-decoder policy over dynamic mask proposals, without overlap exclusions."""
import torch

from ..object_decoder import ObjectDecoder


class ProposalObjectDecoder(ObjectDecoder):
    """Exclude only previously emitted IDs; leave overlap decisions to the policy."""

    def prepare_vocabulary_exclusions(self, mask_logits):
        batch, queries = mask_logits.shape[:2]
        return torch.eye(queries, device=mask_logits.device, dtype=torch.bool)[None].expand(batch, -1, -1)
