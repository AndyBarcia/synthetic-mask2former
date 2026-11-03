"""Decoder-only RLOO PQ optimization for frozen dynamic-mask proposals."""
import torch
from torch.nn import functional as F


def rloo_loss(rewards,greedy,log_probability):
    assert rewards.shape==log_probability.shape and rewards.ndim==2
    k=rewards.shape[0]
    baseline=(greedy[None]+rewards.sum(0,keepdim=True)-rewards)/k
    advantage=(rewards-baseline).detach()
    return -(advantage*log_probability).mean(),advantage


def trajectory_log_probability(decoder,embeddings,memories,masks,sizes,trajectories):
    """Replay exactly sampled actions, including real EOF and excluding forced EOF.

    A capped sequence may have no EOF. Its likelihood contains only the actions
    actually sampled, unlike the legacy EOF-only trajectory replay helper.
    """
    lengths=torch.tensor([len(x) for x in trajectories],device=embeddings.device)
    assert bool((lengths>0).all())
    steps=int(lengths.max());q=embeddings.shape[1]
    actions=torch.full((len(trajectories),steps),q,device=embeddings.device,dtype=torch.long)
    for i,trace in enumerate(trajectories):actions[i,:len(trace)]=torch.as_tensor(trace,device=embeddings.device)
    previous=F.pad(actions[:,:-1],(1,0),value=-1)
    exclusions=decoder.prepare_vocabulary_exclusions(masks)
    regions=decoder.prepare_regions(masks,sizes)
    logits=decoder(embeddings,memories,regions,previous,exclusions)
    blocked=decoder._vocabulary_mask(exclusions,previous)
    logp=logits.float().masked_fill(blocked,-torch.inf).log_softmax(-1).gather(-1,actions[...,None])[...,0]
    return torch.where(torch.arange(steps,device=embeddings.device)[None]<lengths[:,None],logp,0.).sum(-1)


@torch.no_grad()
def prepare_pq_reward(proposals,targets,size):
    from panoptic_evaluator import PanopticBatch
    prepared=[]
    for p,target in zip(proposals,targets):
        masks=F.interpolate(p['masks'][None].float(),size=size,mode='bilinear',align_corners=False)[0]
        gt=target['masks'].bool()
        if tuple(gt.shape[-2:])!=tuple(size):gt=F.interpolate(gt[:,None].float(),size=size,mode='nearest')[:,0]>.5
        identifiers=torch.arange(1,len(gt)+1,device=masks.device,dtype=torch.int32)
        canvas=(gt*identifiers[:,None,None]).amax(0) if len(gt) else torch.zeros(size,device=masks.device,dtype=torch.int32)
        labels=torch.cat((target['labels'].new_full((1,),-1),target['labels'])).to(torch.int32)
        prepared.append((masks,p['labels'],p['scores'],PanopticBatch(canvas[None],labels[None])))
    return prepared


@torch.no_grad()
def score_orders(orders,prepared,isthing,num_classes,rendering="forward"):
    from panoptic_evaluator import PanopticBatch,panoptic_quality
    from ..panoptic_paint import score_panoptic_masks, paint_panoptic_masks
    if rendering not in ('forward', 'pixel-score'):
        raise ValueError(rendering)
    values=[]
    for index,order in enumerate(orders):
        masks,labels,scores,target=prepared[index%len(prepared)]
        ids=torch.as_tensor(order,device=masks.device,dtype=torch.long)
        valid=torch.ones((1,len(ids)),device=masks.device,dtype=torch.bool)
        if rendering=="forward":
            canvas,table=paint_panoptic_masks(masks[ids][None],labels[ids][None],valid,isthing,reverse=False)
        else:
            canvas,table=score_panoptic_masks(masks[ids][None],labels[ids][None],valid,isthing,scores[ids][None],overlap_threshold=0.)
        value=panoptic_quality(PanopticBatch(canvas,table),target,num_classes,validate=False).float()
        values.append(value)
    return torch.cat(values)
