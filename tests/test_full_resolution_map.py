"""Native mask-prediction features reach map encoding in training and inference."""
from unittest.mock import patch
from detectron2.config import get_cfg
from mask2former.config import add_maskformer2_config
import torch
from mask2former.modeling.object_decoder import ObjectDecoder
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


def check(device='cpu', amp=False, mask_dim=16):
    torch.manual_seed(67)
    head=MultiScaleMaskedTransformerDecoder(16,num_classes=3,hidden_dim=16,num_queries=5,
        nheads=4,dim_feedforward=32,dec_layers=2,pre_norm=False,mask_dim=mask_dim,
        enforce_input_project=False).to(device)
    decoder=ObjectDecoder(mask_dim,16,5,2,4,32,map_feature_dim=mask_dim).to(device)
    x=[torch.randn(2,16,h,h,device=device,requires_grad=True) for h in (2,4,8)]
    pixels=torch.randn(2,mask_dim,16,16,device=device,requires_grad=True)
    with torch.autocast(device_type=device,enabled=amp):
        out=head(x,pixels)
        features=out['object_decoder_image_features'];sizes=out['object_decoder_image_sizes']
        assert sizes==[(2,2),(4,4),(8,8),(16,16)]
        torch.testing.assert_close(features[-1],pixels.flatten(2).transpose(1,2))
        regions=decoder.prepare_regions(out['pred_masks'],sizes)
        previous=torch.tensor([[-1,0,1],[-1,1,0]],device=device)
        exclusions=decoder.prepare_vocabulary_exclusions(out['pred_masks'])
        cache=decoder._prepare_decode_cache(out['mask_embeddings'],features,regions,exclusions,3)
        assert cache['owner'].shape==(2,256) and len(cache['image_kv'])==2
        assert len(cache['covered'])==3
        assert cache['image_kv'][0][0].shape[-2]==5  # 2x2 plus null, not 16x16.
        with patch.object(decoder.map_encoder,'encode_maps',wraps=decoder.map_encoder.encode_maps) as encode:
            logits=decoder(out['mask_embeddings'],features,regions,previous,exclusions)
            assert encode.call_args.args[1].shape==(2,256,16)
            assert encode.call_args.args[2].shape==(2,5,256)
        loss=logits.float().square().mean()
    loss.backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum()>0
    # Native features affect the summary independently of the attention memories.
    decoder.eval()
    with torch.no_grad(),torch.autocast(device_type=device,enabled=amp):
        detached=[a.detach() for a in features]
        a=decoder(out['mask_embeddings'].detach(),detached,regions,previous,exclusions)
        b=decoder(out['mask_embeddings'].detach(),detached[:3]+[detached[-1]+2],regions,previous,exclusions)
        assert not torch.allclose(a,b)
        expected=decoder(out['mask_embeddings'].detach(),detached,regions,previous,exclusions)
        cache=decoder._prepare_decode_cache(out['mask_embeddings'].detach(),detached,regions,exclusions,3)
        actual=torch.stack([decoder._decode_step(token,cache) for token in previous.unbind(1)],1)
        torch.testing.assert_close(actual,expected,atol=.005 if amp else 5e-6,rtol=.01 if amp else 5e-6)
        head.eval()
        efficient=head([a.detach() for a in x],pixels.detach(),defer_mask_predictions=True)
        assert len(efficient['object_decoder_image_regions'])==4
        full=torch.einsum('bqc,bchw->bqhw',efficient['mask_embeddings'],pixels.detach())
        assert torch.equal(efficient['object_decoder_image_regions'][-1],full.flatten(2)>0)
        decoder.generate(efficient['mask_embeddings'],efficient['object_decoder_image_features'],
            efficient['pred_masks'],efficient['object_decoder_image_sizes'],
            image_regions=efficient['object_decoder_image_regions'])
        # The combined rollout and likelihood scorer also accept the native level.
        sampled,greedy,actions,_=decoder.rollout_samples_and_greedy(out['mask_embeddings'].detach(),
            detached,out['pred_masks'].detach(),sizes,2,max_steps=6)
        decoder.trajectory_log_probability(out['mask_embeddings'].detach().repeat(2,1,1),
            [a.repeat(2,1,1) for a in detached],out['pred_masks'].detach().repeat(2,1,1,1),sizes,actions)
    print('native map features, gradients, cached decode, deferred inference and rollouts PASS',device,amp)


def check_large_pool(device='cpu'):
    from mask2former.modeling.map_encoder import MapEncoder
    features=torch.full((1,65536,8),2.,device=device,dtype=torch.float16)
    visible=torch.ones(1,1,1,65536,device=device,dtype=torch.bool)
    for incremental in [False,True]:
        cache={"float32_accumulation":True}
        with torch.autocast(device_type=device,enabled=device=='cuda'):
            pooled=(MapEncoder._pool_inference_segments(visible,features,cache,1) if incremental
                    else MapEncoder._pool_visible_segments(visible,features,cache))
        torch.testing.assert_close(pooled,torch.full_like(pooled,2))
    print('65536-pixel FP16 regions pool without overflow PASS',device)


if __name__=='__main__':
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    assert cfg.MODEL.MASK_FORMER.MAP_ENCODER_FULL_RESOLUTION is True
    torch.set_num_threads(2)
    check()
    check(mask_dim=8)
    check_large_pool()
    if torch.cuda.is_available():
        check_large_pool('cuda')
        check('cuda')
        check('cuda',True)
        check('cuda',True,mask_dim=8)
