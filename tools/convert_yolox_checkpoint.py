"""Convert research head/joint checkpoints to a native Detectron2 model checkpoint.

Optimizer state is intentionally not transferred: this is weight initialization,
not an exact training resume. Native Detectron2 checkpoints support normal resume.
"""
import argparse
from pathlib import Path
import sys

import torch

# Permit invocation as `python tools/convert_yolox_checkpoint.py` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config


def convert(cfg, base, joint=None, head=None):
    cfg = cfg.clone()
    cfg.defrost()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = ''
    model = build_model(cfg)
    state = model.state_dict()
    segmentation = base['model']
    source_policy = joint['policy'] if joint else None
    source_head = joint['head'] if joint else head['head']
    if joint and joint['arguments']['budget'] != cfg.MODEL.YOLOX.TOPK:
        raise ValueError('Proposal budget does not match the source joint checkpoint')
    missing = []
    for name, value in state.items():
        source = None
        if name.startswith('backbone.'):
            source = segmentation.get(name)
        elif name.startswith('sem_seg_head.pixel_decoder.'):
            key = name[len('sem_seg_head.pixel_decoder.'):]
            source = (joint['pixel_decoder'].get(key) if joint and 'pixel_decoder' in joint
                      else segmentation.get(name))
        elif name.startswith('sem_seg_head.proposal_head.'):
            source = source_head.get(name[len('sem_seg_head.proposal_head.'):])
        elif name.startswith('sem_seg_head.memory.'):
            source = segmentation.get(name.replace('sem_seg_head.memory.', 'sem_seg_head.predictor.'))
        elif source_policy is not None:
            source = source_policy.get(name.removeprefix('policy.'))
        elif name.startswith('policy.decoder.'):
            source = segmentation.get(name.replace('policy.decoder.', 'criterion.object_decoder.'))
            if name == 'policy.decoder.position.weight' and source is not None and source.shape != value.shape:
                if cfg.MODEL.MASK_FORMER.OBJECT_DEC_POSITION_ENCODING:
                    raise ValueError('Cannot resize an active sequence-position table implicitly')
                grown = value.clone()
                count = min(len(source), len(grown))
                grown[:count] = source[:count]
                source = grown
        elif name.startswith('policy.adapter.'):
            # Match the zero-residual adapter initialization used by the pilots.
            source = value
        if source is None:
            missing.append(name)
        else:
            source = torch.as_tensor(source)
            if source.shape != value.shape:
                raise ValueError(f'{name}: source {tuple(source.shape)}, expected {tuple(value.shape)}')
            state[name] = source.to(value)
    if missing:
        raise ValueError('Missing initialization tensors: ' + ', '.join(missing))
    model.load_state_dict(state, strict=True)
    return {'model': state, 'yolox_conversion': {
        'source_step': joint.get('step') if joint else head.get('step'),
        'optimizer_transferred': False, 'pixel_decoder_from_joint': bool(joint and 'pixel_decoder' in joint)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--base-checkpoint', required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--joint-checkpoint')
    sources.add_argument('--head-checkpoint')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    load = lambda p: torch.load(p, map_location='cpu', weights_only=False) if p else None
    result = convert(cfg, load(args.base_checkpoint), load(args.joint_checkpoint), load(args.head_checkpoint))
    result['yolox_conversion']['base_checkpoint'] = args.base_checkpoint
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f'Saved {len(result["model"])} model tensors to {output}; optimizer/schedule start fresh')


if __name__ == '__main__':
    main()
