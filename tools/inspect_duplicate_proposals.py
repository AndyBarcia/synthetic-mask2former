"""Inspect paired whole-object proposals using the run's pinned matching code."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--step',type=int,default=3000);p.add_argument('--images',type=int,default=64);a=p.parse_args()
run=Path(a.run).resolve();sys.path.insert(0,str(run/'source'))
from train_net import get_cfg,add_deeplab_config,add_maskformer2_config,Trainer,DetectionCheckpointer
from detectron2.structures import ImageList
from mask2former.data.synthetic_scene_batch_loader import SyntheticSceneBatchLoader
from mask2former.modeling.multiple_proposals import repeat_object_targets
assert a.images%2==0
out=run/f'duplicate_inspection_{a.step}';out.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(42)
cfg=get_cfg();add_deeplab_config(cfg);add_maskformer2_config(cfg);cfg.merge_from_file(str(run/'supervised.yaml'));cfg.SOLVER.IMS_PER_BATCH=2;cfg.INPUT.SYNTHETIC_SCENE.SEED=904321
model=Trainer.build_model(cfg).eval().requires_grad_(False);ckpt=run/f'supervised/model_{a.step-1:07d}.pth';DetectionCheckpointer(model).load(str(ckpt))
rows=[]; examples={}; names=['terrain','house','tree','cloud','car','person']
def iou(x,y):
 return float((x&y).sum())/max(1,int((x|y).sum()))
# Keep a bounded set of images per class, then select examples near each class median.
with torch.inference_mode():
 for iteration,batch in enumerate(SyntheticSceneBatchLoader(cfg,True)):
  if iteration==a.images//2:break
  images=ImageList.from_tensors([(x['image'].to(model.device)-model.pixel_mean)/model.pixel_std for x in batch],model.size_divisibility)
  raw=model.sem_seg_head(model.backbone(images.tensor));targets=[dict(labels=x['instances'].gt_classes,masks=x['instances'].gt_masks) for x in batch]
  matches=model.criterion.matcher(raw,repeat_object_targets(targets,2))
  masks=F.interpolate(raw['pred_masks'].float(),size=(512,512),mode='bilinear',align_corners=False)>0
  cls=raw['pred_logits'][...,:-1].argmax(-1)
  for b,(record,target,(src,dst)) in enumerate(zip(batch,targets,matches)):
   for obj in (dst//2).unique().tolist():
    category=int(target['labels'][obj]);qq=src[dst//2==obj]
    if category==0 or len(qq)!=2:continue
    x,y=masks[b,qq];gt=target['masks'][obj];gx,gy=iou(x,gt),iou(y,gt);union=iou(x|y,gt)
    row=dict(image_id=record['image_id'],object=obj,class_name=names[category],queries=qq.tolist(),pair_iou=iou(x,y),gt_ious=[gx,gy],union_gt_iou=union,union_gain=union-max(gx,gy),both_empty=not bool((x|y).any()),class_agreement=bool(cls[b,qq[0]]==cls[b,qq[1]]),both_classes_correct=bool((cls[b,qq]==category).all()),gt_area=int(gt.sum()))
    rows.append(row)
    if row['gt_area']>=200 and len(examples.setdefault(category,[]))<80:
     examples[category].append(dict(row=row,rgb=record['image'].permute(1,2,0).byte().cpu().numpy(),gt=gt.cpu().numpy(),masks=torch.stack([x,y]).cpu().numpy()))
  print('Processed',(iteration+1)*2,flush=True)
def describe(rr):
 v=np.array([r['pair_iou'] for r in rr]);g=np.array([r['gt_ious'] for r in rr]);return dict(count=len(rr),mean_pair_iou=float(v.mean()),median_pair_iou=float(np.median(v)),pair_iou_ge_90=float((v>=.9).mean()),pair_iou_ge_75=float((v>=.75).mean()),pair_iou_lt_50=float((v<.5).mean()),both_empty_fraction=float(np.mean([r['both_empty'] for r in rr])),mean_best_gt_iou=float(g.max(1).mean()),mean_worst_gt_iou=float(g.min(1).mean()),mean_union_gain=float(np.mean([r['union_gain'] for r in rr])),class_agreement=float(np.mean([r['class_agreement'] for r in rr])))
report=dict(checkpoint=str(ckpt),images=a.images,all_pairs=describe(rows),by_class={n:describe([r for r in rows if r['class_name']==n]) for n in names[1:]})
(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'pairs.json').write_text(json.dumps(rows,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
chosen=[]
for c in [1,2,4,5,3]:
 median=report['by_class'][names[c]]['median_pair_iou'];chosen.append(min(examples[c],key=lambda e:abs(e['row']['pair_iou']-median)))
# Include a lower-overlap example to show what disagreement looks like.
chosen.append(min([e for es in examples.values() for e in es if min(e['row']['gt_ious'])>.15],key=lambda e:e['row']['pair_iou']))
fig,axes=plt.subplots(len(chosen),4,figsize=(12,3*len(chosen)),squeeze=False)
for axis,ex in zip(axes,chosen):
 r=ex['row'];gt=ex['gt'];yy,xx=np.where(gt);margin=15;x0=max(0,xx.min()-margin);x1=min(512,xx.max()+margin+1);y0=max(0,yy.min()-margin);y1=min(512,yy.max()+margin+1);sl=np.s_[y0:y1,x0:x1];rgb=ex['rgb'][sl];x,y=ex['masks'][:,y0:y1,x0:x1]
 titles=[f"{r['class_name']} | {r['image_id']}:{r['object']}\npair IoU {r['pair_iou']:.2f}",f"q{r['queries'][0]} | GT IoU {r['gt_ious'][0]:.2f}",f"q{r['queries'][1]} | GT IoU {r['gt_ious'][1]:.2f}",'Agreement / disagreement']
 for j,ax in enumerate(axis):
  ax.imshow(rgb)
  overlay=np.zeros(x.shape+(4,))
  if j in [1,2]:overlay[[x,y][j-1]]=[1,.05,.65,.55]
  if j==3:
   overlay[x&y]=[.2,1,.2,.6];overlay[x&~y]=[1,.15,.1,.8];overlay[y&~x]=[.1,.5,1,.8]
  ax.imshow(overlay);ax.contour(gt[sl],levels=[.5],colors='white',linewidths=.8);ax.set_title(titles[j],fontsize=9);ax.axis('off')
fig.tight_layout();fig.savefig(out/'duplicate_examples.png',dpi=130);plt.close(fig)
(out/'examples.json').write_text(json.dumps([e['row'] for e in chosen],indent=2));print(json.dumps(report,indent=2),flush=True)
