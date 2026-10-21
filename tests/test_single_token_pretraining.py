"""Supervision samples one teacher prefix and encodes exactly one state per image."""
import copy
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.object_decoder import ObjectDecoder


def check_sampler(device):
    sources=[torch.tensor(v,device=device) for v in ([3,1,4],[0,2],[2],[])]
    sources=[x.long() for x in sources]
    indices=[(x,torch.arange(len(x),device=device)) for x in sources]
    priority=torch.tensor([[.9,.2,.8,.1,.3],[.1,.9,.2,.8,.7],
                           [.9,.8,.1,.7,.6],[.9,.8,.7,.6,.5]],device=device)
    # Exercise empty prefix, an interior prediction, EOF, and an empty scene.
    prediction=torch.tensor([[0],[1],[1],[0]],device=device)
    captured=[]
    def choose(weights,n):
        captured.append(weights)
        return prediction
    with patch('torch.rand',return_value=priority),patch('torch.multinomial',side_effect=choose):
        previous,target=SetCriterion._sample_teacher_state(indices,5,device)
    assert target.tolist()==[3,2,5,5]
    assert previous.tolist()==[[-1]*6,[-1,0,-1,-1,-1,-1],[-1,2,-1,-1,-1,-1],[-1]*6]
    expected=(torch.arange(6,device=device)[None] <= torch.tensor([3,2,1,0],device=device)[:,None]).float()
    torch.testing.assert_close(captured[0],expected)
    # Padding and unmatched queries never enter the sampled teacher history.
    for _ in range(5):
        previous,target=SetCriterion._sample_teacher_state(indices,5,device)
        for row,source,t in zip(previous,sources,target):
            prefix=row[row>=0]
            assert len(prefix.unique())==len(prefix)
            assert all(int(v) in source.tolist() for v in prefix)
            assert int(t)==5 or (int(t) in source.tolist() and not (prefix==t).any())
            assert int(t)!=5 or len(prefix)==len(source)
    empty=torch.empty(0,device=device,dtype=torch.long)
    previous,target=SetCriterion._sample_teacher_state([(empty,empty)]*2,5,device)
    assert (previous==-1).all() and (target==5).all()
    complete=torch.arange(5,device=device)
    with patch('torch.multinomial',return_value=torch.tensor([[5]],device=device)):
        previous,target=SetCriterion._sample_teacher_state([(complete,complete)],5,device)
    assert int(target[0])==5 and previous[0,1:].unique().numel()==5 and (previous[0,1:]>=0).all()
    print('uniform teacher prediction points, BOS/interior/EOF/empty scenes and no leakage PASS',device)


def check_loss(device,amp=False):
    torch.manual_seed(63)
    batch,queries,width=3,5,16
    model=ObjectDecoder(width,width,queries,3,4,32,map_feature_dim=width).to(device).train()
    reference=copy.deepcopy(model)
    c=SetCriterion.__new__(SetCriterion);nn.Module.__init__(c);c.object_decoder=model
    embeddings=torch.randn(batch,queries,width,device=device,requires_grad=True)
    features=[torch.randn(batch,n,width,device=device,requires_grad=True) for n in [4,16,64,256]]
    masks=torch.randn(batch,queries,16,16,device=device)
    outputs=dict(mask_embeddings=embeddings,object_decoder_image_features=features,
                 pred_masks=masks,object_decoder_image_sizes=[(2,2),(4,4),(8,8),(16,16)])
    previous=torch.tensor([[-1,-1,-1,-1,-1,-1],[-1,0,2,-1,-1,-1],[-1,1,3,4,-1,-1]],device=device)
    target=torch.tensor([0,4,queries],device=device)
    ref_embeddings=embeddings.detach().clone().requires_grad_()
    ref_features=[x.detach().clone().requires_grad_() for x in features]
    with patch.object(c,'_sample_teacher_state',return_value=(previous,target)), \
         patch.object(model.map_encoder,'encode_maps',wraps=model.map_encoder.encode_maps) as encode, \
         patch.object(model,'pack_prefix_trees',side_effect=AssertionError('must not build prefix tree')), \
         torch.autocast(device_type=device,enabled=amp):
        actual=c.loss_object_decoder(outputs,[])['loss_object_decoder']
        assert encode.call_count==1
        assert encode.call_args.args[3].shape==(batch,1,256)
        regions=reference.prepare_regions(masks,outputs['object_decoder_image_sizes'])
        exclusions=reference.prepare_vocabulary_exclusions(masks)
        expected=F.cross_entropy(reference(ref_embeddings,ref_features,regions,previous,exclusions)[:,-1].float(),target)
    torch.testing.assert_close(actual,expected,atol=.003 if amp else 3e-6,rtol=.01 if amp else 3e-6)
    actual.backward();expected.backward()
    for x,y in zip([embeddings,*features,*model.parameters()],
                   [ref_embeddings,*ref_features,*reference.parameters()]):
        assert x.grad is not None and y.grad is not None
        torch.testing.assert_close(x.grad,y.grad,atol=.003 if amp else 2e-5,rtol=.03 if amp else 2e-5)
    print('exactly one encoded state/image, sampled CE and full-sequence reference gradients PASS',device,amp)


if __name__=='__main__':
    torch.set_num_threads(2)
    check_sampler('cpu');check_loss('cpu')
    if torch.cuda.is_available():
        check_sampler('cuda');check_loss('cuda');check_loss('cuda',True)
