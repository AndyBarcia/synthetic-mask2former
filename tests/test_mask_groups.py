import math,types
import numpy as np
import torch
from mask2former.modeling.mask_groups import group_logits,mask_compatibility
from mask2former.modeling.object_decoder import ObjectDecoder

def test_group_math():
    raw=torch.tensor([[.2,.25,.3,.25]]).log().requires_grad_()
    compatible=np.eye(3,dtype=bool)[None];compatible[:,0,1]=compatible[:,1,0]=True
    collapsed=group_logits(raw,compatible);p=collapsed.softmax(-1)
    torch.testing.assert_close(p,torch.tensor([[0.,.45,.3,.25]]))
    torch.testing.assert_close(p[:,-1],raw.softmax(-1)[:,-1])
    (-collapsed.log_softmax(-1)[0,1]).backward()
    assert raw.grad[0,0]<0 and raw.grad[0,1]<0  # both aliases receive credit
    chain=np.array([[[1,1,0],[1,1,1],[0,1,1]]],dtype=bool)
    x=group_logits(torch.tensor([[3.,2.,1.,0.]]),chain)
    assert torch.isfinite(x).tolist()==[[True,False,True,True]]
    masked=torch.tensor([[-torch.inf,-torch.inf,-torch.inf,1.]])
    torch.testing.assert_close(group_logits(masked,chain).softmax(-1),torch.tensor([[0.,0.,0.,1.]]))
    masks=torch.ones(1,3,2,2);labels=torch.tensor([[0,0,1]])
    c=mask_compatibility(masks,labels,.8)
    assert c[0,0,1] and not c[0,0,2]
    print('group probability/EOF conservation, shared gradients, no chaining, empty masks/classes PASS',flush=True)

def test_scripted_selection():
    m=ObjectDecoder(8,8,3,2,2,16,collapse_masks=True)
    def fixed(self,emb,features,regions,previous,exclusions,**kwargs):
        return torch.tensor([.2,.25,.3,.25]).log().to(emb).expand(len(emb),previous.shape[1],-1).clone()
    m.forward=types.MethodType(fixed,m)
    emb=torch.randn(1,3,8);f=[torch.randn(1,4,8)]*3
    masks=torch.tensor([[[[1.,-1.],[1.,-1.]],[[1.,-1.],[1.,-1.]],[[-1.,1.],[-1.,1.]]]])
    cls=torch.tensor([[0,0,1]])
    generated=m.generate(emb,f,masks,[(2,2)]*3,query_classes=cls)
    assert generated[0][0].tolist()==[1,2]
    torch.testing.assert_close(generated[0][1][0],torch.tensor(math.log(.45/.25)))
    greedy=m.rollout(emb,f,masks,[(2,2)]*3,sample=False,max_steps=4,query_classes=cls)
    assert greedy[0]==[[1,2]]
    print('inference and greedy RL select probability groups with strict EOF gate PASS',flush=True)

def test_sampling_scoring(device='cpu',amp=False):
    torch.manual_seed(29)
    m=ObjectDecoder(8,8,3,2,2,16,collapse_masks=True).to(device).train()
    batch=8
    emb=torch.randn(batch,3,8,device=device);f=[torch.randn(batch,4,8,device=device) for _ in range(3)]
    masks=torch.tensor([[[[1.,-1.],[1.,-1.]],[[1.,-1.],[1.,-1.]],[[-1.,1.],[-1.,1.]]]],device=device).expand(batch,-1,-1,-1)
    cls=torch.tensor([[0,0,1]],device=device).expand(batch,-1)
    args=(emb,f,masks,[(2,2)]*3)
    with torch.autocast(device_type=device,enabled=amp):
        with torch.no_grad():
            orders,online,actions=m.rollout(*args,sample=True,max_steps=4,return_actions=True,query_classes=cls)
        scored=m.trajectory_log_probability(*args,actions,query_classes=cls)
    torch.testing.assert_close(online,scored,atol=.02 if amp else 5e-6,rtol=.005 if amp else 5e-6)
    assert all(not(0 in order and 1 in order) for order in orders)
    (-scored.mean()).backward()
    assert torch.isfinite(scored).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    assert m.eof.grad.abs().sum()>0
    if amp:
        with torch.no_grad(), torch.autocast('cuda'):
            generated=m.generate(emb.half(),[x.half() for x in f],masks.half(),[(2,2)]*3,query_classes=cls)
        assert len(generated)==batch
    print('group sampling/rescoring/AMP parity, trainable map policy, stochastic EOF gradient PASS',device,amp,flush=True)

def test_criterion_class_routing():
    from torch import nn
    from mask2former.modeling.criterion import SetCriterion
    for objective in ("policy_gradient", "best_of_n_ft"):
        c=SetCriterion.__new__(SetCriterion);nn.Module.__init__(c)
        c.object_decoder=ObjectDecoder(8,8,3,2,2,16,collapse_masks=True)
        c.object_rl_num_samples=4;c.object_rl_max_steps=4;c.object_rl_reward="image_pq"
        c.object_rl_baseline="rloo_greedy";c.object_rl_objective=objective
        masks=torch.tensor([[[[1.,-1.],[1.,-1.]],[[1.,-1.],[1.,-1.]],[[-1.,1.],[-1.,1.]]]]).expand(2,-1,-1,-1)
        outputs=dict(mask_embeddings=torch.randn(2,3,8),object_decoder_image_features=[torch.randn(2,4,8) for _ in range(3)],pred_masks=masks,object_decoder_image_sizes=[(2,2)]*3,pred_logits=torch.tensor([[[2.,0.,-4.],[2.,0.,-4.],[0.,2.,-4.]]]).expand(2,-1,-1))
        c._object_rollout_rewards=lambda *args:(torch.arange(8.).reshape(4,2)/8,torch.zeros(2))
        method=c.loss_object_rl_eof if objective=="policy_gradient" else c.loss_object_best_of_n
        loss=method(outputs,[])["loss_object_rl"];assert torch.isfinite(loss);loss.backward()
        assert c.object_decoder.eof.grad is not None
    print('real criterion RLOO and best-of-N route predicted classes through sampling and rescoring PASS',flush=True)

def test_representative_flip(device='cpu', amp=False):
    """Replay sampled topology when numerical rounding reverses tied queries."""
    m = ObjectDecoder(8,8,3,2,2,16,collapse_masks=True).to(device)
    m.test_logits = torch.nn.Parameter(torch.tensor([8.,8.,-8.,0.],device=device))
    def fixed(self,emb,features,regions,previous,exclusions,**kwargs):
        # Emulate the opposite rounding of incremental versus batched decoding.
        perturbation = self.test_logits.new_tensor(
            [1e-4,0.,0.,0.] if not torch.is_grad_enabled() else [0.,1e-4,0.,0.])
        return (self.test_logits + perturbation).expand(len(emb),previous.shape[1],-1)
    m.forward = types.MethodType(fixed,m)
    emb=torch.randn(1,3,8,device=device)
    features=[torch.randn(1,4,8,device=device) for _ in range(3)]
    masks=torch.tensor([[[[1.,-1.],[1.,-1.]],[[1.,-1.],[1.,-1.]],
                         [[-1.,1.],[-1.,1.]]]],device=device)
    cls=torch.tensor([[0,0,1]],device=device)
    args=(emb,features,masks,[(2,2)]*3)
    with torch.autocast(device_type=device,enabled=amp):
        with torch.no_grad():
            _, online, actions=m.rollout(*args,return_actions=True,use_cache=False,
                                        max_steps=4,query_classes=cls)
        assert actions[0][0] == 0
        broken=m.trajectory_log_probability(*args,[list(path) for path in actions],query_classes=cls)
        assert torch.isneginf(broken).all()  # Original failure, without saved groups.
        scored=m.trajectory_log_probability(*args,actions,query_classes=cls)
        torch.testing.assert_close(scored,online,atol=1e-5,rtol=1e-5)
        (-scored.mean()).backward()
    assert torch.isfinite(m.test_logits.grad).all()
    assert m.test_logits.grad[0] != 0 and m.test_logits.grad[1] != 0
    # Non-transitive compatibility can change the entire partition, not only
    # the representative. Preserve all memberships from the sampling pass.
    chain=np.array([[[1,1,0],[1,1,1],[0,1,1]]],dtype=bool)
    raw=torch.tensor([[3.,3.-1e-4,2.,0.]],device=device)
    _, topology=group_logits(raw,chain,return_indices=True)
    flipped=raw.clone();flipped[0,1]+=2e-4;flipped.requires_grad_()
    replay=group_logits(flipped,chain,group_indices=topology)
    assert torch.isfinite(replay).tolist()==[[True,False,True,True]]
    (-replay.log_softmax(-1)[0,0]).backward()
    assert torch.isfinite(flipped.grad).all()
    print('near-tied representatives and complete-link topology replay PASS',device,amp,flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);test_group_math();test_scripted_selection();test_sampling_scoring();test_criterion_class_routing();test_representative_flip()
    if torch.cuda.is_available():
        test_sampling_scoring('cuda',True);test_representative_flip('cuda',True)
