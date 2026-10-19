import math,types
import torch
from mask2former.modeling.object_decoder import ObjectDecoder

def check():
    m=ObjectDecoder(8,8,3,2,2,16).eval()
    logits=torch.tensor([[0.,0.,0.,math.log(2)]])
    assert m._greedy_action(logits).item()==0
    assert m._greedy_action(logits,0).item()==3
    exact=torch.tensor([[-torch.inf,0.,0.,math.log(2)]])
    assert m._greedy_action(exact).item()==1  # strict >, not >=
    assert m._greedy_action(torch.tensor([[-torch.inf]*3+[0.]])).item()==3
    emb=torch.randn(2,3,8);features=[torch.randn(2,16,8) for _ in range(3)];masks=torch.randn(2,3,4,4)
    args=(emb,features,masks,[(4,4)]*3)
    with torch.no_grad():
        torch.manual_seed(7);a=m.rollout(*args,sample=True,max_steps=4,return_actions=True)
        m.eof_threshold=0
        torch.manual_seed(7);b=m.rollout(*args,sample=True,max_steps=4,return_actions=True)
    assert a[2]==b[2];torch.testing.assert_close(a[1],b[1])
    scored=m.autoregressive_log_probability(*args,a[2]);torch.testing.assert_close(scored,a[1],atol=3e-6,rtol=3e-6)
    scored.sum().backward()
    print('EOF default: strict gate, argmax opt-out, exclusions, stochastic sample and likelihood invariance PASS')
if __name__=='__main__':
    torch.set_num_threads(4);check()
