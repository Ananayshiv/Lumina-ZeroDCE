import torch
from lumina.train import DCENet,loss_parts,LowLightDataset

def test_forward_and_gradients():
    net=DCENet(width=8);x=torch.rand(2,3,32,32)*.2
    y,a=net(x);assert y.shape==x.shape and a.shape==(2,24,32,32)
    loss,parts=loss_parts(x,y,a);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in parts.values())
    assert net.c1.weight.grad is not None

def test_dataset():
    x=LowLightDataset('synthetic',size=32,count=4)[0]
    assert x.shape==(3,32,32) and 0<=x.min()<=x.max()<=1


def test_benchmark_degradation_and_metrics():
    from lumina.benchmark import degrade, psnr, ssim_global
    import numpy as np
    ref=np.ones((12,12,3),dtype=np.float32)*.8
    a=degrade(ref,123);b=degrade(ref,123)
    assert np.allclose(a,b) and float(a.mean())<.8
    ref_t=torch.ones(1,3,12,12)*.8
    assert psnr(ref_t,ref_t)>=100
    assert .99<=ssim_global(ref_t,ref_t)<=1.01
