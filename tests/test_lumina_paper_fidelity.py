"""Deterministic algorithm-level checks against paper math; not performance parity."""
import numpy as np
from PIL import Image
import torch
from lumina.train import DCENet, spatial_loss, exposure_loss, color_loss, curve_tv_loss, adaptive_exposure_target, LowLightDataset


def test_curve_identity_and_eight_step_recurrence():
    net=DCENet(width=8)
    for p in net.parameters():
        p.data.zero_()
    x=torch.rand(1,3,32,32)
    result,curves=net(x)
    assert torch.allclose(result,x,atol=1e-6)
    assert curves.shape[1]==24
    net.c7.bias.data.fill_(0.3)
    with torch.no_grad():result,curves=net(x)
    expected=x.clone()
    for param in curves.chunk(8,dim=1):
        expected=expected+param*expected*(1-expected)
    assert torch.allclose(result,expected,atol=1e-6)


def test_spatial_abs_difference_and_no_wraparound():
    # Reverse patch luminance gradient: abs gradients are identical.
    x=torch.zeros(1,3,8,12)
    y=torch.zeros_like(x)
    x[:,:,:,0:4]=.2;x[:,:,:,4:8]=.5;x[:,:,:,8:12]=.8
    y[:,:,:,0:4]=.8;y[:,:,:,4:8]=.5;y[:,:,:,8:12]=.2
    assert spatial_loss(x,y).abs()<1e-7
    # Even with a changed far edge, it must not be compared cyclically to first edge.
    # Explicit expected adjacent gradient losses: (|.5-.2|-|.5-.2|)^2=0,
    # (|.9-.5|-|.8-.5|)^2=.01, averaged over two horizontal edges and
    # a vertical direction of zero -> .0025.
    y=x.clone();y[:,:,:,8:12]=.9
    assert torch.allclose(spatial_loss(x,y),torch.tensor(.0025),atol=1e-6)


def test_nonreference_loss_properties_and_extension():
    gray=torch.full((2,3,32,32),.6)
    assert exposure_loss(gray).item()<1e-6
    assert torch.allclose(color_loss(gray),torch.tensor(0.))
    assert torch.allclose(curve_tv_loss(torch.ones(2,24,32,32)),torch.tensor(0.))
    x=torch.cat([torch.ones(1,3,32,16)*.03,torch.ones(1,3,32,16)*.8],dim=-1)
    exposure=adaptive_exposure_target(x)
    assert exposure.shape==(1,1,2,2)
    assert (exposure[...,0] > exposure[...,1]).all()


def test_real_folder_images_not_darkened_twice(tmp_path):
    clean=np.full((32,32,3),128,dtype=np.uint8)
    Image.fromarray(clean).save(tmp_path/'already_dark.jpg')
    ds=LowLightDataset(mode='folder',folder=str(tmp_path),size=32)
    a=ds[0];b=ds[0]
    assert torch.allclose(a,b)
    assert .48<a.mean()<.52


def test_paper_exposure_absolute_and_color_sum_squares():
    sample=torch.full((1,3,32,32),0.4)
    # Eq. (5): |0.4-0.6|, not the squared-error in authors' public code.
    assert torch.allclose(exposure_loss(sample),torch.tensor(0.2),atol=1e-6)
    color=torch.zeros(1,3,32,32)
    color[:,0]=1.0
    # pairwise differences: (R-G)=1, (R-B)=1, (G-B)=0
    assert torch.allclose(color_loss(color),torch.tensor(2.0),atol=1e-6)


def test_exact_dce_net_parameter_count():
    net=DCENet()
    assert sum(p.numel() for p in net.parameters())==79416


def test_four_losses_have_finite_parameter_gradients():
    from lumina.train import loss_parts
    model=DCENet(width=8)
    x=torch.rand(2,3,32,32)
    y,curves=model(x)
    total,_=loss_parts(x,y,curves)
    total.backward()
    assert torch.isfinite(total)
    assert any(p.grad is not None for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
