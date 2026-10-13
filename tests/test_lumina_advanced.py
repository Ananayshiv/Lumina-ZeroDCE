import numpy as np
import torch
from PIL import Image
from lumina.evaluate_advanced import bootstrap_ci,paired_files,load_pair,evaluate
from lumina.train import DCENet

def test_bootstrap_reproducible_and_singleton():
    assert bootstrap_ci([2,2,2])==[2,2]
    assert bootstrap_ci([1,2,3],seed=12)==bootstrap_ci([1,2,3],seed=12)

def test_paired_real_image_protocol(tmp_path):
    d=tmp_path/'dark';r=tmp_path/'ref';d.mkdir();r.mkdir()
    Image.new('RGB',(32,32),(40,20,20)).save(d/'A.png')
    Image.new('RGB',(32,32),(160,140,140)).save(r/'A.png')
    items=paired_files(d,r)
    assert len(items)==1
    low,ref=load_pair(items[0][1],items[0][2],64)
    assert low.shape==ref.shape==(1,3,32,32)
    ckpt=tmp_path/'m.pt'
    torch.save({'model':DCENet(width=8).state_dict(),'width':8},ckpt)
    result=evaluate(str(ckpt),str(tmp_path/'out'),str(d),str(r),preview=1)
    assert result['dataset_type']=='paired-real' and len(result['model_minus_gamma_psnr_95pct_bootstrap_ci'])==2
    assert (tmp_path/'out/comparison_00.png').is_file()

def test_misalignment_rejected(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    Image.new('RGB',(32,32)).save(a/'a.png')
    Image.new('RGB',(20,20)).save(b/'a.png')
    k,x,y=paired_files(a,b)[0]
    try:load_pair(x,y,50);assert False
    except ValueError as e:assert 'Unaligned pair' in str(e)
