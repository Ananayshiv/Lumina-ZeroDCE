"""Held-out PAIRED SYNTHETIC benchmark (not an official Zero-DCE or LOL benchmark).

Evaluates image restoration after deterministic simulated low-light degradation.
There is no ground truth for real unpaired photos: PSNR/SSIM here are ONLY
valid against the procedurally generated clean reference images.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from lumina.train import DCENet, synth_image, as_pil, quality_stats


def degrade(clean, seed):
    """Deterministic low-light corruption, independent of original train.py RNG."""
    rng=np.random.default_rng(seed)
    gamma=float(rng.uniform(1.6,2.5))
    strength=float(rng.uniform(.3,.65))
    return np.clip(clean**gamma*strength,0,1).astype(np.float32)


def gamma_correct(x, exponent=.5):
    return torch.clamp(x,0,1).pow(exponent)


def psnr(a,b):
    mse=F.mse_loss(a,b).item()
    return float(10*np.log10(1/max(mse,1e-12)))


def ssim_global(a,b):
    """Global-channel SSIM-like score: explicitly NOT the standard windowed SSIM."""
    # For true standard SSIM compare via pytorch_msssim or skimage.
    dims=(2,3);ma=a.mean(dim=dims);mb=b.mean(dim=dims)
    va=((a-ma[...,None,None])**2).mean(dim=dims)
    vb=((b-mb[...,None,None])**2).mean(dim=dims)
    cov=((a-ma[...,None,None])*(b-mb[...,None,None])).mean(dim=dims)
    c1=.01**2;c2=.03**2
    return float((((2*ma*mb+c1)*(2*cov+c2))/((ma**2+mb**2+c1)*(va+vb+c2))).mean().item())


def run(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(args.checkpoint,weights_only=True,map_location=device)
    model=DCENet(width=ckpt.get('width',32)).to(device).eval()
    model.load_state_dict(ckpt['model'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(args.n):
        # disjoint train/test synthetic seeds: training uses 42+train_idx by default
        seed=100_000+i
        ref=np.asarray(synth_image(seed,args.size),dtype=np.float32)/255.0
        dark=degrade(ref,seed)
        x=torch.from_numpy(dark).permute(2,0,1).unsqueeze(0).to(device)
        target=torch.from_numpy(ref.copy()).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            enhanced=model(x)[0]
            gamma=gamma_correct(x)
        row={'seed':seed}
        for label,result in [('input',x),('gamma',gamma),('zero_dce',enhanced)]:
            row[f'{label}_psnr']=psnr(result,target)
            row[f'{label}_global_ssim']=ssim_global(result,target)
            row[f'{label}_brightness']=quality_stats(result[0])['brightness']
        rows.append(row)
        if i<min(args.preview,4):
            imgs=[as_pil(v[0]) for v in (x,gamma,enhanced,target)]
            canvas=Image.new('RGB',(args.size*4,args.size),'white')
            for j,im in enumerate(imgs):canvas.paste(im,(j*args.size,0))
            canvas.resize((args.size*8,args.size*2)).save(out/f'example_{i:02d}.png')
    keys=list(rows[0])
    with (out/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    summary={k:{'mean':float(np.mean([r[k] for r in rows])),
                'std':float(np.std([r[k] for r in rows]))} for k in keys if k!='seed'}
    summary['evaluation_protocol']='Held-out procedurally-generated images, with deterministic simulated low-light corruption; not a real-image or official-paper benchmark.'
    summary['metric_note']='global_ssim is the global-statistics SSIM variant (not windowed SSIM); use external library for standardized SSIM.'
    summary['n_images']=len(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='outputs/model.pt')
    p.add_argument('--n',type=int,default=64)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--preview',type=int,default=4)
    p.add_argument('--out',default='benchmark_outputs')
    return p.parse_args()

if __name__=='__main__':run(parse_args())
