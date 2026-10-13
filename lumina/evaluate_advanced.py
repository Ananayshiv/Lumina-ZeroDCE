"""Paired low-light restoration benchmark with paired bootstrap uncertainty.

For synthetic tests the clean reference is procedural. For --dark-dir/--reference-dir,
input images must be aligned and matched by filename stem. Do not train on the test pairs.
This is an independent evaluation protocol, NOT official Zero-DCE leaderboard parity.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps,ImageDraw
from lumina.train import DCENet,synth_image,as_pil
from lumina.benchmark import degrade,psnr,gamma_correct

SUFFIXES={'.png','.jpeg','.jpg','.webp','.bmp'}

def paired_files(dark_dir,ref_dir):
    def listing(d):
        paths=[p for p in Path(d).rglob('*') if p.suffix.lower() in SUFFIXES]
        byname={}
        for p in paths:
            key=p.stem.lower()
            if key in byname:raise ValueError(f'Duplicate basename: {key} in {d}')
            byname[key]=p
        return byname
    a,b=listing(dark_dir),listing(ref_dir)
    if not a or set(a)!=set(b):
        raise ValueError('Dark/reference folders must contain matching nonempty filename stems')
    return [(key,a[key],b[key]) for key in sorted(a)]


def load_pair(low_path,ref_path,max_side):
    with Image.open(low_path) as f: dark=f.convert('RGB').copy()
    with Image.open(ref_path) as f: ref=f.convert('RGB').copy()
    if dark.size != ref.size:
        raise ValueError(f'Unaligned pair: {low_path} / {ref_path} dimensions {dark.size} != {ref.size}')
    dark.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    ref.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    x=torch.from_numpy(np.asarray(dark,dtype=np.float32).copy()).permute(2,0,1).unsqueeze(0)/255
    y=torch.from_numpy(np.asarray(ref,dtype=np.float32).copy()).permute(2,0,1).unsqueeze(0)/255
    return x,y


def bootstrap_ci(values,seed=42,repeats=2000):
    values=np.asarray(values,dtype=np.float64)
    if not len(values):raise ValueError('Empty evaluation set')
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(values),(repeats,len(values)))
    means=values[samples].mean(axis=1)
    return [float(v) for v in np.quantile(means,[.025,.975])]


def evaluate(checkpoint,out='eval_advanced',dark_dir='',reference_dir='',n=64,size=64,max_side=1024,seed=2026,preview=5):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights=torch.load(checkpoint,map_location=device,weights_only=True)
    model=DCENet(width=int(weights.get('width',32))).to(device).eval()
    model.load_state_dict(weights['model'])
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    if bool(dark_dir)!=bool(reference_dir):raise ValueError('Specify both --dark-dir and --reference-dir')
    use_pairs=bool(dark_dir)
    pairs=paired_files(dark_dir,reference_dir) if use_pairs else list(range(n))
    if not pairs:raise ValueError('No images for evaluation')
    rows=[]
    with torch.inference_mode():
        for i,item in enumerate(pairs):
            if use_pairs:
                name,low,reference=item;x,target=load_pair(low,reference,max_side)
            else:
                name=str(100000+i)
                ref=np.asarray(synth_image(100000+i,size),dtype=np.float32)/255.
                dark=degrade(ref,100000+i)
                x=torch.from_numpy(dark).permute(2,0,1).unsqueeze(0)
                target=torch.from_numpy(ref.copy()).permute(2,0,1).unsqueeze(0)
            x=x.to(device);target=target.to(device)
            generated=model(x)[0]
            gamma=gamma_correct(x)
            mse=lambda a,b:float((a-b).square().mean().item())
            row=dict(image=name,input_psnr=psnr(x,target),gamma_psnr=psnr(gamma,target),model_psnr=psnr(generated,target),
                     input_mae=float((x-target).abs().mean().item()),
                     gamma_mae=float((gamma-target).abs().mean().item()),
                     model_mae=float((generated-target).abs().mean().item()),
                     model_clip_pct=float((generated>.98).float().mean().item()*100),
                     model_minus_gamma_psnr=float(psnr(generated,target)-psnr(gamma,target)))
            rows.append(row)
            if i<preview:
                images=[as_pil(z[0]) for z in (x,gamma,generated,target)]
                w,h=images[0].size;panel=Image.new('RGB',(w*4,h+28),'white')
                draw=ImageDraw.Draw(panel)
                for j,(im,label) in enumerate(zip(images,['LOW-LIGHT','GAMMA','ZERO-DCE','REFERENCE'])):
                    panel.paste(im,(j*w,28));draw.text((j*w+4,5),label,fill='#111111')
                panel.save(folder/f'comparison_{i:02d}.png')
    with (folder/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    names=['input_psnr','gamma_psnr','model_psnr','input_mae','gamma_mae','model_mae','model_clip_pct','model_minus_gamma_psnr']
    result={'n':len(rows),'dataset_type':'paired-real' if use_pairs else 'held-out-procedural',
            'protocol':'Paired PSNR/MAE for aligned reference images; held-out synthetic unless paired real folders supplied.',
            'means':{k:float(np.mean([row[k] for row in rows])) for k in names},
            'model_minus_gamma_psnr_95pct_bootstrap_ci':bootstrap_ci([r['model_minus_gamma_psnr'] for r in rows],seed=seed),
            'bootstrap_seed':seed,'no_paper_benchmark_claim':True}
    (folder/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',default='eval_advanced')
    ap.add_argument('--dark-dir',default='');ap.add_argument('--reference-dir',default='')
    ap.add_argument('--n',type=int,default=64);ap.add_argument('--size',type=int,default=64)
    ap.add_argument('--max-side',type=int,default=1024);ap.add_argument('--preview',type=int,default=5)
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    evaluate(args.checkpoint,args.out,args.dark_dir,args.reference_dir,args.n,args.size,args.max_side,args.seed,args.preview)
