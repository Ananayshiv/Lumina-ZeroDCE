"""Zero-DCE (CVPR 2020): independent from-scratch algorithm reproduction.
Quickstart: python train.py --dataset synthetic --steps 10 --batch-size 2
Better Colab: python train.py --dataset cifar10 --steps 500 --batch-size 8
"""
import argparse, csv, math, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class DCENet(nn.Module):
    """Seven convolutions, symmetric skip connections, 8 RGB curve maps."""
    def __init__(self, width=32):
        super().__init__()
        self.c1=nn.Conv2d(3,width,3,padding=1)
        self.c2=nn.Conv2d(width,width,3,padding=1)
        self.c3=nn.Conv2d(width,width,3,padding=1)
        self.c4=nn.Conv2d(width,width,3,padding=1)
        self.c5=nn.Conv2d(width*2,width,3,padding=1)
        self.c6=nn.Conv2d(width*2,width,3,padding=1)
        self.c7=nn.Conv2d(width*2,24,3,padding=1)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.normal_(m.weight,mean=0,std=0.02); nn.init.constant_(m.bias,0)

    def forward(self,x):
        a1=F.relu(self.c1(x)); a2=F.relu(self.c2(a1)); a3=F.relu(self.c3(a2)); a4=F.relu(self.c4(a3))
        a5=F.relu(self.c5(torch.cat([a3,a4],dim=1)))
        a6=F.relu(self.c6(torch.cat([a2,a5],dim=1)))
        curves=torch.tanh(self.c7(torch.cat([a1,a6],dim=1)))
        y=x
        for a in curves.chunk(8,dim=1):
            # Eq. (3): LE_n = LE_(n-1) + A_n * LE_(n-1) * (1-LE_(n-1)).
            # The original released code uses the algebraically sign-flipped
            # a*(y*y-y); tanh makes the model classes equivalent under a -> -a.
            y=y+a*y*(1-y)
        return torch.clamp(y,0,1),curves


def spatial_loss(x,y):
    """Paper Eq. (4): compare absolute four-neighbor gradients of 4x4 means.

    No cyclic/torch.roll boundaries: cyclic wrapping would connect unrelated
    left/right or top/bottom image edges and invent false spatial neighbors.
    """
    x=F.avg_pool2d(x.mean(1,keepdim=True),4)
    y=F.avg_pool2d(y.mean(1,keepdim=True),4)
    if min(x.shape[-2:]) < 2:
        raise ValueError('spatial_loss requires images at least 8x8')
    # Four directions per interior cell. Use non-wrapped shared valid neighbors.
    terms=[]
    for dim in (-2,-1):
        x0=x.narrow(dim,0,x.shape[dim]-1)
        x1=x.narrow(dim,1,x.shape[dim]-1)
        y0=y.narrow(dim,0,y.shape[dim]-1)
        y1=y.narrow(dim,1,y.shape[dim]-1)
        terms.append(((y1-y0).abs()-(x1-x0).abs()).square().mean())
    return sum(terms)/2


def exposure_loss(y,target=0.6):
    """Paper Eq. (5): absolute 16x16 patch-mean exposure deviation."""
    p=min(16,y.shape[-2],y.shape[-1])
    local=F.avg_pool2d(y.mean(dim=1,keepdim=True),p)
    # Published Zero-DCE Eq. (5) uses absolute exposure deviation.
    # Authors' released training code instead uses a squared deviation.
    return (local-target).abs().mean()


def adaptive_exposure_target(x):
    """EXTENSION: tone-aware per-patch exposure to protect existing highlights.

    Uses only the input image (never a paired target); this is not paper Eq. (5).
    """
    p=min(16,x.shape[-2],x.shape[-1])
    local=F.avg_pool2d(x.mean(1,keepdim=True),p)
    return (0.60-0.35*(local-0.30).relu()).clamp(0.42,0.60)

def color_loss(y):
    """Published Eq. (6): sum of squared differences of channel means.

    Note: the authors' released code instead takes a norm of the squared
    channel differences. This implementation deliberately follows the paper.
    """
    ch=y.mean(dim=(2,3))
    return ((ch[:,0]-ch[:,1]).square() +
            (ch[:,0]-ch[:,2]).square() +
            (ch[:,1]-ch[:,2]).square()).mean()


def curve_tv_loss(curves):
    """Paper Eq. (7): sum_c (|horizontal gradient|+|vertical gradient|)^2.

    Average over eight enhancement maps, images, and spatial positions, while
    summing the three RGB channels as in the mathematical expression. The two
    gradients share an interior pixel, with no cyclic boundary artifacts.
    """
    batch,channels,h,w=curves.shape
    if channels!=24 or min(h,w)<2:
        raise ValueError('Expected 8 RGB curve maps and spatial dimensions >=2')
    a=curves.reshape(batch,8,3,h,w)
    dh=(a[...,1:,:-1]-a[...,:-1,:-1]).abs()
    dw=(a[...,:-1,1:]-a[...,:-1,:-1]).abs()
    return (dh+dw).square().sum(dim=2).mean()

def loss_parts(x,y,curves,ablations=(),extension='none'):
    """Baseline is the four paper losses; optional extension changes exposure."""
    if extension not in ('none','adaptive-exposure'):
        raise ValueError('Unknown extension: '+str(extension))
    exposure_target=adaptive_exposure_target(x) if extension=='adaptive-exposure' else .6
    terms=dict(spatial=spatial_loss(x,y),exposure=exposure_loss(y,exposure_target),
               color=color_loss(y),smooth=curve_tv_loss(curves))
    # Published Eq. (8) coefficients. The authors' public training code
    # has different internal losses and scalings; paper-equation reproduction
    # must not be misrepresented as numeric parity with those released weights.
    weights=dict(spatial=1.,exposure=1.,color=.5,smooth=20.)
    total=sum(weights[k]*v for k,v in terms.items() if k not in ablations)
    return total,terms

def synth_image(seed,sz=96):
    r=random.Random(seed)
    img=Image.new('RGB',(sz,sz),(r.randrange(60,220),r.randrange(60,220),r.randrange(60,220)))
    d=ImageDraw.Draw(img)
    for _ in range(12):
        a=r.randrange(sz);b=r.randrange(sz);c=r.randrange(12,sz//2)
        xy=(a-c,b-c,a+c,b+c);col=tuple(r.randrange(20,255) for i in range(3))
        if r.random()<.5: d.ellipse(xy,fill=col)
        else: d.rectangle(xy,fill=col)
    return img


class LowLightDataset(Dataset):
    """Image folder or CIFAR-10 (unpaired), plus dependency-free procedural smoke-test mode."""
    def __init__(self,mode='synthetic',folder='',size=64,count=1500,seed=7):
        self.mode=mode;self.folder=folder;self.size=size;self.count=count;self.seed=seed
        if mode=='folder':
            self.paths=[p for p in Path(folder).rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')]
            if not self.paths: raise RuntimeError('No images found in --folder')
        if mode=='cifar10':
            from torchvision.datasets import CIFAR10
            self.ds=CIFAR10(root='./data',train=True,download=True)
    def __len__(self):
        if self.mode=='folder':return len(self.paths)
        if self.mode=='cifar10':return min(self.count,len(self.ds))
        return self.count
    def __getitem__(self,index):
        if self.mode=='folder':
            with Image.open(self.paths[index]) as im: img=im.convert('RGB').copy()
        elif self.mode=='cifar10':img=self.ds[index][0].convert('RGB')
        else:img=synth_image(index+self.seed,size_or_default(self.size))
        img=ImageOps.fit(img,(self.size,self.size),method=Image.Resampling.BICUBIC)
        x=np.asarray(img,dtype=np.float32)/255.0
        # Deterministic darkness for validation and variable on-the-fly exposure in training.
        if self.mode!='folder':
            # Procedural / CIFAR toy benchmark ONLY; real paper mode must preserve
            # the actual low-light image intensities rather than darken them again.
            rng=random.Random(self.seed+int(index)*1009)
            gamma=1.4+1.5*rng.random();scale=0.35+0.4*rng.random()
            x=np.clip((x**gamma)*scale,0.,1.)
        return torch.from_numpy(x.copy()).permute(2,0,1)


def size_or_default(sz):return max(sz,32)


def as_pil(x):
    t=x.detach().cpu().float().clamp(0,1)
    return Image.fromarray((t.permute(1,2,0).numpy()*255).astype('uint8'))


def save_comparison(x,y,path):
    a=as_pil(x);b=as_pil(y);g=Image.fromarray(np.asarray(a)).convert('RGB')
    gamma=np.asarray(g).astype('float32')/255.
    gamma=Image.fromarray(np.uint8(255*np.clip(gamma**0.5,0,1)))
    w,h=a.size;out=Image.new('RGB',(w*3,h+24),'white')
    d=ImageDraw.Draw(out)
    for idx,(im,label) in enumerate([(a,'Input'),(gamma,'Gamma baseline'),(b,'Zero-DCE')]):
        out.paste(im,(idx*w,24));d.text((idx*w+3,5),label,fill='black')
    out.save(path)


def quality_stats(x):
    arr=np.asarray(as_pil(x),dtype=np.float32)/255.
    luminance=arr.mean(2)
    return {'brightness':float(luminance.mean()),'contrast_std':float(luminance.std()),'clipped_high_fraction':float((arr>.98).mean())}


def run(args):
    seed_all(args.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    payload=torch.load(args.checkpoint,map_location=device,weights_only=True) if args.checkpoint else None
    effective_width=payload.get('width',args.width) if isinstance(payload,dict) else args.width
    model=DCENet(width=effective_width).to(device)
    if payload is not None:
        model.load_state_dict(payload['model'] if isinstance(payload,dict) and 'model' in payload else payload)
    if args.image:
        with Image.open(args.image) as im: rgb=im.convert('RGB')
        rgb.thumbnail((args.max_side,args.max_side),Image.Resampling.LANCZOS)
        x=torch.tensor(np.asarray(rgb,dtype=np.float32).copy()/255.,device=device).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():y,_=model.eval()(x)
        as_pil(y[0]).save(out/'enhanced.png');save_comparison(x[0],y[0],out/'comparison.png')
        print('Saved',out/'enhanced.png');return
    trainset=LowLightDataset(mode=args.dataset,folder=args.folder,size=args.size,count=args.max_images,seed=args.seed)
    dl=DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=0,drop_last=False)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    iterator=iter(dl);ablations=set(args.ablate.split(',')) if args.ablate else set()
    if not ablations.issubset({'spatial','exposure','color','smooth'}):
        raise ValueError('Unrecognized ablation term(s): '+str(ablations))
    history=[];model.train()
    for step in range(1,args.steps+1):
        try:x=next(iterator)
        except StopIteration:iterator=iter(dl);x=next(iterator)
        x=x.to(device)
        y,curves=model(x);loss,parts=loss_parts(x,y,curves,ablations,args.extension)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        history.append([step,float(loss.detach()),*[float(parts[n].detach()) for n in ('spatial','exposure','color','smooth')]])
        if step==1 or step%max(1,args.steps//10)==0:
            print(f'step {step}/{args.steps}: loss={float(loss.detach()):.4f}')
    with (out/'loss.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['step','loss','spatial','exposure','color','smooth']);w.writerows(history)
    model.eval()
    with torch.no_grad():
        sample=trainset[0].unsqueeze(0).to(device);result,_=model(sample)
    save_comparison(sample[0],result[0],out/'comparison.png')
    stats={k:{'input':v,'enhanced':quality_stats(result[0])[k]} for k,v in quality_stats(sample[0]).items()}
    import json
    (out/'metrics.json').write_text(json.dumps(stats,indent=2))
    torch.save({'model':model.state_dict(),'width':effective_width,'steps':args.steps,'extension':args.extension,
                'seed':args.seed,'dataset':args.dataset,'size':args.size},out/'model.pt')
    print('Saved model, loss.csv, comparison.png, metrics.json to',out,'device',device)


def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['synthetic','cifar10','folder'],default='synthetic')
    p.add_argument('--folder',default='');p.add_argument('--size',type=int,default=64)
    p.add_argument('--steps',type=int,default=400);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-images',type=int,default=1500);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--width',type=int,default=32);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',default='outputs');p.add_argument('--checkpoint',default='')
    p.add_argument('--image',default='');p.add_argument('--max-side',type=int,default=1024);p.add_argument('--ablate',default='',help='comma-separated spatial,exposure,color,smooth')
    p.add_argument('--extension',choices=['none','adaptive-exposure'],default='none',help='Keep none for paper objective; adaptive-exposure is an independently configurable exposure extension')
    return p.parse_args(argv)

if __name__=='__main__':run(parse_args())
