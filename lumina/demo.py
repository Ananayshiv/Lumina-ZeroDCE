"""Creative photo editor: trained Zero-DCE enhancement + reversible strength and white balance.

All manual controls are postprocessing (not learned), clearly separated from the model.
Requires pip install gradio, trained checkpoint.
"""
import argparse
import numpy as np
from PIL import Image
import torch
from lumina.train import DCENet


def edit_image(image, model, device, strength=1., warmth=0., max_side=1400):
    if image is None:return None,None
    original=image.convert('RGB').copy();original.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    a=np.asarray(original,dtype=np.float32).copy()/255
    x=torch.from_numpy(a).permute(2,0,1)[None].to(device)
    with torch.inference_mode(): enhanced,_=model(x)
    enhanced=enhanced[0].cpu().permute(1,2,0).numpy()
    # Exposure blend: strength=0 is exactly the source photo.
    output=np.clip(a*(1-float(strength))+enhanced*float(strength),0,1)
    # Manual warm/cool balance, intentionally separate from learned Zero-DCE.
    w=float(warmth)/100
    output=np.clip(output*np.array([1+.22*w,1.,1-.22*w])[None,None,:],0,1)
    return original,Image.fromarray(np.uint8(output*255))


def create_app(checkpoint='outputs/model.pt'):
    import gradio as gr
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved=torch.load(checkpoint,map_location=device,weights_only=True)
    model=DCENet(width=saved.get('width',32)).to(device)
    model.load_state_dict(saved['model']);model.eval()
    def enhance(image,strength,warmth):
        return edit_image(image,model,device,strength,warmth)
    return gr.Interface(enhance,
      inputs=[gr.Image(type='pil',label='Low-light source photo'),
              gr.Slider(0,1,value=1,step=.05,label='Zero-DCE enhancement strength'),
              gr.Slider(-100,100,value=0,step=5,label='Manual temperature (cool → warm)')],
      outputs=[gr.Image(type='pil',label='Original'),gr.Image(type='pil',label='Enhanced / downloadable')],
      title='Lumina · Controllable Low-Light Photo Editor',
      description='From-scratch Zero-DCE research prototype. White balance and strength controls are manual postprocessing. Model quality depends on training.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='outputs/model.pt')
    create_app(p.parse_args().checkpoint).launch(share=False)
