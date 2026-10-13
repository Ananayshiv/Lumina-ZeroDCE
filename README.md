# Lumina — Zero-Reference Low-Light Enhancement

From-scratch PyTorch implementation of **Zero-DCE** (CVPR 2020): seven convolutional layers, eight pixelwise curve updates, and four non-reference loss families. The independent adaptive-exposure extension adjusts local exposure targets rather than modifying the network architecture.

## Measured results — 64 held-out procedural scenes
| Variant | PSNR (dB) ↑ | MAE ↓ | Reported clipped-pixel metric ↓ |
|---|---:|---:|---:|
| Unenhanced input | 8.455 | 0.3586 | — |
| Gamma correction | 15.188 | 0.1624 | — |
| Zero-DCE baseline (1,200 steps) | 16.176 | 0.1330 | 0.1673 |
| Adaptive exposure (1,200 steps) | **16.304** | **0.1308** | **0.1364** |

The +0.129 dB adaptive-vs-baseline improvement is **preliminary**. Training and evaluation here used procedurally generated, synthetically darkened images, **not** LOL test images or official paper benchmarks. Neither a multi-seed improvement nor real-photograph generalization has been established.

## Method
Eight iterations use `I_(k+1) = I_k + A_k I_k (1 - I_k)` with learned RGB-valued maps and spatial, exposure, color-consistency, and smoothness losses. Exactly black/clipped pixels lack recoverable intensity information. The implementation follows the published mathematical loss expressions where parts of the released reference code differ.

## Reproduce
```bash
pip install -r requirements.txt
python -m lumina.train --dataset synthetic --size 64 --steps 1200 --batch-size 8 --seed 42 --extension none --out runs/baseline
python -m lumina.train --dataset synthetic --size 64 --steps 1200 --batch-size 8 --seed 42 --extension adaptive-exposure --out runs/adaptive
python -m lumina.evaluate_advanced --checkpoint runs/adaptive/model.pt --n 64 --size 64 --out runs/adaptive_eval
python -m pytest -q
```

Trained checkpoints and evaluation artifacts were generated in Colab and retained in Google Drive. They are intentionally excluded from Git history until published through an appropriate release or external artifact store. Do not mistake a smoke test for a trained research benchmark.

**Paper:** Guo et al., [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html).
