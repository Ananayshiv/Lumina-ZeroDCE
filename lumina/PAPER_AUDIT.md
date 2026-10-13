# Paper fidelity audit — Zero-DCE (CVPR 2020)

**Official publication:** https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html

## Implemented directly from the paper (not weights copied from an upstream repo)

- Seven convolutional layers, 32 feature maps, symmetric concatenation, ReLU, final tanh; 24 parameter maps / eight RGB iterations; default **79,416 learnable parameters**. Checked in a unit test.
- Equation (3) recurrence: `x_next = x + A*x*(1-x)` with each `A` in [-1,1]. No downsampling or batch norm.
- Equation (4) spatial consistency: patch-mean **absolute-neighbor-difference preservation**, 4x4 spatial regions and no cyclic wrapping.
- Equation (5) exposure: mean **absolute** 16x16 brightness-patch deviation from `E=0.6`.
- Equation (6) color: sum of **squared** deviations between RGB channel means.
- Equation (7) illumination-map smoothing: squared sum of absolute horizontal and vertical curve-map gradients, averaged over eight stages.
- Paper-equation training weights `1 (spatial)`, `1 (exposure)`, `0.5 (color)`, `20 (smooth)`.

## Crucial source-code discrepancy / reproducibility note

The authors' **released PyTorch implementation** uses `A*(x*x-x)` (equivalent after `A -> -A`), *squared* exposure loss instead of printed Eq. (5) absolute deviation, fourth-order **color constancy** instead of printed Eq. (6) squared differences, and different normalizations/weights. Our default follows the **published equations**, rather than claiming byte-for-byte numerical parity with the authors' public training code. This is an algorithmic **paper-equation reproduction**, NOT a benchmark reproduction nor reimplementation of their released code's exact objective. See authors' original code: https://github.com/Li-Chongyi/Zero-DCE/tree/master/Zero-DCE_code

## Extension (clearly optional)

`--extension adaptive-exposure`: adjust the per-patch target based on original illumination, with bright regions receiving a lower target. No paired targets are used. The original paper baseline is `--extension none`. Gradio's learned-enhancement-strength control and nonlearned color-temperature slider are presentation/product controls, not part of the published method.

## Limits

- Paper's real-image datasets and original evaluation tables not reproduced here.
- Synthetic paired PSNR only verifies the custom benchmark, not LOL test-set fidelity.
- Parameter-count, algebra, finite gradients and executable training are locally tested; **real photo quality requires full Colab training and review**.

## V6 real-pair and confidence-interval evaluation

`evaluate_advanced.py` supports an aligned dark/reference folder pair with strict filename and dimension matching, optionally downscaling both images identically. It reports per-image model, input and gamma PSNR and MAE, clipped highlights and the paired-bootstrap 95% interval of `model_psnr - gamma_psnr`. A CI over image examples is not a CI over independent training runs; it does not prove restoration quality on unrelated real datasets. The `folder` train loader does *not* artificially darken already-dark images.
