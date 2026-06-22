# SHINE-PPG: Non-Lambertian Intrinsic Decomposition for Illumination-Robust rPPG [ECCV 26]

**SHINE-PPG** is a deep learning framework for remote photoplethysmography (rPPG) that estimates heart rate from facial videos via **intrinsic image decomposition**. It disentangles a face video into illumination, reflectance, and specular components, then extracts the rPPG signal from the reflectance branch — making the signal estimation robust to lighting variation and specular highlights.

---

## Method Overview

SHINE-PPG decomposes each video frame according to the Lambertian model:

```
I = L · R + H
```

where **I** is the observed image, **L** is illumination, **R** is reflectance, and **S** is the specular component. The rPPG signal is estimated from **R**, which is lighting-invariant by design.

### Architecture

The model consists of three branches, each built from shared 3D convolutional encoder–decoder components:

| Branch | Model | Role |
|--------|-------|------|
| Illumination | `IlluminationModel` | Predicts per-frame illumination map **L** |
| Specular | `IlluminationModel` | Predicts specular highlight map **H** |
| Reflectance | `ReflectanceModel` | Predicts reflectance map **R** and rPPG signal |

---

## Training Stages

Training proceeds in four progressive stages:

| Stage | Epochs | Active Branches | Purpose |
|-------|--------|-----------------|---------|
| 1 | 1 – 20 | Illumination + Reflectance | Lambertian Initialization |
| 2 | 21 – 40 | Specular only | Specular Isolation |
| 3 | 41 – 60 | All three branches | Joint Refinement |
| 4 | 61 – 100 | All + AdaIN adversary | Adversarial Enhancement |

In Stage 4, an AdaIN-based perturbation is applied to the predicted illumination map to generate out-of-distribution (OOD) samples. The adversary maximizes rPPG loss to expose hard cases; the main network is then trained to be robust against them.

---

## Loss Functions

| Loss | Description |
|------|-------------|
| `Rec Loss` | Log-space Lambertian reconstruction: `log (I - H) − log L − log R` |
| `Reflect Loss` | Augmentation consistency: reflectance of two augmented versions should match |
| `Freq Loss` | Spatial frequency: R should be low-frequency, L should be high-frequency |
| `rPPG Loss` | Negative Pearson correlation between predicted and ground-truth signal |
| `Specular Loss` | KL divergence + color consistency + L1 sparsity on the specular map |
| `Adv Loss` | rPPG loss on AdaIN-augmented samples (Stage 4 only) |

---

## File Structure

```
SHINE-PPG/
├── __init__.py       # Main Model class — training logic and 4-stage scheduling
├── module.py         # IlluminationModel and ReflectanceModel
├── component.py      # LowLevelEncoder, HighLevelEncoder, Decoder, Estimator
├── loss.py           # All loss functions
├── adain.py          # Adaptive Instance Normalization for adversarial augmentation
├── template.py       # ModelTemplate base class — train/inference modes
└── params.yaml       # Default hyperparameters
```

---

## Configuration

Default parameters in `params.yaml`:

```yaml
size: 64        # Spatial crop size
epoch: 100      # Total training epochs
max_gpu_mem: 41 # GPU memory limit (GB)
augmentation:
  enable: True
modalities: ["rgb_face", "gt"]
```

---

## Dependencies

- Python 3.10+
- PyTorch
- einops
- thop
- matplotlib
- numpy

---

## Citation

If you use this code, please cite the corresponding paper.
