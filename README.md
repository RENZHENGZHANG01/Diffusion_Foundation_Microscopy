[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/)

# Microscopy Diffusion Prior

Automated degraded condition training pipeline for microscopy diffusion models using DiT-XL/8 architecture.

## Features

- **DiT-XL/8 Architecture**: Optimized for 512×512 microscopy images
- **EDM Scheduler**: Elucidating the Design Space of Diffusion Models (Karras et al.)
- **OminiControl-Style Conditioning**: Multi-condition hint-based control
- **Automated Phase Training**: Phase 1 (unconditional) → Phase 2 (conditional)
- **Task-Agnostic Approach**: Pure hint-based control (no task labels)
- **Real-Time Monitoring**: Live loss plots, sample generation, metrics tracking
- **EMA Training**: Exponential moving average for stable convergence
- **Smart Checkpoint Management**: Custom epoch-based checkpointing
- **State Persistence**: Full resume capability
- **Lightning Integration**: Professional training with DDP, mixed precision
- **Comprehensive Sampling System**: Standalone inference with DDPM, DDIM, and EDM Euler support
- **Conditional Generation**: Full support for Phase 2 models with condition hints

## Usage

### 1. Test Setup
```bash
python test_setup.py
```

### 2. Train Model
```bash
python train.py --config config/microscopy.yaml
```

### 3. Resume Training
```bash
python train.py --config config/microscopy.yaml --resume
```

### 4. Generate Samples
```bash
# Basic sampling
python sample.py --config config/sampling.yaml

# Conditional sampling
python sample.py --config config/sampling.yaml --conditional --condition_types super_resolution,denoising

# Custom parameters
python sample.py --config config/sampling.yaml --num_samples 8 --method ddim --steps 25
```

## Training Phases

### Phase 1: Unconditional Foundation
- **Objective**: Learn general microscopy image distribution
- **Data**: 46K+ foundation samples (SR-Caco2 + FMD)
- **Architecture**: DiT-XL/8 (no conditioning)
- **Epochs**: 50 (configurable)

### Phase 2: OminiControl-Style Conditional Control
- **Objective**: Add multi-condition control capabilities
- **Data**: Foundation images + generated condition hints
- **Architecture**: DiT-XL/8 + OminiControl-style condition encoders
- **Conditions**: Super-resolution, denoising, canny edges, depth estimation
- **Base Weights**: Auto-loaded from Phase 1 best checkpoint
- **Epochs**: 30 (configurable)

## Sampling System

The project includes a comprehensive sampling system for generating samples from trained models:

### Sampling Methods
- **DDPM**: Standard denoising diffusion probabilistic models
- **DDIM**: Denoising diffusion implicit models (faster, deterministic)
- **EDM Euler**: Elucidating the Design Space of Diffusion Models

### Features
- **Standalone Inference**: Generate samples independently from training
- **Conditional Sampling**: Support for Phase 2 models with condition hints
- **Batch Generation**: Efficient large-scale sampling
- **VAE Integration**: Automatic latent space handling
- **Flexible Configuration**: YAML-based parameter configuration

### Configuration
Edit `config/sampling.yaml` to customize:
- Sampling method and steps
- Output image size and format
- Conditioning parameters
- Device and performance settings

## Data Format

Expects processed microscopy data with manifest structure:
- Foundation: High-quality images
- Degraded: Corresponding low-res/noisy versions
- Manifests: CSV files with metadata

## Data

The dataset used for training can be found at the following link:
[Microscopy Dataset](https://drive.google.com/drive/folders/1t-8SP_YxnNr99ELzuNWtRAws_CNlwPK0?usp=sharing)

## Citation

### DiT

```bibtex
@article{peebles2023scalable,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  journal={arXiv preprint arXiv:2212.09748},
  year={2023}
}
```

### OminiControl

```bibtex
@misc{tan2024ominicontrol,
      title={OminiControl: Minimal and Universal Control for Diffusion Transformer}, 
      author={Zhenxiong Tan and Songhua Liu and Xingyi Yang and Qiaochu Xue and Xinchao Wang},
      year={2024},
      eprint={2411.15098},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```