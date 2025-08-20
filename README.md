# Microscopy Diffusion Prior

Automated phase training pipeline for microscopy diffusion models using DiT-XL/8 architecture.

## Structure

```
diffusion-prior/
├── core/                    # Core modules (DiT-based, self-contained)
│   ├── dit_models.py       # DiT model architectures
│   ├── diffusion/          # Diffusion process utilities
│   ├── conditioning.py     # OminiControl-style conditioning system
│   ├── models.py           # Microscopy DiT models with conditioning
│   ├── datasets.py         # Data loading for foundation/degraded pairs  
│   ├── trainer.py          # Lightning training orchestrator
│   └── phase_manager.py    # Automated phase transitions
├── config/
│   └── microscopy.yaml     # Training configuration
├── train.py               # Main training script
├── test_setup.py          # Setup verification
└── README.md
```

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

## Configuration

Key settings in `config/microscopy.yaml`:

```yaml
model:
  architecture: "DiT-XL/8"
  latent_size: 64           # 512//8 for VAE
  hidden_size: 1152         # DiT-XL dimension

# EDM Scheduler configuration
scheduler:
  type: "EDMEulerScheduler"           # EDM scheduler for better sampling
  num_train_timesteps: 1000
  sigma_min: 0.002
  sigma_max: 80.0
  sigma_data: 0.5
  rho: 7.0
  prediction_type: "v_prediction"    # v-parameterization for stability

# EMA configuration
ema:
  enabled: true
  decay: 0.9999
  update_every: 1
  start_step: 2000

# Real-time monitoring
monitoring:
  log_every_n_steps: 10
  val_check_interval: 100
  plot_loss: true
  plot_lr: true
  plot_samples: true
  sample_every_n_epochs: 5

train:
  batch_size: 2             # Optimized for single GPU
  dataloader_workers: 4

phases:
  - name: "phase1_unconditional"
    epochs: 50
    epochs_to_save: [12, 25, 37, 50]    # Custom checkpoint epochs
  - name: "phase2_conditional"
    epochs: 30
    epochs_to_save: [8, 15, 22, 30]
    condition_types: ["super_resolution", "denoising", "canny", "depth"]
    condition_dropout: 0.1
```

## Requirements

- PyTorch Lightning
- Diffusers (for VAE)
- timm (for ViT components)
- Standard ML stack (torch, numpy, PIL, etc.)

## Data Format

Expects processed microscopy data with manifest structure:
- Foundation: High-quality images
- Degraded: Corresponding low-res/noisy versions
- Manifests: CSV files with metadata

## Checkpoints

- Automatically saves 4 checkpoints per phase (25%, 50%, 75%, 100%)
- Old checkpoints automatically cleaned up
- Best model selected based on validation loss
- Phase 2 auto-loads Phase 1 best weights

## Logging

- ASCII-only terminal output
- TensorBoard logs
- Optional WandB integration
- Full state tracking in JSON

## Monitoring

Training state persisted in `phase_state.json`:
- Current phase progress
- Checkpoint locations
- Training metrics
- Event history

## Real-Time Monitoring

The training pipeline provides comprehensive real-time monitoring:

### **Live Plots**
- **Loss Tracking**: Training and validation loss in real-time
- **Learning Rate**: Scheduler visualization
- **Sample Generation**: Generated samples every N epochs

### **Output Structure**
```
training_output/
├── checkpoints/           # Model checkpoints at specified epochs
│   ├── phase1_unconditional_epoch_012.ckpt
│   ├── phase1_unconditional_epoch_025.ckpt
│   └── ...
├── plots/                 # Real-time training plots
│   ├── phase1_unconditional/
│   │   ├── latest_progress.png
│   │   └── training_progress_step_*.png
│   └── phase2_conditional/
├── samples/               # Generated samples during training
│   ├── phase1_unconditional/
│   │   ├── samples_epoch_005.png
│   │   └── samples_epoch_010.png
│   └── phase2_conditional/
└── logs/                  # TensorBoard and WandB logs
```

### **EMA (Exponential Moving Average)**
- Automatically enabled for stable training
- Uses EMA weights for validation and final checkpoints
- Configurable decay rate and start step

### **Custom Checkpointing**
- Save at specific epochs defined in `epochs_to_save`
- Automatic cleanup (keeps only specified checkpoints)
- Includes both model and EMA weights

### **EDM Scheduler (Karras et al. 2022)**
- **Improved Noise Schedule**: Better sigma distribution for training
- **v-Parameterization**: More stable than epsilon prediction
- **Continuous Time**: Superior to discrete DDPM formulation
- **Preconditioning**: Optimal scaling for model inputs/outputs
- **Loss Weighting**: Theoretically motivated loss weights
- **Better Sampling**: Euler method with improved trajectories

## Customization

Easily extend by:
- Adding new condition types in `models.py`
- Modifying phase configurations and checkpoint schedules
- Custom dataset loaders in `datasets.py`
- Additional monitoring callbacks in `callbacks.py`
- Custom evaluation metrics