#!/usr/bin/env python3
"""
Microscopy Diffusion Training - DiT-XL/8 Based Pipeline
=======================================================
Automated phase training for 512x512 microscopy generation

Usage:
    python train.py --config config/microscopy.yaml
"""

import torch
import argparse
import yaml
from pathlib import Path
import sys
import csv
import json
import numpy as np

# Add repo and core module paths for absolute imports
repo_dir = Path(__file__).parent
sys.path.insert(0, str(repo_dir))
sys.path.insert(0, str(repo_dir / 'core'))
from core.trainer import MicroscopyTrainer
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
from core.callbacks import RealTimeMonitorCallback, ImageLoggingCallback
from core.models import MicroscopyDiTModel


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add run ID for tracking
    config['run_id'] = f"run_{torch.randint(1000, 9999, (1,)).item()}"
    
    return config


def main():
    """Main training entry point"""
    
    parser = argparse.ArgumentParser(description="Microscopy Diffusion Training")
    parser.add_argument('--config', type=str, required=True, help='Configuration YAML file')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--smoke-test', action='store_true', help='Run a one-image integration test and exit')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("[INIT] Microscopy Diffusion Training Pipeline")
    print(f"[CONFIG] {args.config}")
    print(f"[RUN_ID] {config['run_id']}")
    
    # Optional smoke test
    if args.smoke_test:
        run_smoke_test()
        return
    
    # Create trainer and run pipeline
    trainer = MicroscopyTrainer(config)
    summary = trainer.run_training_pipeline()
    print(f"[FINAL] Training completed: {summary}")


def run_smoke_test():
    """Run a short, single-image integration test using SR-Caco2 manifest."""
    repo = Path(__file__).parent
    root = repo.parent  # fluorescence_datasets

    manifest = root / 'processed' / 'foundation' / 'sr_caco2' / 'manifests' / 'manifest.csv'
    if not manifest.exists():
        print(f"[SMOKE] Manifest not found: {manifest}")
        return

    def map_win(p: str) -> str:
        return ('/mnt/' + p.split(':', 1)[0].lower() + p.split(':', 1)[1].replace('\\', '/')) if ':' in p else p

    # Read first row
    with open(manifest, 'r', newline='') as f:
        row = next(csv.DictReader(f))
    img_path = map_win(row['image'])
    img = Image.open(img_path).convert('L')
    x = torch.from_numpy(np.array(img)).float()
    try:
        lo = json.loads(row['norm_lo']).get('ch0_lo', 0.0)
        hi = json.loads(row['norm_hi']).get('ch0_hi', 255.0)
    except Exception:
        lo, hi = 0.0, 255.0
    x = ((x - lo) / max(1e-6, (hi - lo))).clamp(0, 1)[None, None, ...]

    # Minimal config and conditional phase
    config = {
        'model': {'architecture': 'DiT-XL/8', 'latent_size': 64, 'hidden_size': 1152},
        'vae': {'model_id': 'stabilityai/sd-vae-ft-ema'},
        'scheduler': {
            'type': 'EDMEulerScheduler', 'num_train_timesteps': 1000,
            'sigma_min': 0.002, 'sigma_max': 80.0, 'sigma_data': 0.5,
            'rho': 7.0, 'prediction_type': 'v_prediction',
        },
        'optimizer': {'lr': 1e-4, 'min_lr': 1e-6},
        'train': {'dataloader_workers': 0, 'accumulate_grad_batches': 1, 'save_path': str(repo / 'microscopy_runs')},
        'monitoring': {'log_every_n_steps': 1, 'val_check_interval': 2, 'progress_bar': True,
                       'monitor_metric': 'val_loss', 'image_log_every_n_steps': 1,
                       'image_log_max_images': 4, 'plot_loss': True, 'plot_lr': True, 'plot_samples': False},
    }
    phase = {'type': 'conditional', 'name': 'one_image_cond', 'condition_types': ['super_resolution'],
             'epochs': 1, 'batch_size': 1, 'datasets': ['sr_caco2'], 'condition_dropout': 0.0}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MicroscopyDiTModel(config, phase).to(device)

    # Build SR hint sample (no batch dim)
    low = F.interpolate(x, scale_factor=0.25, mode='area')
    low_up = F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)
    sample = {'image': x.squeeze(0), 'condition_0': low_up.squeeze(0).squeeze(0), 'condition_type_0': 'super_resolution'}

    class SingleBatchDataset(Dataset):
        def __init__(self, batch_dict, length=4): self.batch, self.length = batch_dict, length
        def __len__(self): return self.length
        def __getitem__(self, idx): return self.batch

    train_loader = DataLoader(SingleBatchDataset(sample, 3), batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(SingleBatchDataset(sample, 1), batch_size=1, shuffle=False, num_workers=0)

    save_root = Path(config['train']['save_path']) / 'one_image_test'
    (save_root / 'logs').mkdir(parents=True, exist_ok=True)
    (save_root / 'plots').mkdir(parents=True, exist_ok=True)
    (save_root / 'images').mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=str(save_root / 'logs'), name='phase_test')
    monitor = RealTimeMonitorCallback(log_every_n_steps=1, plot_loss=True, plot_lr=True,
                                      save_plots=True, plot_dir=str(save_root / 'plots'),
                                      track_throughput=True, track_gpu_memory=True, track_grad_norm=True)
    imglog = ImageLoggingCallback(log_every_n_steps=1, max_images=4, save_dir=str(save_root / 'images'))

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    precision = ('bf16-mixed' if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else ('16-mixed' if torch.cuda.is_available() else '32-true'))

    trainer = L.Trainer(max_epochs=1, limit_train_batches=3, limit_val_batches=1,
                        callbacks=[monitor, imglog], logger=logger, accelerator=accelerator,
                        devices='auto', strategy='auto', precision=precision, log_every_n_steps=1,
                        enable_progress_bar=True, enable_checkpointing=False, gradient_clip_val=1.0)

    print('[SMOKE] Starting tiny training to trigger logs...')
    trainer.fit(model, train_loader, val_loader)
    print('[SMOKE] Done. Logs at:', save_root)


if __name__ == "__main__":
    main()