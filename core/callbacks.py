"""
Custom Callbacks for Microscopy Diffusion Training
==================================================
EMA, real-time monitoring, and sample generation callbacks
"""

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import io
import base64
from PIL import Image
import time


class EMACallback(Callback):
    """Exponential Moving Average callback for stable training"""
    
    def __init__(
        self,
        decay: float = 0.9999,
        update_every: int = 1,
        start_step: int = 2000
    ):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.start_step = start_step
        self.ema_model = None
        self.step_count = 0
    
    def on_train_start(self, trainer, pl_module):
        """Initialize EMA model"""
        self.ema_model = self._create_ema_copy(pl_module)
        print(f"[EMA] Initialized with decay={self.decay}, start_step={self.start_step}")
    
    def _create_ema_copy(self, model):
        """Create EMA copy of the model"""
        ema_model = type(model)(model.config, model.phase_config)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        
        # Move to same device as original model
        ema_model = ema_model.to(model.device)
        
        # Disable gradients for EMA model
        for param in ema_model.parameters():
            param.requires_grad = False
        
        return ema_model
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA after each batch"""
        self.step_count += 1
        
        if self.step_count < self.start_step:
            return
        
        if self.step_count % self.update_every == 0:
            self._update_ema(pl_module)
    
    def _update_ema(self, model):
        """Update EMA model parameters"""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), 
                model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1 - self.decay
                )
    
    def on_validation_start(self, trainer, pl_module):
        """Switch to EMA model for validation"""
        if self.ema_model is not None and self.step_count >= self.start_step:
            # Store original state
            self._original_state = pl_module.state_dict()
            # Load EMA state
            pl_module.load_state_dict(self.ema_model.state_dict())
    
    def on_validation_end(self, trainer, pl_module):
        """Restore original model after validation"""
        if hasattr(self, '_original_state'):
            pl_module.load_state_dict(self._original_state)
            delattr(self, '_original_state')


class RealTimeMonitorCallback(Callback):
    """Real-time monitoring with plots and metrics"""
    
    def __init__(
        self,
        log_every_n_steps: int = 10,
        plot_loss: bool = True,
        plot_lr: bool = True,
        save_plots: bool = True,
        plot_dir: str = "training_plots",
        track_throughput: bool = True,
        track_gpu_memory: bool = True,
        track_grad_norm: bool = True,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.plot_loss = plot_loss
        self.plot_lr = plot_lr
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        self.track_throughput = track_throughput
        self.track_gpu_memory = track_gpu_memory
        self.track_grad_norm = track_grad_norm
        
        # Tracking lists
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        self.steps = []
        self.step_times = []
        self.throughputs = []
        self.gpu_mem_mb = []
        self.grad_norms = []
        
        if self.save_plots:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
        self._last_step_start_time = None
    
    def _log_scalar(self, trainer, tag: str, value: float, step: int):
        try:
            for lg in (trainer.loggers if isinstance(trainer.loggers, list) else [trainer.logger]):
                if isinstance(lg, TensorBoardLogger):
                    lg.experiment.add_scalar(tag, value, step)
        except Exception:
            pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._last_step_start_time = time.time()

    def on_train_start(self, trainer, pl_module):
        """Initialize monitoring"""
        print(f"[MONITOR] Real-time monitoring started")
        print(f"[MONITOR] Logging every {self.log_every_n_steps} steps")
        if self.save_plots:
            print(f"[MONITOR] Plots will be saved to: {self.plot_dir}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log training metrics"""
        if trainer.global_step % self.log_every_n_steps == 0:
            
            # Get current loss
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss'].item()
            elif hasattr(outputs, 'item'):
                loss = outputs.item()
            else:
                loss = float(outputs)
            
            # Get learning rate
            optimizer = trainer.optimizers[0]
            lr = optimizer.param_groups[0]['lr']
            
            # Step time and throughput
            step_time = None
            throughput = None
            if self._last_step_start_time is not None:
                step_time = max(1e-8, time.time() - self._last_step_start_time)
            batch_size = None
            try:
                if isinstance(batch, dict) and 'image' in batch and hasattr(batch['image'], 'shape'):
                    batch_size = int(batch['image'].shape[0])
                elif isinstance(batch, (list, tuple)) and len(batch) > 0 and hasattr(batch[0], 'shape'):
                    batch_size = int(batch[0].shape[0])
            except Exception:
                batch_size = None
            if step_time is not None and batch_size is not None:
                throughput = batch_size / step_time

            # GPU memory (MB)
            gpu_mem = None
            if torch.cuda.is_available():
                try:
                    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                except Exception:
                    gpu_mem = None

            # Gradient norm
            grad_norm = None
            try:
                total_sq = 0.0
                for p in pl_module.parameters():
                    if p.grad is not None:
                        g = p.grad.detach()
                        total_sq += float(torch.sum(g * g).item())
                grad_norm = float(total_sq ** 0.5)
            except Exception:
                grad_norm = None

            # Store metrics
            self.train_losses.append(loss)
            self.learning_rates.append(lr)
            self.steps.append(trainer.global_step)
            if step_time is not None:
                self.step_times.append(step_time)
            if throughput is not None and self.track_throughput:
                self.throughputs.append(throughput)
                self._log_scalar(trainer, 'train/throughput_images_per_sec', throughput, trainer.global_step)
            if gpu_mem is not None and self.track_gpu_memory:
                self.gpu_mem_mb.append(gpu_mem)
                self._log_scalar(trainer, 'system/gpu_mem_mb', gpu_mem, trainer.global_step)
            if grad_norm is not None and self.track_grad_norm:
                self.grad_norms.append(grad_norm)
                self._log_scalar(trainer, 'train/grad_norm', grad_norm, trainer.global_step)
            
            # Log to console
            extra = []
            if throughput is not None and self.track_throughput:
                extra.append(f"TPS: {throughput:.1f} img/s")
            if gpu_mem is not None and self.track_gpu_memory:
                extra.append(f"GPU: {gpu_mem:.0f} MB")
            if grad_norm is not None and self.track_grad_norm:
                extra.append(f"|g|: {grad_norm:.2f}")
            extra_str = (" | " + " | ".join(extra)) if extra else ""
            print(f"[STEP {trainer.global_step:06d}] "
                  f"Loss: {loss:.6f} | "
                  f"LR: {lr:.2e} | "
                  f"Epoch: {trainer.current_epoch}{extra_str}")
            
            # Update plots
            if len(self.train_losses) > 10:  # Only plot after some data
                self._update_plots(trainer.current_epoch, trainer.global_step)
    
    def on_validation_end(self, trainer, pl_module):
        """Log validation metrics"""
        # Get validation loss
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
            self.epochs.append(trainer.current_epoch)
            
            print(f"[VAL] Epoch {trainer.current_epoch} | Val Loss: {val_loss:.6f}")
    
    def _update_plots(self, current_epoch, current_step):
        """Update real-time plots"""
        if not (self.plot_loss or self.plot_lr):
            return
        
        try:
            # Create subplots
            n_plots = 0
            order = []
            if self.plot_loss:
                n_plots += 1
                order.append('loss')
            if self.plot_lr:
                n_plots += 1
                order.append('lr')
            if self.track_throughput and len(self.throughputs) > 0:
                n_plots += 1
                order.append('throughput')
            if self.track_gpu_memory and len(self.gpu_mem_mb) > 0:
                n_plots += 1
                order.append('gpu')
            if self.track_grad_norm and len(self.grad_norms) > 0:
                n_plots += 1
                order.append('grad')
            fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 4))
            if n_plots == 1:
                axes = [axes]
            
            for plot_idx, kind in enumerate(order):
                ax = axes[plot_idx]
                ax.clear()
                if kind == 'loss':
                    ax.plot(self.steps, self.train_losses, 'b-', alpha=0.7, label='Train Loss')
                    if self.val_losses and self.epochs:
                        val_steps = [e * len(self.steps) // max(self.epochs) for e in self.epochs]
                        ax.plot(val_steps, self.val_losses, 'r-', marker='o', label='Val Loss')
                    ax.set_xlabel('Steps')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'Training Progress (Epoch {current_epoch})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                elif kind == 'lr':
                    ax.plot(self.steps, self.learning_rates, 'g-', alpha=0.7)
                    ax.set_xlabel('Steps')
                    ax.set_ylabel('Learning Rate')
                    ax.set_title('Learning Rate Schedule')
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                elif kind == 'throughput':
                    ax.plot(self.steps[-len(self.throughputs):], self.throughputs, 'm-', alpha=0.7)
                    ax.set_xlabel('Steps')
                    ax.set_ylabel('Images / sec')
                    ax.set_title('Throughput')
                    ax.grid(True, alpha=0.3)
                elif kind == 'gpu':
                    ax.plot(self.steps[-len(self.gpu_mem_mb):], self.gpu_mem_mb, 'c-', alpha=0.7)
                    ax.set_xlabel('Steps')
                    ax.set_ylabel('GPU MB')
                    ax.set_title('GPU Memory (allocated)')
                    ax.grid(True, alpha=0.3)
                elif kind == 'grad':
                    ax.plot(self.steps[-len(self.grad_norms):], self.grad_norms, 'y-', alpha=0.7)
                    ax.set_xlabel('Steps')
                    ax.set_ylabel('||grad||_2')
                    ax.set_title('Gradient Norm')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if self.save_plots:
                plot_path = self.plot_dir / f"training_progress_step_{current_step:06d}.png"
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                
                # Also save latest
                latest_path = self.plot_dir / "latest_progress.png"
                plt.savefig(latest_path, dpi=100, bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            print(f"[WARNING] Plot update failed: {e}")


class SampleGenerationCallback(Callback):
    """Generate samples during training for monitoring"""
    
    def __init__(
        self,
        sample_every_n_epochs: int = 5,
        num_samples: int = 4,
        save_dir: str = "training_samples"
    ):
        super().__init__()
        self.sample_every_n_epochs = sample_every_n_epochs
        self.num_samples = num_samples
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Generate samples at specified intervals"""
        if trainer.current_epoch % self.sample_every_n_epochs == 0:
            self._generate_samples(trainer, pl_module)
    
    def _generate_samples(self, trainer, pl_module):
        """Generate and save samples"""
        try:
            pl_module.eval()
            
            with torch.no_grad():
                # Create random inputs in the correct space
                device = next(pl_module.parameters()).device
                if getattr(pl_module, 'vae', None) is not None:
                    # Latent space sampling
                    ch = getattr(pl_module.model, 'in_channels', 4)
                    # Infer spatial size from patch embedder
                    try:
                        num_patches = pl_module.model.x_embedder.num_patches
                        p = pl_module.model.x_embedder.patch_size[0]
                        h = w = int(num_patches ** 0.5) * p
                        # For typical SD VAE latents, h=w=64 with p=8 and 8x8 patches
                    except Exception:
                        h = w = 64
                    latents = torch.randn(self.num_samples, ch, h, w, device=device)
                else:
                    # Pixel space sampling (single-channel)
                    # Use model embedder resolution
                    try:
                        num_patches = pl_module.model.x_embedder.num_patches
                        p = pl_module.model.x_embedder.patch_size[0]
                        h = w = int(num_patches ** 0.5) * p
                    except Exception:
                        h = w = 512
                    latents = torch.randn(self.num_samples, 1, h, w, device=device)
                
                # Generate timesteps (for visualization, use middle timestep)
                t = torch.full((self.num_samples,), 500, device=device)
                
                # Forward pass (this is just for monitoring, not full generation)
                if hasattr(pl_module, 'model'):
                    # Try simple forward pass with dummy labels
                    y = torch.zeros(latents.shape[0], dtype=torch.long, device=device)
                    outputs = pl_module.model(latents, t, y)
                    
                    # Convert to images (simplified)
                    images = torch.sigmoid(outputs)  # Normalize to [0,1]
                    images = images.cpu()
                    
                    # Save samples
                    self._save_sample_grid(images, trainer.current_epoch)
                    
                    print(f"[SAMPLES] Generated {self.num_samples} samples at epoch {trainer.current_epoch}")
                
        except Exception as e:
            print(f"[WARNING] Sample generation failed: {e}")
        
        finally:
            pl_module.train()
    
    def _save_sample_grid(self, images, epoch):
        """Save sample grid"""
        try:
            # Create grid
            grid_size = int(np.ceil(np.sqrt(self.num_samples)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
            
            for i in range(self.num_samples):
                row = i // grid_size
                col = i % grid_size
                
                if grid_size == 1:
                    ax = axes
                else:
                    ax = axes[row, col] if grid_size > 1 else axes[col]
                
                # Convert tensor to image
                if images[i].shape[0] == 4:  # 4 channels
                    # Take first channel or average
                    img = images[i][0]
                elif images[i].shape[0] == 1:  # 1 channel
                    img = images[i][0]
                else:
                    img = images[i].mean(0)  # Average channels
                
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                ax.set_title(f'Sample {i+1}')
            
            # Hide empty subplots
            for i in range(self.num_samples, grid_size * grid_size):
                row = i // grid_size
                col = i % grid_size
                ax = axes[row, col] if grid_size > 1 else axes[col]
                ax.axis('off')
            
            plt.suptitle(f'Generated Samples - Epoch {epoch}')
            plt.tight_layout()
            
            # Save
            save_path = self.save_dir / f"samples_epoch_{epoch:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"[WARNING] Sample grid saving failed: {e}")


class ImageLoggingCallback(Callback):
    """Log input and condition images periodically to disk and TensorBoard"""
    def __init__(
        self,
        log_every_n_steps: int = 200,
        max_images: int = 8,
        save_dir: str = "training_images",
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_images = max_images
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.log_every_n_steps != 0:
            return
        try:
            if not isinstance(batch, dict) or 'image' not in batch:
                return
            images = batch['image']
            if not isinstance(images, torch.Tensor):
                return
            images = images.detach().cpu()
            num = min(self.max_images, images.shape[0])
            imgs = images[:num]

            # Convert to HWC grid
            grid_rows = int(np.ceil(np.sqrt(num)))
            grid_cols = int(np.ceil(num / grid_rows))
            c, h, w = imgs.shape[1:]
            if c == 1:
                imgs_disp = imgs.repeat(1, 3, 1, 1)  # make 3-channel for visualization
            elif c >= 3:
                imgs_disp = imgs[:, :3]
            else:
                imgs_disp = imgs.repeat(1, 3 // c + 1, 1, 1)[:, :3]
            grid = np.zeros((grid_rows * h, grid_cols * w, 3), dtype=np.float32)
            for i in range(num):
                r = i // grid_cols
                col = i % grid_cols
                img = imgs_disp[i].permute(1, 2, 0).numpy()
                grid[r*h:(r+1)*h, col*w:(col+1)*w, :] = img

            # Save PNG
            save_path = self.save_dir / f"train_batch_step_{trainer.global_step:06d}.png"
            plt.figure(figsize=(10, 10))
            plt.imshow(np.clip(grid, 0, 1))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()

            # Log to TensorBoard
            try:
                for lg in (trainer.loggers if isinstance(trainer.loggers, list) else [trainer.logger]):
                    if isinstance(lg, TensorBoardLogger):
                        lg.experiment.add_image('train/input_grid', grid, trainer.global_step, dataformats='HWC')
                        break
            except Exception:
                pass

            # Also try to log first available condition image
            cond_keys = [k for k in batch.keys() if k.startswith('condition_') and isinstance(batch[k], torch.Tensor)]
            if cond_keys:
                cond = batch[cond_keys[0]].detach().cpu()
                cond = cond[:num]
                if cond.dim() == 3:
                    cond = cond.unsqueeze(1)
                ch = cond.shape[1]
                if ch == 1:
                    cond_disp = cond.repeat(1, 3, 1, 1)
                else:
                    cond_disp = cond[:, :3]
                grid_c = np.zeros((grid_rows * h, grid_cols * w, 3), dtype=np.float32)
                for i in range(min(num, cond_disp.shape[0])):
                    r = i // grid_cols
                    col = i % grid_cols
                    img = cond_disp[i].permute(1, 2, 0).numpy()
                    grid_c[r*h:(r+1)*h, col*w:(col+1)*w, :] = img
                cpath = self.save_dir / f"condition_batch_step_{trainer.global_step:06d}.png"
                plt.figure(figsize=(10, 10))
                plt.imshow(np.clip(grid_c, 0, 1))
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(cpath, dpi=120, bbox_inches='tight')
                plt.close()
                try:
                    for lg in (trainer.loggers if isinstance(trainer.loggers, list) else [trainer.logger]):
                        if isinstance(lg, TensorBoardLogger):
                            lg.experiment.add_image('train/condition_grid', grid_c, trainer.global_step, dataformats='HWC')
                            break
                except Exception:
                    pass

            print(f"[IMAGES] Logged input images at step {trainer.global_step}")
        except Exception as e:
            print(f"[WARNING] Image logging failed: {e}")


class CustomCheckpointCallback(Callback):
    """Custom checkpoint callback with epochs_to_save"""
    
    def __init__(self, save_dir: str, epochs_to_save: List[int], phase_name: str):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.epochs_to_save = set(epochs_to_save)
        self.phase_name = phase_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[CHECKPOINT] Will save at epochs: {sorted(self.epochs_to_save)}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Save checkpoint at specified epochs"""
        current_epoch = trainer.current_epoch
        
        if current_epoch in self.epochs_to_save:
            self._save_checkpoint(trainer, pl_module, current_epoch)
    
    def _save_checkpoint(self, trainer, pl_module, epoch):
        """Save model checkpoint"""
        try:
            checkpoint_path = self.save_dir / f"{self.phase_name}_epoch_{epoch:03d}.ckpt"
            
            # Save Lightning checkpoint
            trainer.save_checkpoint(checkpoint_path)
            
            print(f"[CHECKPOINT] Saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"[ERROR] Checkpoint save failed: {e}")