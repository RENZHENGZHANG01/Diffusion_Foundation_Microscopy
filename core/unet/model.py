"""
Lightning wrapper MicroscopyUnetModel mirroring MicroscopyDiTModel API, using EDM scheduler.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import lightning as L

from ..edm_scheduler import create_edm_scheduler, EDMPreconditioner
from .unet import Unet


class MicroscopyUnetModel(L.LightningModule):
    def __init__(self, config: Dict, phase_config: Dict):
        super().__init__()
        self.config = config
        self.phase_config = phase_config
        self.save_hyperparameters()

        # Build backbone
        model_cfg = self.config.get('model', {})
        channels = 1 if not self.config['vae'].get('enabled', False) else model_cfg.get('latent_channels', 4)
        dim = int(model_cfg.get('unet_base_channels', 64))
        dim_mults = tuple(model_cfg.get('unet_channel_mult', [1, 2, 4, 8, 8]))
        time_dim = int(model_cfg.get('unet_time_dim', 256))
        self.model = Unet(
            dim=dim,
            dim_mults=dim_mults,
            channels=channels,
            out_dim=channels,
            time_embedding_dim=time_dim,
        )

        # EDM scheduler + preconditioner
        self.scheduler = create_edm_scheduler(self.config)
        self.edm_preconditioner = EDMPreconditioner(
            model=self.model,
            sigma_data=self.config.get('scheduler', {}).get('sigma_data', 0.4),
            prediction_type=self.config.get('scheduler', {}).get('prediction_type', 'sample')
        )
        self.use_edm = True

        # VAE (optional, reuse logic from DiT model if needed later)
        self.vae = None
        if self.config.get('vae', {}).get('enabled', False):
            try:
                from diffusers.models import AutoencoderKL
                self.vae = AutoencoderKL.from_pretrained(self.config['vae']['model_id'])
                self.vae.requires_grad_(False)
                self.vae.eval()
            except Exception:
                self.vae = None

        self.automatic_optimization = True

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x, t, y)

    def encode_conditions(self, batch) -> Optional[torch.Tensor]:
        """Placeholder to mirror MicroscopyDiTModel API.
        UNet path currently trains unconditional; return None to satisfy evaluator.
        """
        return None

    def _compute_edm_loss(self, latents: torch.Tensor) -> torch.Tensor:
        b = latents.shape[0]
        device = latents.device
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (b,), device=device)
        noise = torch.randn_like(latents)
        x_t = self.scheduler.add_noise(latents, noise, timesteps)
        sigma_vals = self.scheduler.sigmas[timesteps.detach().to('cpu')].to(device)
        x0_hat = self.edm_preconditioner(x_t, sigma_vals)
        target = latents
        weights = self.scheduler.get_loss_weights(timesteps)
        loss = torch.nn.functional.mse_loss(x0_hat, target, reduction='none')
        while len(weights.shape) < len(loss.shape):
            weights = weights.unsqueeze(-1)
        return (loss * weights).mean()

    def training_step(self, batch, batch_idx):
        images = batch['image']
        latents = images
        if latents.shape[1] == 3:
            latents = latents[:, :1]
        loss = self._compute_edm_loss(latents)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        latents = images
        if latents.shape[1] == 3:
            latents = latents[:, :1]
        loss = self._compute_edm_loss(latents)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt_cfg = self.config['optimizer']
        opt = torch.optim.AdamW(self.parameters(), lr=opt_cfg['lr'], weight_decay=opt_cfg.get('weight_decay', 0.01))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs, eta_min=opt_cfg.get('min_lr', 1e-6))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


