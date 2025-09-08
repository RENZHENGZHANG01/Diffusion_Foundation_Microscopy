"""
Centralized Sampling System for Microscopy Diffusion Models
==========================================================
Extracts and unifies sampling logic from EDM scheduler and Gaussian diffusion.
Supports DDPM, DDIM, and EDM Euler sampling with full conditioning support.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Local imports
from .models import MicroscopyDiTModel
from .diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from .edm_scheduler import EDMEulerScheduler, EDMPreconditioner
from .conditioning import DiTConditionEncoder, DiTConditionInjector, create_microscopy_conditions
from diffusers.models import AutoencoderKL


class MicroscopySampler:
    """
    Centralized sampling system for microscopy diffusion models.
    Supports multiple sampling methods with full conditioning capabilities.
    """
    
    def __init__(self, config_path: str):
        """Initialize sampler with configuration"""
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.model = None
        self.vae = None
        self.diffusion = None
        self.scheduler = None
        self.condition_encoder = None
        self.condition_injector = None
        
        # Setup components
        self._setup_model()
        self._setup_vae()
        self._setup_sampling()
        self._setup_conditioning()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load sampling configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        device_config = self.config.get('device', {})
        device_str = device_config.get('device', 'auto')
        
        if device_str == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_str)
        
        print(f"[SAMPLER] Using device: {device}")
        return device
    
    def _setup_model(self):
        """Load trained model from checkpoint"""
        checkpoint_path = self.config['model']['checkpoint_path']
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[SAMPLER] Loading model from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model configuration from checkpoint
        if 'hyper_parameters' in checkpoint:
            model_config = checkpoint['hyper_parameters']['config']
            phase_config = checkpoint['hyper_parameters']['phase_config']
        else:
            # Fallback: use default configs
            model_config = self._get_default_model_config()
            phase_config = self._get_default_phase_config()
        
        # Create model
        self.model = MicroscopyDiTModel(model_config, phase_config)
        
        # Load state dict
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[SAMPLER] Model loaded successfully")
    
    def _setup_vae(self):
        """Setup VAE for latent space decoding"""
        vae_config = self.config.get('vae', {})
        
        if not vae_config.get('enabled', True):
            print("[SAMPLER] VAE disabled - using pixel space")
            self.vae = None
            return
        
        try:
            model_id = vae_config.get('model_id', 'stabilityai/stable-diffusion-2-1')
            self.vae = AutoencoderKL.from_pretrained(model_id)
            self.vae.requires_grad_(False)
            self.vae.to(self.device)
            self.vae.eval()
            print(f"[SAMPLER] VAE loaded: {model_id}")
        except Exception as e:
            print(f"[SAMPLER] VAE loading failed: {e}")
            print("[SAMPLER] Using pixel space fallback")
            self.vae = None
    
    def _setup_sampling(self):
        """Setup sampling method based on configuration"""
        sampling_config = self.config['sampling']
        method = sampling_config['method']
        
        if method == 'edm_euler':
            self._setup_edm_sampling()
        else:
            self._setup_diffusion_sampling()
        
        print(f"[SAMPLER] Sampling method: {method}")
    
    def _setup_edm_sampling(self):
        """Setup EDM Euler sampling"""
        sampling_config = self.config['sampling']
        edm_config = sampling_config.get('edm', {})
        
        # Create EDM scheduler
        self.scheduler = EDMEulerScheduler(
            num_train_timesteps=1000,
            sigma_min=edm_config.get('sigma_min', 0.002),
            sigma_max=edm_config.get('sigma_max', 80.0),
            rho=edm_config.get('rho', 7.0)
        )
        
        # Setup preconditioner if conditional
        if self.config['model']['phase'] == 'conditional':
            self.edm_preconditioner = EDMPreconditioner(
                model=self.model.model,
                sigma_data=0.5,
                prediction_type='v_prediction'
            )
        
        self.use_edm = True
    
    def _setup_diffusion_sampling(self):
        """Setup standard diffusion sampling"""
        # Create diffusion process
        betas = self._get_beta_schedule('linear', 1000)
        
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        
        self.use_edm = False
    
    def _setup_conditioning(self):
        """Setup conditioning for Phase 2 models"""
        if self.config['model']['phase'] != 'conditional':
            return
        
        conditioning_config = self.config.get('conditioning', {})
        if not conditioning_config.get('enabled', True):
            return
        
        condition_types = conditioning_config.get('condition_types', [])
        
        if condition_types:
            # Setup condition encoder
            self.condition_encoder = DiTConditionEncoder(
                condition_types=condition_types,
                hidden_size=768  # DiT-XL hidden size
            )
            
            # Setup condition injector
            self.condition_injector = DiTConditionInjector(hidden_size=768)
            
            print(f"[SAMPLER] Conditioning enabled: {condition_types}")
    
    def generate_samples(
        self,
        num_samples: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
        condition_hints: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Generate samples using the configured sampling method.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Output image size (H, W)
            seed: Random seed for reproducibility
            condition_hints: Optional condition hints for conditional sampling
            
        Returns:
            Generated samples as tensor
        """
        # Use config defaults if not provided
        if num_samples is None:
            num_samples = self.config['output']['num_samples']
        if image_size is None:
            image_size = tuple(self.config['output']['image_size'])
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"[SAMPLER] Generating {num_samples} samples of size {image_size}")
        
        # Determine latent shape
        if self.vae is not None:
            # VAE latent space: typically 4x64x64 for SD VAEs
            latent_size = image_size[0] // 8  # VAE downsampling factor
            shape = (num_samples, 4, latent_size, latent_size)
        else:
            # Pixel space: single channel microscopy
            shape = (num_samples, 1, image_size[0], image_size[1])
        
        # Generate samples
        if self.use_edm:
            samples = self._generate_edm_samples(shape, condition_hints)
        else:
            samples = self._generate_diffusion_samples(shape, condition_hints)
        
        # Decode from latent space if using VAE
        if self.vae is not None:
            samples = self._decode_vae_samples(samples)
        
        return samples
    
    def _generate_edm_samples(self, shape: Tuple[int, ...], condition_hints: Optional[Dict] = None) -> torch.Tensor:
        """Generate samples using EDM Euler sampling"""
        sampling_config = self.config['sampling']
        num_steps = sampling_config['num_steps']
        
        # Start with pure noise
        x = torch.randn(shape, device=self.device)
        
        # Generate timesteps
        timesteps = self.scheduler.timesteps[:num_steps]
        
        # Progress bar
        if self.config['advanced'].get('show_progress', True):
            timesteps = tqdm(timesteps, desc="EDM Sampling")
        
        # Sampling loop
        for i, t in enumerate(timesteps):
            # Create timestep tensor
            timestep = torch.tensor([t], device=self.device).repeat(shape[0])
            
            # Model prediction
            if condition_hints is not None and hasattr(self, 'condition_encoder'):
                # Conditional sampling
                condition_emb = self._encode_conditions(condition_hints)
                sigma_vals = self.scheduler.sigmas[t].to(self.device)
                model_output = self.edm_preconditioner(x, sigma_vals)
            else:
                # Unconditional sampling
                y = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
                model_output = self.model.model(x, timestep, y)
            
            # Euler step
            x = self.scheduler.step(model_output, t, x).prev_sample
        
        return x
    
    def _generate_diffusion_samples(self, shape: Tuple[int, ...], condition_hints: Optional[Dict] = None) -> torch.Tensor:
        """Generate samples using standard diffusion sampling"""
        sampling_config = self.config['sampling']
        method = sampling_config['method']
        num_steps = sampling_config['num_steps']
        
        # Create model function
        def model_fn(x, t):
            if condition_hints is not None and hasattr(self, 'condition_encoder'):
                # Conditional sampling
                condition_emb = self._encode_conditions(condition_hints)
                return self.model.forward(x, t, condition_emb)
            else:
                # Unconditional sampling
                y = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
                return self.model.model(x, t, y)
        
        # Choose sampling method
        if method == 'ddim':
            samples = self.diffusion.ddim_sample_loop(
                model_fn,
                shape,
                device=self.device,
                progress=self.config['advanced'].get('show_progress', True),
                eta=sampling_config.get('ddim_eta', 0.0)
            )
        else:  # ddpm
            samples = self.diffusion.p_sample_loop(
                model_fn,
                shape,
                device=self.device,
                progress=self.config['advanced'].get('show_progress', True)
            )
        
        return samples
    
    def _encode_conditions(self, condition_hints: Dict) -> torch.Tensor:
        """Encode condition hints for conditional sampling"""
        if not hasattr(self, 'condition_encoder'):
            return None
        
        # Create condition dictionary
        conditions = {}
        condition_types = self.config['conditioning']['condition_types']
        
        for i, condition_type in enumerate(condition_types):
            if condition_type in condition_hints:
                conditions[condition_type] = condition_hints[condition_type]
        
        if conditions:
            return self.condition_encoder(conditions)
        
        return None
    
    def _decode_vae_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Decode samples from VAE latent space to pixel space"""
        if self.vae is None:
            return samples
        
        # Decode latents
        with torch.no_grad():
            decoded = self.vae.decode(samples / self.config['vae']['scaling_factor']).sample
        
        # Convert from [-1, 1] to [0, 1]
        decoded = (decoded + 1) / 2
        
        # Convert to grayscale (take first channel)
        if decoded.shape[1] == 3:
            decoded = decoded[:, :1, :, :]
        
        return decoded
    
    def save_samples(self, samples: torch.Tensor, save_dir: str, prefix: str = "sample") -> List[str]:
        """Save generated samples to disk"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        # Convert to numpy and ensure [0, 1] range
        samples_np = samples.detach().cpu().numpy()
        samples_np = np.clip(samples_np, 0, 1)
        
        # Save individual samples
        if self.config['output'].get('save_individual', True):
            for i, sample in enumerate(samples_np):
                # Convert to PIL Image
                if sample.shape[0] == 1:  # Grayscale
                    img_array = (sample[0] * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='L')
                else:  # RGB
                    img_array = (sample.transpose(1, 2, 0) * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='RGB')
                
                # Save
                filename = f"{prefix}_{i:03d}.{self.config['output']['image_format']}"
                save_path = save_dir / filename
                img.save(save_path, dpi=(self.config['output']['dpi'], self.config['output']['dpi']))
                saved_paths.append(str(save_path))
        
        # Save sample grid
        if self.config['output'].get('save_grid', True):
            grid_path = self._save_sample_grid(samples_np, save_dir, prefix)
            if grid_path:
                saved_paths.append(grid_path)
        
        return saved_paths
    
    def _save_sample_grid(self, samples: np.ndarray, save_dir: Path, prefix: str) -> Optional[str]:
        """Save sample grid as single image"""
        try:
            num_samples = len(samples)
            grid_layout = self.config['output'].get('grid_layout', 'auto')
            
            if grid_layout == 'auto':
                grid_size = int(np.ceil(np.sqrt(num_samples)))
            else:
                grid_size = int(grid_layout.split('x')[0])
            
            # Create figure
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
            if grid_size == 1:
                axes = [axes]
            elif grid_size > 1:
                axes = axes.flatten()
            
            # Plot samples
            for i in range(num_samples):
                if i < len(axes):
                    ax = axes[i]
                    if samples[i].shape[0] == 1:  # Grayscale
                        ax.imshow(samples[i][0], cmap='gray')
                    else:  # RGB
                        ax.imshow(samples[i].transpose(1, 2, 0))
                    ax.set_title(f'Sample {i+1}')
                    ax.axis('off')
            
            # Hide empty subplots
            for i in range(num_samples, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # Save
            grid_filename = f"{prefix}_grid.{self.config['output']['image_format']}"
            grid_path = save_dir / grid_filename
            plt.savefig(grid_path, dpi=self.config['output']['dpi'], bbox_inches='tight')
            plt.close()
            
            return str(grid_path)
        
        except Exception as e:
            print(f"[SAMPLER] Grid saving failed: {e}")
            return None
    
    def _get_beta_schedule(self, schedule_name: str, num_timesteps: int) -> np.ndarray:
        """Get beta schedule for diffusion sampling"""
        if schedule_name == "linear":
            scale = 1000 / num_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        else:
            raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")
    
    def _get_default_model_config(self) -> Dict:
        """Get default model configuration"""
        return {
            'model': {
                'architecture': 'DiT-XL/8',
                'input_size': [512, 512],
                'hidden_size': 768,
                'latent_size': 64,
                'latent_channels': 4
            },
            'vae': {'enabled': True, 'model_id': 'stabilityai/stable-diffusion-2-1'},
            'optimizer': {'lr': 1e-4, 'weight_decay': 0.01}
        }
    
    def _get_default_phase_config(self) -> Dict:
        """Get default phase configuration"""
        return {
            'type': 'unconditional',
            'condition_types': [],
            'condition_dropout': 0.0
        }
    
    @classmethod
    def _create_from_module(cls, pl_module, config: Dict):
        """Create sampler instance from existing Lightning module (for callbacks)"""
        sampler = cls.__new__(cls)
        sampler.config = config
        sampler.device = next(pl_module.parameters()).device
        sampler.model = pl_module
        sampler.vae = getattr(pl_module, 'vae', None)
        sampler.diffusion = getattr(pl_module, 'diffusion', None)
        sampler.scheduler = getattr(pl_module, 'scheduler', None)
        sampler.use_edm = getattr(pl_module, 'use_edm', False)
        sampler.condition_encoder = getattr(pl_module, 'condition_encoder', None)
        sampler.condition_injector = getattr(pl_module, 'condition_injector', None)
        sampler.edm_preconditioner = getattr(pl_module, 'edm_preconditioner', None)
        return sampler


def create_condition_hints(
    image_size: Tuple[int, int],
    condition_types: List[str],
    hints_config: Dict
) -> Dict:
    """
    Create condition hints for conditional sampling.
    
    Args:
        image_size: Target image size (H, W)
        condition_types: List of condition types to generate
        hints_config: Configuration for hint generation
        
    Returns:
        Dictionary of condition hints
    """
    hints = {}
    
    for condition_type in condition_types:
        if condition_type == 'super_resolution':
            # Create low-resolution version
            scale = hints_config.get('super_resolution_scale', 2.0)
            low_res_size = (int(image_size[0] / scale), int(image_size[1] / scale))
            hints[condition_type] = torch.randn(1, 1, *low_res_size)
            
        elif condition_type == 'denoising':
            # Create noisy version
            hints[condition_type] = torch.randn(1, 1, *image_size) * hints_config.get('denoising_strength', 0.5)
            
        elif condition_type == 'canny_edge':
            # Create random edge map
            hints[condition_type] = torch.randn(1, 1, *image_size)
            
        elif condition_type == 'depth':
            # Create random depth map
            hints[condition_type] = torch.randn(1, 1, *image_size)
            
        else:
            # Generic condition
            hints[condition_type] = torch.randn(1, 1, *image_size)
    
    return hints