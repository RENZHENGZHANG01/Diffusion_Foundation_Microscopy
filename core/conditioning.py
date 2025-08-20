"""
DiT-Compatible Conditioning System
=================================
OminiControl-style conditioning adapted for DiT-XL/8 microscopy models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image, ImageFilter
import cv2


class Condition:
    """Condition container similar to OminiControl"""
    
    def __init__(
        self, 
        condition_image: Union[torch.Tensor, Image.Image, np.ndarray],
        condition_type: str,
        strength: float = 1.0,
        position_delta: List[int] = None,
        position_scale: float = 1.0
    ):
        self.condition_image = self._process_condition_image(condition_image)
        self.condition_type = condition_type
        self.strength = strength
        self.position_delta = position_delta or [0, 0]
        self.position_scale = position_scale
    
    def _process_condition_image(self, image):
        """Process condition image to tensor format"""
        if isinstance(image, Image.Image):
            # Convert PIL to tensor
            image = np.array(image)
            if image.ndim == 2:  # Grayscale
                image = image[..., None]
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)  # HWC -> CHW
        
        elif isinstance(image, np.ndarray):
            # Convert numpy to tensor
            if image.ndim == 2:
                image = image[..., None]
            image = torch.from_numpy(image).float()
            if image.max() > 1.0:
                image = image / 255.0
            image = image.permute(2, 0, 1)
        
        elif isinstance(image, torch.Tensor):
            # Ensure proper format
            if image.ndim == 2:
                image = image.unsqueeze(0)
            if image.max() > 1.0:
                image = image.float() / 255.0
        
        return image


def convert_to_condition(condition_type: str, image: Union[Image.Image, np.ndarray], param: float = None):
    """Convert image to condition hint (OminiControl style)"""
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if condition_type == "super_resolution":
        # Create low-resolution version
        h, w = image.shape[:2]
        scale_factor = param if param else 4
        new_h, new_w = h // scale_factor, w // scale_factor
        
        # Downsample then upsample
        downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        condition = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)
        
    elif condition_type == "denoising":
        # Add noise
        noise_std = param if param else 25.0
        noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
        condition = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    elif condition_type == "canny":
        # Edge detection
        threshold1 = param if param else 100
        threshold2 = threshold1 * 2
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        condition = cv2.Canny(gray, threshold1, threshold2)
        condition = np.stack([condition] * (3 if image.ndim == 3 else 1), axis=-1)
        
    elif condition_type == "depth":
        # Simple depth estimation (placeholder)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        # Simple gradient-based depth
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        condition = np.sqrt(grad_x**2 + grad_y**2)
        condition = (condition / condition.max() * 255).astype(np.uint8)
        condition = np.stack([condition] * (3 if image.ndim == 3 else 1), axis=-1)
        
    else:
        raise ValueError(f"Unknown condition type: {condition_type}")
    
    return condition


class DiTConditionEncoder(nn.Module):
    """DiT-compatible condition encoder"""
    
    def __init__(self, condition_types: List[str], hidden_size: int = 1152):
        super().__init__()
        self.condition_types = condition_types
        self.hidden_size = hidden_size
        
        # Individual encoders for each condition type
        self.encoders = nn.ModuleDict()
        
        for condition_type in condition_types:
            if condition_type in ["super_resolution", "denoising"]:
                # Image-based conditions
                self.encoders[condition_type] = self._create_image_encoder()
            elif condition_type in ["canny", "depth"]:
                # Edge/structure conditions
                self.encoders[condition_type] = self._create_edge_encoder()
        
        # Condition fusion
        if len(condition_types) > 1:
            self.fusion = nn.Sequential(
                nn.Linear(len(condition_types) * hidden_size, hidden_size * 2),
                nn.SiLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size)
            )
        else:
            self.fusion = nn.Identity()
    
    def _create_image_encoder(self):
        """Create encoder for image-based conditions"""
        return nn.Sequential(
            # Initial conv layers
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            # Downsampling blocks
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            
            # Global pooling and projection
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
    
    def _create_edge_encoder(self):
        """Create encoder for edge/structure conditions"""
        return nn.Sequential(
            # Edge-specific processing
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            # Global features
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(256 * 16, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
    
    def forward(self, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode multiple conditions"""
        
        condition_embeddings = []
        
        for condition_type in self.condition_types:
            if condition_type in conditions:
                condition_input = conditions[condition_type]
                
                # Ensure proper shape [B, 1, H, W] for grayscale
                if condition_input.dim() == 3:
                    condition_input = condition_input.unsqueeze(1)
                elif condition_input.shape[1] == 3:
                    # Convert RGB to grayscale
                    condition_input = condition_input.mean(dim=1, keepdim=True)
                
                # Encode condition
                encoder = self.encoders[condition_type]
                embedding = encoder(condition_input)
                condition_embeddings.append(embedding)
            else:
                # Zero embedding for missing condition
                zero_embedding = torch.zeros(
                    conditions[list(conditions.keys())[0]].shape[0], 
                    self.hidden_size,
                    device=conditions[list(conditions.keys())[0]].device
                )
                condition_embeddings.append(zero_embedding)
        
        # Fuse conditions
        if len(condition_embeddings) > 1:
            fused = torch.cat(condition_embeddings, dim=1)
            return self.fusion(fused)
        else:
            return condition_embeddings[0]


class DiTConditionInjector(nn.Module):
    """Inject conditions into DiT transformer blocks"""
    
    def __init__(self, hidden_size: int = 1152):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Condition modulation (similar to AdaLN)
        self.condition_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2)  # scale + shift
        )
    
    def forward(self, x: torch.Tensor, condition_emb: torch.Tensor) -> torch.Tensor:
        """Apply condition to transformer features"""
        
        # Get scale and shift parameters
        condition_params = self.condition_mlp(condition_emb)  # [B, 2*hidden_size]
        scale, shift = condition_params.chunk(2, dim=1)  # [B, hidden_size] each
        
        # Apply modulation (similar to AdaLN)
        # x: [B, N, hidden_size], scale/shift: [B, hidden_size]
        scale = scale.unsqueeze(1)  # [B, 1, hidden_size]
        shift = shift.unsqueeze(1)  # [B, 1, hidden_size]
        
        return x * (1 + scale) + shift


def create_microscopy_conditions(
    foundation_image: torch.Tensor,
    condition_types: List[str],
    noise_std: float = 25.0,
    downsample_factor: int = 4
) -> Dict[str, Condition]:
    """Create microscopy-specific conditions from foundation image"""
    
    conditions = {}
    
    # Convert tensor to PIL for processing
    if isinstance(foundation_image, torch.Tensor):
        if foundation_image.dim() == 4:
            foundation_image = foundation_image[0]  # Take first batch
        if foundation_image.dim() == 3:
            foundation_image = foundation_image.permute(1, 2, 0)  # CHW -> HWC
        
        foundation_np = (foundation_image.cpu().numpy() * 255).astype(np.uint8)
        if foundation_np.shape[-1] == 1:
            foundation_np = foundation_np.squeeze(-1)
    
    for condition_type in condition_types:
        if condition_type == "super_resolution":
            condition_image = convert_to_condition("super_resolution", foundation_np, downsample_factor)
            conditions[condition_type] = Condition(condition_image, condition_type, strength=1.0)
        
        elif condition_type == "denoising":
            condition_image = convert_to_condition("denoising", foundation_np, noise_std)
            conditions[condition_type] = Condition(condition_image, condition_type, strength=1.0)
        
        elif condition_type == "canny":
            condition_image = convert_to_condition("canny", foundation_np, 100)
            conditions[condition_type] = Condition(condition_image, condition_type, strength=0.8)
        
        elif condition_type == "depth":
            condition_image = convert_to_condition("depth", foundation_np)
            conditions[condition_type] = Condition(condition_image, condition_type, strength=0.6)
    
    return conditions