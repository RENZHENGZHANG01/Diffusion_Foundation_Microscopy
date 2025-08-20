"""Auto Degradation Utilities - Physically-motivated fluorescence microscopy degradation"""
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict
import re

# Constants and lookup tables
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC

DYE_EMISSION_LOOKUP = {
    'DAPI': 461, 'NUCLEI': 461, 'HOECHST': 461, 'GFP': 509, 'EGFP': 509, 'E-GFP': 509, 'YFP': 527,
    'E-CADHERIN': 509, 'TUBULIN': 509, 'SURVIVIN': 520, 'F-ACTIN': 520, 'PHALLOIDIN': 520, 'FITC': 520,
    'ALEXA 488': 520, 'ALEXA488': 520, 'TRITC': 570, 'CY3': 570, 'AF555': 565, 'ALEXA 555': 565,
    'AF568': 603, 'ALEXA 568': 603, 'MITO': 600, 'MITOCHONDRIA': 600, 'MITOTRACKER RED': 599,
    'MITOTRACKER': 599, 'TEXAS RED': 615, 'AF594': 617, 'MCHERRY': 610, 'CHERRY': 610, 'H2B': 610,
    'AF647': 668, 'ALEXA 647': 668, 'CY5': 670, 'DEEP RED': 665, 'DEEPRED': 665, 'FISH': 520,
}

RXRX1_CHANNEL_DYE_MAP = {
    1: {'emission_nm': 461}, 2: {'emission_nm': 520}, 3: {'emission_nm': 520}, 
    4: {'emission_nm': 585}, 5: {'emission_nm': 603}, 6: {'emission_nm': 665}
}

PSF_FWHM_FACTORS = {'WideField': 0.51, 'Confocal': 0.37, 'TwoPhoton': 0.37}

def _gaussian_kernel1d(sigma_px: float, radius_mult: float = 3.0) -> np.ndarray:
    """Generate 1D Gaussian kernel for PSF modeling."""
    if sigma_px <= 0: return np.array([1.0], dtype=np.float32)
    r = int(np.ceil(radius_mult * sigma_px))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma_px) ** 2)
    return (k / k.sum()).astype(np.float32)

def _separable_conv2d(img: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """Apply separable 2D convolution."""
    if k1d.size == 1: return img.astype(np.float32, copy=True)
    r = (k1d.size - 1) // 2
    # Vertical pass
    ypad = np.pad(img, ((r, r), (0, 0)), mode='reflect')
    tmp = sum(w * ypad[i:i + img.shape[0], :] for i, w in enumerate(k1d))
    # Horizontal pass
    xpad = np.pad(tmp, ((0, 0), (r, r)), mode='reflect')
    return sum(w * xpad[:, i:i + img.shape[1]] for i, w in enumerate(k1d))

def _block_reduce_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """Simulate pixel binning by block averaging."""
    if factor <= 1: return img
    H, W = img.shape
    h, w = (H // factor) * factor, (W // factor) * factor
    cropped = img[:h, :w]
    return cropped.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3)).astype(img.dtype)

def _radial_vignette(h: int, w: int, strength: float) -> np.ndarray:
    """Generate radial vignetting pattern."""
    if strength <= 0: return np.ones((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ry, rx = (yy - cy) / max(cy, 1.0), (xx - cx) / max(cx, 1.0)
    return np.clip(1.0 - strength * (rx * rx + ry * ry), 0.0, 1.0).astype(np.float32)

def _lowfreq_field(h: int, w: int, rng, sigma_px: float = 64, strength: float = 0.2) -> np.ndarray:
    """Generate low-frequency flat-field illumination variation."""
    base = rng.normal(1.0, 0.05, size=(h, w)).astype(np.float32)
    field = _separable_conv2d(base, _gaussian_kernel1d(sigma_px))
    field = field / np.mean(field)
    return np.clip(1.0 + strength * (field - 1.0), 0.5, 1.5).astype(np.float32)

def resolve_emission_wavelength(metadata: Dict) -> float:
    """Resolve emission wavelength from manifest metadata."""
    # Direct lambda specification
    if 'lambda' in metadata and pd.notna(metadata['lambda']):
        try: return float(metadata['lambda'])
        except (ValueError, TypeError): pass
    
    # RxRx1 channel mapping
    if metadata.get('dataset', '').startswith('rxrx1'):
        try:
            ch_idx = metadata.get('channel_idx')
            if ch_idx is not None:
                ch_int = int(ch_idx) if not isinstance(ch_idx, int) else ch_idx
                key = ch_int + 1 if 0 <= ch_int < 6 else ch_int
                if key in RXRX1_CHANNEL_DYE_MAP:
                    return float(RXRX1_CHANNEL_DYE_MAP[key]['emission_nm'])
        except (ValueError, TypeError): pass
    
    # Dye lookup from marker
    if 'marker' in metadata:
        marker_raw = str(metadata['marker']).upper()
        marker_norm = re.sub(r'[^A-Z0-9]+', ' ', marker_raw).strip()
        marker_compact = re.sub(r'[^A-Z0-9]+', '', marker_raw)
        
        for key, val in DYE_EMISSION_LOOKUP.items():
            if not isinstance(val, (int, float)): continue
            k_norm = re.sub(r'[^A-Z0-9]+', ' ', key).strip()
            k_compact = re.sub(r'[^A-Z0-9]+', '', key)
            if k_norm and (k_norm in marker_norm or k_compact in marker_compact):
                return float(val)
    
    return 520.0  # Default fallback

def degrade_microscopy_physical(arr: np.ndarray, sample_metadata: Dict, 
                               degradation_mode: str = 'blur', **kwargs) -> Tuple[np.ndarray, str]:
    """Physically-motivated fluorescence microscopy degradation."""
    # Prepare image: convert to 2D grayscale and normalize
    x = arr[..., 0].astype(np.float32, copy=True) if arr.ndim == 3 else arr.astype(np.float32, copy=True)
    if x.size and (x.max() > 1.0 or x.min() < 0.0):
        p995 = np.percentile(x, 99.5)
        scale = p995 if p995 > 0 else x.max()
        if scale and np.isfinite(scale): x = np.clip(x / scale, 0.0, 1.0)
    
    tag = []
    
    # Resolve optical parameters from metadata with safe conversions
    emission_nm = resolve_emission_wavelength(sample_metadata)
    NA_val = 1.0
    try: NA_val = float(sample_metadata['NA']) if 'NA' in sample_metadata and pd.notna(sample_metadata['NA']) else 1.0
    except (ValueError, TypeError): pass
    
    um_per_px = 0.15
    try: um_per_px = float(sample_metadata['um_pixel']) if 'um_pixel' in sample_metadata and pd.notna(sample_metadata['um_pixel']) else 0.15
    except (ValueError, TypeError): pass
    
    # Calculate PSF parameters
    fwhm_factor = PSF_FWHM_FACTORS.get(sample_metadata.get('modality', 'WideField'), 0.51)
    lam_um = emission_nm * 1e-3
    sigma_um = (fwhm_factor * lam_um / max(NA_val, 1e-6)) / 2.355
    base_psf_sigma_px = sigma_um / max(um_per_px, 1e-12)
    
    # Apply degradation based on mode
    if degradation_mode == 'blur':
        psf_scale = float(kwargs.get('psf_scale', 1.0))
        extra_sigma_px = float(kwargs.get('extra_sigma_px', 0.0))
        psf_sigma_px = np.hypot(base_psf_sigma_px * psf_scale, extra_sigma_px)
        k = _gaussian_kernel1d(psf_sigma_px, kwargs.get('radius_mult', 3.0))
        x = _separable_conv2d(x, k)
        fwhm_px = 2.355 * psf_sigma_px
        scale_info = f"×{psf_scale:.1f}" if psf_scale != 1.0 else ""
        extra_info = f"+{extra_sigma_px:.1f}" if extra_sigma_px > 0 else ""
        tag.append(f"psf_blur(sig={psf_sigma_px:.2f}px,FWHM={fwhm_px:.2f}px{scale_info}{extra_info})")
        
    elif degradation_mode == 'noise':
        rng = np.random.default_rng(kwargs.get('rng_seed', None))
        
        # Adaptive electron scaling based on image brightness
        electrons_at_one = float(kwargs.get('electrons_at_one', 1500.0))
        target_snr = kwargs.get('target_snr_at_full_scale', None)
        if target_snr is not None:
            base_electrons = float(target_snr) ** 2
            image_mean = x.mean()
            boost_factor = 3.0 if image_mean > 0.3 else (8.0 if image_mean > 0.1 else 20.0)
            electrons_at_one = base_electrons * boost_factor
        
        read_noise_e = float(kwargs.get('read_noise_e', 1.5))
        bias_e = float(kwargs.get('bias_e', 100.0))
        dark_e = float(kwargs.get('dark_current_e', 0.0))
        
        # Camera simulation: signal -> electrons -> noise -> ADC
        exp_e = np.clip(x, 0.0, None) * electrons_at_one + dark_e + bias_e
        shot = rng.poisson(exp_e).astype(np.float32)
        noisy_e = shot + rng.normal(0.0, read_noise_e, size=shot.shape).astype(np.float32)
        noisy_e = np.clip(noisy_e - bias_e, 0.0, None)
        
        # ADC quantization
        bit_depth = int(kwargs.get('bit_depth', 16))
        gain_e_per_adu = float(kwargs.get('gain_e_per_adu', 0.2))
        max_adu = (1 << bit_depth) - 1
        adu = np.clip(noisy_e / gain_e_per_adu, 0, max_adu).round()
        x = adu / max_adu
        
        snr_at_1 = np.sqrt(electrons_at_one)
        tag_parts = [f"SNR1={snr_at_1:.0f}", f"read={read_noise_e:.1f}e-"]
        if bit_depth != 16: tag_parts.append(f"bit={bit_depth}")
        if gain_e_per_adu != 1.0: tag_parts.append(f"gain={gain_e_per_adu:.1f}")
        tag.append(f"noise({','.join(tag_parts)})")
        
    elif degradation_mode == 'downsample':
        bin_factor = kwargs.get('factor', 4)
        upsample_back = kwargs.get('upsample_back', True)
        
        reduced = _block_reduce_mean(x, bin_factor)
        if upsample_back:
            H, W = x.shape
            upsampled = np.repeat(np.repeat(reduced, bin_factor, axis=0), bin_factor, axis=1)
            if upsampled.shape != (H, W):
                result = np.zeros((H, W), dtype=upsampled.dtype)
                h_min, w_min = min(H, upsampled.shape[0]), min(W, upsampled.shape[1])
                result[:h_min, :w_min] = upsampled[:h_min, :w_min]
                x = result
            else:
                x = upsampled
            tag.append(f"bin{bin_factor}x_upsampled")
        else:
            x = reduced
            tag.append(f"bin{bin_factor}x")
    
    # Apply illumination effects with adaptive scaling
    v_strength = float(kwargs.get('vignetting_strength', 0.0))
    ff_strength = float(kwargs.get('flatfield_strength', 0.0))
    
    if v_strength > 0 or ff_strength > 0:
        image_mean = x.mean()
        illum_scale = 0.5 if image_mean > 0.3 else (0.8 if image_mean > 0.1 else 1.0)
        v_strength *= illum_scale
        ff_strength *= illum_scale
        h, w = x.shape
        rng = np.random.default_rng(kwargs.get('rng_seed', None))
        illum = np.ones_like(x)
        
        if v_strength > 0: illum *= _radial_vignette(h, w, v_strength)
        if ff_strength > 0:
            ff_sigma = kwargs.get('flatfield_sigma_px', 64)
            illum *= _lowfreq_field(h, w, rng, sigma_px=ff_sigma, strength=ff_strength)
        
        x = np.clip(x * illum, 0.0, 1.0)
        illum_parts = []
        if v_strength > 0: illum_parts.append(f"vig={v_strength:.2f}")
        if ff_strength > 0: illum_parts.append(f"ff={ff_strength:.2f}")
        tag.append(f"illum({','.join(illum_parts)})")
    
    return np.clip(x, 0.0, 1.0), "+".join(tag)

# Main interface functions
def create_auto_degraded_blur(image: np.ndarray, sample_metadata: Dict, **kwargs) -> Tuple[np.ndarray, str]:
    """Create physically-motivated blur degradation."""
    return degrade_microscopy_physical(image, sample_metadata, 'blur', **kwargs)

def create_auto_degraded_noise(image: np.ndarray, sample_metadata: Dict, **kwargs) -> Tuple[np.ndarray, str]:
    """Create physically-motivated noise degradation."""
    default_kwargs = {'read_noise_e': 1.5, 'electrons_at_one': 1500.0}
    default_kwargs.update(kwargs)
    return degrade_microscopy_physical(image, sample_metadata, 'noise', **default_kwargs)

def create_auto_degraded_downsample(image: np.ndarray, sample_metadata: Dict, **kwargs) -> Tuple[np.ndarray, str]:
    """Create physically-motivated downsample degradation."""
    default_kwargs = {'factor': 4, 'upsample_back': True}
    default_kwargs.update(kwargs)
    return degrade_microscopy_physical(image, sample_metadata, 'downsample', **default_kwargs)

def auto_degrade_with_metadata(image: np.ndarray, sample_metadata: Dict, mode: str = 'blur', **kwargs) -> Tuple[np.ndarray, str]:
    """Apply realistic auto-degradation using sample metadata from manifest."""
    return degrade_microscopy_physical(image, sample_metadata, mode, **kwargs)

def save_image_array(image: np.ndarray, output_path: str, bit_depth: int = 16):
    """Save image array as PNG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if image.ndim == 3: image = image[..., 0]
    image = np.clip(image, 0.0, 1.0)
    
    if bit_depth == 16:
        image_16 = (image * 65535 + 0.5).astype(np.uint16)
        Image.fromarray(image_16, mode='I;16').save(output_path, compress_level=4)
    else:
        image_8 = (image * 255 + 0.5).astype(np.uint8)
        Image.fromarray(image_8, mode='L').save(output_path, compress_level=4)