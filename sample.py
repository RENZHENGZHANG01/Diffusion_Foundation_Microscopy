#!/usr/bin/env python3
"""
Standalone Inference Script for Microscopy Diffusion Models
==========================================================
Generate samples from trained checkpoints using centralized sampling system.

Usage:
    # Basic sampling
    python sample.py --config config/sampling.yaml
    
    # Custom parameters
    python sample.py --config config/sampling.yaml --num_samples 8 --method ddim --steps 25
    
    # Conditional sampling with hints
    python sample.py --config config/sampling.yaml --conditional --condition_types super_resolution,denoising
    
    # Batch generation
    python sample.py --config config/sampling.yaml --num_samples 16 --batch_size 4
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import json
from typing import List, Dict, Optional

# Add repo and core module paths
repo_dir = Path(__file__).parent
sys.path.insert(0, str(repo_dir))
sys.path.insert(0, str(repo_dir / 'core'))

from core.sampling import MicroscopySampler, create_condition_hints


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Microscopy Diffusion Sampling")
    
    # Configuration
    parser.add_argument('--config', type=str, required=True, help='Sampling configuration YAML file')
    
    # Sampling parameters
    parser.add_argument('--num_samples', type=int, help='Number of samples to generate')
    parser.add_argument('--method', type=str, choices=['ddpm', 'ddim', 'edm_euler'], help='Sampling method')
    parser.add_argument('--steps', type=int, help='Number of sampling steps')
    parser.add_argument('--guidance_scale', type=float, help='Guidance scale for conditional sampling')
    parser.add_argument('--ddim_eta', type=float, help='DDIM eta parameter')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, help='Output directory for samples')
    parser.add_argument('--image_size', type=int, nargs=2, help='Output image size (H W)')
    parser.add_argument('--prefix', type=str, default='sample', help='Filename prefix')
    
    # Conditional sampling
    parser.add_argument('--conditional', action='store_true', help='Enable conditional sampling')
    parser.add_argument('--condition_types', type=str, help='Comma-separated condition types')
    parser.add_argument('--condition_strength', type=float, help='Condition strength (0.0-1.0)')
    
    # Advanced options
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, help='Batch size for generation')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--phase', type=str, choices=['unconditional', 'conditional'], help='Model phase')
    
    return parser.parse_args()


def update_config_from_args(config: Dict, args) -> Dict:
    """Update configuration with command line arguments"""
    
    # Sampling parameters
    if args.num_samples is not None:
        config['output']['num_samples'] = args.num_samples
    if args.method is not None:
        config['sampling']['method'] = args.method
    if args.steps is not None:
        config['sampling']['num_steps'] = args.steps
    if args.guidance_scale is not None:
        config['sampling']['guidance_scale'] = args.guidance_scale
    if args.ddim_eta is not None:
        config['sampling']['ddim_eta'] = args.ddim_eta
    
    # Output parameters
    if args.output_dir is not None:
        config['output']['save_dir'] = args.output_dir
    if args.image_size is not None:
        config['output']['image_size'] = list(args.image_size)
    
    # Conditional sampling
    if args.conditional:
        config['conditioning']['enabled'] = True
    if args.condition_types is not None:
        config['conditioning']['condition_types'] = args.condition_types.split(',')
    if args.condition_strength is not None:
        config['conditioning']['condition_strength'] = args.condition_strength
    
    # Advanced options
    if args.seed is not None:
        config['advanced']['seed'] = args.seed
    if args.batch_size is not None:
        config['advanced']['batch_size'] = args.batch_size
    if args.device is not None:
        config['device']['device'] = args.device
    if args.no_progress:
        config['advanced']['show_progress'] = False
    
    # Model parameters
    if args.checkpoint is not None:
        config['model']['checkpoint_path'] = args.checkpoint
    if args.phase is not None:
        config['model']['phase'] = args.phase
    
    return config


def generate_samples_batch(
    sampler: MicroscopySampler,
    total_samples: int,
    batch_size: int,
    condition_hints: Optional[Dict] = None
) -> List[torch.Tensor]:
    """Generate samples in batches for memory efficiency"""
    
    all_samples = []
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"[BATCH] Generating {total_samples} samples in {num_batches} batches of {batch_size}")
    
    for batch_idx in range(num_batches):
        # Calculate samples for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_samples = end_idx - start_idx
        
        print(f"[BATCH] Generating batch {batch_idx + 1}/{num_batches} ({batch_samples} samples)")
        
        # Generate batch
        batch_output = sampler.generate_samples(
            num_samples=batch_samples,
            condition_hints=condition_hints
        )
        
        all_samples.append(batch_output)
    
    # Concatenate all batches
    return torch.cat(all_samples, dim=0)


def main():
    """Main sampling function"""
    args = parse_args()
    
    print("[SAMPLER] Microscopy Diffusion Sampling")
    print(f"[CONFIG] {args.config}")
    
    # Load and update configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config = update_config_from_args(config, args)
    
    # Set random seed
    if config['advanced'].get('seed') is not None:
        torch.manual_seed(config['advanced']['seed'])
        np.random.seed(config['advanced']['seed'])
        print(f"[SEED] Set to {config['advanced']['seed']}")
    
    # Initialize sampler
    try:
        sampler = MicroscopySampler(args.config)
        print("[SAMPLER] Initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize sampler: {e}")
        return 1
    
    # Prepare condition hints for conditional sampling
    condition_hints = None
    if config['conditioning'].get('enabled', False):
        condition_types = config['conditioning'].get('condition_types', [])
        if condition_types:
            image_size = tuple(config['output']['image_size'])
            hints_config = config['conditioning'].get('hints', {})
            
            condition_hints = create_condition_hints(
                image_size=image_size,
                condition_types=condition_types,
                hints_config=hints_config
            )
            
            print(f"[CONDITIONS] Generated hints for: {condition_types}")
    
    # Generate samples
    num_samples = config['output']['num_samples']
    batch_size = config['advanced'].get('batch_size', 1)
    
    try:
        if batch_size > 1 and num_samples > batch_size:
            # Batch generation for memory efficiency
            samples = generate_samples_batch(
                sampler=sampler,
                total_samples=num_samples,
                batch_size=batch_size,
                condition_hints=condition_hints
            )
        else:
            # Single batch generation
            samples = sampler.generate_samples(
                num_samples=num_samples,
                condition_hints=condition_hints
            )
        
        print(f"[SUCCESS] Generated {samples.shape[0]} samples")
        
    except Exception as e:
        print(f"[ERROR] Sample generation failed: {e}")
        return 1
    
    # Save samples
    try:
        output_dir = config['output']['save_dir']
        saved_paths = sampler.save_samples(samples, output_dir, args.prefix)
        
        print(f"[SAVE] Saved {len(saved_paths)} files to {output_dir}")
        for path in saved_paths:
            print(f"  - {path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save samples: {e}")
        return 1
    
    # Save generation info
    try:
        info_path = Path(output_dir) / f"{args.prefix}_info.json"
        generation_info = {
            'num_samples': num_samples,
            'sampling_method': config['sampling']['method'],
            'num_steps': config['sampling']['num_steps'],
            'image_size': config['output']['image_size'],
            'conditioning_enabled': config['conditioning'].get('enabled', False),
            'condition_types': config['conditioning'].get('condition_types', []),
            'seed': config['advanced'].get('seed'),
            'saved_files': saved_paths
        }
        
        with open(info_path, 'w') as f:
            json.dump(generation_info, f, indent=2)
        
        print(f"[INFO] Saved generation info to {info_path}")
        
    except Exception as e:
        print(f"[WARNING] Failed to save generation info: {e}")
    
    print("[COMPLETE] Sampling finished successfully!")
    return 0


if __name__ == "__main__":
    exit(main())