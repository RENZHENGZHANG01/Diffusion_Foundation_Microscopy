#!/usr/bin/env python3
"""
Sampling Examples for Microscopy Diffusion Models
==================================================
Demonstrates various sampling configurations and use cases.
"""

import sys
from pathlib import Path

# Add repo path
repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir))
sys.path.insert(0, str(repo_dir / 'core'))

from core.sampling import MicroscopySampler, create_condition_hints


def example_unconditional_sampling():
    """Example: Generate unconditional samples from Phase 1 model"""
    print("=== Unconditional Sampling Example ===")
    
    # Create sampler
    sampler = MicroscopySampler("config/sampling.yaml")
    
    # Generate samples
    samples = sampler.generate_samples(
        num_samples=4,
        image_size=(512, 512),
        seed=42
    )
    
    # Save samples
    saved_paths = sampler.save_samples(samples, "examples/unconditional_samples", "unconditional")
    print(f"Saved {len(saved_paths)} files:")
    for path in saved_paths:
        print(f"  - {path}")


def example_conditional_sampling():
    """Example: Generate conditional samples from Phase 2 model"""
    print("\n=== Conditional Sampling Example ===")
    
    # Create sampler with conditional config
    config_path = "config/sampling.yaml"
    sampler = MicroscopySampler(config_path)
    
    # Create condition hints
    condition_hints = create_condition_hints(
        image_size=(512, 512),
        condition_types=["super_resolution", "denoising", "canny_edge"],
        hints_config={
            'super_resolution_scale': 2.0,
            'denoising_strength': 0.3,
            'canny_low_threshold': 50,
            'canny_high_threshold': 150
        }
    )
    
    # Generate samples
    samples = sampler.generate_samples(
        num_samples=4,
        image_size=(512, 512),
        condition_hints=condition_hints,
        seed=42
    )
    
    # Save samples
    saved_paths = sampler.save_samples(samples, "examples/conditional_samples", "conditional")
    print(f"Saved {len(saved_paths)} files:")
    for path in saved_paths:
        print(f"  - {path}")


def example_different_samplers():
    """Example: Compare different sampling methods"""
    print("\n=== Different Sampling Methods Example ===")
    
    methods = ["ddpm", "ddim", "edm_euler"]
    
    for method in methods:
        print(f"\n--- {method.upper()} Sampling ---")
        
        # Create sampler with specific method
        sampler = MicroscopySampler("config/sampling.yaml")
        
        # Update config for this method
        sampler.config['sampling']['method'] = method
        sampler.config['sampling']['num_steps'] = 25  # Faster for demo
        
        # Generate samples
        samples = sampler.generate_samples(
            num_samples=2,
            image_size=(512, 512),
            seed=42
        )
        
        # Save samples
        saved_paths = sampler.save_samples(samples, f"examples/{method}_samples", method)
        print(f"Saved {len(saved_paths)} files for {method}")


def example_batch_generation():
    """Example: Generate many samples efficiently"""
    print("\n=== Batch Generation Example ===")
    
    sampler = MicroscopySampler("config/sampling.yaml")
    
    # Generate many samples in batches
    total_samples = 16
    batch_size = 4
    
    all_samples = []
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"Generating {total_samples} samples in {num_batches} batches of {batch_size}")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_samples = end_idx - start_idx
        
        print(f"Batch {batch_idx + 1}/{num_batches}: {batch_samples} samples")
        
        batch_output = sampler.generate_samples(
            num_samples=batch_samples,
            image_size=(512, 512),
            seed=42 + batch_idx  # Different seed per batch
        )
        
        all_samples.append(batch_output)
    
    # Concatenate and save
    import torch
    all_samples = torch.cat(all_samples, dim=0)
    
    saved_paths = sampler.save_samples(all_samples, "examples/batch_samples", "batch")
    print(f"Saved {len(saved_paths)} files for batch generation")


def example_custom_configuration():
    """Example: Use custom sampling configuration"""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom config
    custom_config = {
        'model': {
            'checkpoint_path': 'logs/checkpoints/best_model.ckpt',
            'phase': 'conditional'
        },
        'sampling': {
            'method': 'ddim',
            'num_steps': 10,  # Very fast
            'guidance_scale': 5.0,
            'ddim_eta': 0.0
        },
        'output': {
            'num_samples': 2,
            'image_size': [256, 256],  # Smaller for speed
            'save_dir': 'examples/custom_samples',
            'save_individual': True,
            'save_grid': True
        },
        'vae': {
            'enabled': True,
            'scaling_factor': 0.18215
        },
        'conditioning': {
            'enabled': True,
            'condition_types': ['super_resolution']
        },
        'device': {
            'device': 'auto'
        },
        'advanced': {
            'show_progress': True,
            'seed': 123
        }
    }
    
    # Save custom config
    import yaml
    custom_config_path = "examples/custom_sampling_config.yaml"
    with open(custom_config_path, 'w') as f:
        yaml.dump(custom_config, f, default_flow_style=False)
    
    # Create sampler with custom config
    sampler = MicroscopySampler(custom_config_path)
    
    # Generate samples
    samples = sampler.generate_samples()
    
    # Save samples
    saved_paths = sampler.save_samples(samples, "examples/custom_samples", "custom")
    print(f"Saved {len(saved_paths)} files with custom configuration")


def main():
    """Run all examples"""
    print("Microscopy Diffusion Sampling Examples")
    print("=====================================")
    
    try:
        # Create examples directory
        Path("examples").mkdir(exist_ok=True)
        
        # Run examples
        example_unconditional_sampling()
        example_conditional_sampling()
        example_different_samplers()
        example_batch_generation()
        example_custom_configuration()
        
        print("\n=== All Examples Completed Successfully! ===")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()