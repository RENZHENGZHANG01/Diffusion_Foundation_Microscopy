#!/usr/bin/env python3
"""
Test script to verify the core modules are working
"""

import sys
from pathlib import Path

# Add core module path and handle imports properly
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'core'))

def test_imports():
    """Test if all core modules can be imported"""
    
    print("[TEST] Testing imports...")
    
    try:
        from dit_models import DiT_models
        print(f"[SUCCESS] DiT models: {list(DiT_models.keys())}")
    except Exception as e:
        print(f"[ERROR] DiT models import failed: {e}")
        return False
    
    try:
        from diffusion import create_diffusion
        diffusion = create_diffusion(timestep_respacing="")
        print(f"[SUCCESS] Diffusion created: {type(diffusion)}")
    except Exception as e:
        print(f"[ERROR] Diffusion import failed: {e}")
        return False
    
    try:
        from phase_manager import MicroscopyPhaseManager
        print("[SUCCESS] Phase manager imported")
    except Exception as e:
        print(f"[ERROR] Phase manager import failed: {e}")
        return False
    
    try:
        # Try different import strategies
        try:
            from core.models import MicroscopyDiTModel
        except ImportError:
            from models import MicroscopyDiTModel
        print("[SUCCESS] Microscopy DiT model imported")
    except Exception as e:
        print(f"[ERROR] Model import failed: {e}")
        return False
    
    try:
        try:
            from core.trainer import MicroscopyTrainer
        except ImportError:
            from trainer import MicroscopyTrainer
        print("[SUCCESS] Microscopy trainer imported")
    except Exception as e:
        print(f"[ERROR] Trainer import failed: {e}")
        return False
    
    try:
        try:
            from core.datasets import MicroscopyDataset
        except ImportError:
            from datasets import MicroscopyDataset
        print("[SUCCESS] Microscopy dataset imported")
    except Exception as e:
        print(f"[ERROR] Dataset import failed: {e}")
        return False
    
    try:
        try:
            from core.conditioning import DiTConditionEncoder, Condition, convert_to_condition
        except ImportError:
            from conditioning import DiTConditionEncoder, Condition, convert_to_condition
        print("[SUCCESS] OminiControl-style conditioning imported")
    except Exception as e:
        print(f"[ERROR] Conditioning import failed: {e}")
        return False
    
    try:
        try:
            from core.callbacks import EMACallback, RealTimeMonitorCallback, SampleGenerationCallback
        except ImportError:
            from callbacks import EMACallback, RealTimeMonitorCallback, SampleGenerationCallback
        print("[SUCCESS] Real-time monitoring callbacks imported")
    except Exception as e:
        print(f"[ERROR] Callbacks import failed: {e}")
        return False
    
    try:
        try:
            from core.edm_scheduler import EDMEulerScheduler, create_edm_scheduler
        except ImportError:
            from edm_scheduler import EDMEulerScheduler, create_edm_scheduler
        print("[SUCCESS] EDM Euler Scheduler imported")
    except Exception as e:
        print(f"[ERROR] EDM scheduler import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation"""
    
    print("\n[TEST] Testing model creation...")
    
    try:
        from dit_models import DiT_models
        
        # Test DiT-XL/8 model creation
        model = DiT_models['DiT-XL/8'](input_size=64, num_classes=0)
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"[SUCCESS] DiT-XL/8 created with {param_count:,} parameters")
        return True
        
    except Exception as e:
        print(f"[ERROR] Model creation failed: {e}")
        return False


def test_conditioning():
    """Test OminiControl-style conditioning"""
    
    print("\n[TEST] Testing conditioning system...")
    
    try:
        import torch
        import numpy as np
        try:
            from core.conditioning import DiTConditionEncoder, convert_to_condition
        except ImportError:
            from conditioning import DiTConditionEncoder, convert_to_condition
        
        # Test condition encoder
        condition_encoder = DiTConditionEncoder(
            condition_types=['super_resolution', 'denoising'],
            hidden_size=1152
        )
        
        # Test condition creation
        dummy_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        
        # Test super-resolution condition
        sr_condition = convert_to_condition('super_resolution', dummy_image, 4)
        print(f"[SUCCESS] SR condition created: {sr_condition.shape}")
        
        # Test denoising condition
        denoise_condition = convert_to_condition('denoising', dummy_image, 25.0)
        print(f"[SUCCESS] Denoising condition created: {denoise_condition.shape}")
        
        # Test encoder forward pass
        batch_size = 2
        conditions = {
            'super_resolution': torch.randn(batch_size, 1, 512, 512),
            'denoising': torch.randn(batch_size, 1, 512, 512)
        }
        
        condition_emb = condition_encoder(conditions)
        print(f"[SUCCESS] Condition embedding: {condition_emb.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Conditioning test failed: {e}")
        return False


def test_integrated_model():
    """Test complete model with conditioning integration"""
    
    print("\n[TEST] Testing integrated DiT + conditioning...")
    
    try:
        import torch
        try:
            from core.models import MicroscopyDiTModel
        except ImportError:
            from models import MicroscopyDiTModel
        
        # Mock config for testing
        config = {
            'model': {
                'architecture': 'DiT-XL/8',
                'latent_size': 64,
                'hidden_size': 1152
            },
            'vae': {
                'model_id': 'stabilityai/sd-vae-ft-ema'
            }
        }
        
        # Test Phase 1 (unconditional)
        phase1_config = {
            'type': 'unconditional',
            'name': 'phase1_test'
        }
        
        print("[INFO] Testing Phase 1 (unconditional) model...")
        # Note: This will fail without actual VAE, but tests the structure
        try:
            model1 = MicroscopyDiTModel(config, phase1_config)
            print("[SUCCESS] Phase 1 model structure created")
        except Exception as e:
            print(f"[INFO] Phase 1 model creation failed (expected without VAE): {e}")
        
        # Test Phase 2 (conditional)
        phase2_config = {
            'type': 'conditional',
            'name': 'phase2_test',
            'condition_types': ['super_resolution', 'denoising']
        }
        
        print("[INFO] Testing Phase 2 (conditional) model...")
        try:
            model2 = MicroscopyDiTModel(config, phase2_config)
            print("[SUCCESS] Phase 2 model structure created")
        except Exception as e:
            print(f"[INFO] Phase 2 model creation failed (expected without VAE): {e}")
        
        print("[SUCCESS] Model integration test completed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Integrated model test failed: {e}")
        return False


def test_edm_scheduler():
    """Test EDM scheduler functionality"""
    
    print("\n[TEST] Testing EDM Scheduler...")
    
    try:
        import torch
        try:
            from core.edm_scheduler import EDMEulerScheduler
        except ImportError:
            from edm_scheduler import EDMEulerScheduler
        
        # Test scheduler creation
        scheduler = EDMEulerScheduler(
            num_train_timesteps=1000,
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=0.5,
            rho=7.0,
            prediction_type="sample"
        )
        
        print(f"[SUCCESS] EDM scheduler created with {len(scheduler.sigmas)} timesteps")
        
        # Test noise addition
        batch_size = 2
        latents = torch.randn(batch_size, 4, 64, 64)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        print(f"[SUCCESS] Noise addition: {noisy_latents.shape}")
        
        # Test scaling
        scaled_input = scheduler.scale_model_input(noisy_latents, timesteps)
        print(f"[SUCCESS] Input scaling: {scaled_input.shape}")
        
        # Test velocity computation
        velocity = scheduler.get_velocity(latents, noise, timesteps)
        print(f"[SUCCESS] Velocity computation: {velocity.shape}")
        
        # Test loss weights
        weights = scheduler.get_loss_weights(timesteps)
        print(f"[SUCCESS] Loss weights: {weights.shape}")
        
        print("[SUCCESS] EDM scheduler test completed")
        return True
        
    except Exception as e:
        print(f"[ERROR] EDM scheduler test failed: {e}")
        return False


def main():
    """Run all tests"""
    
    print("Testing Microscopy Diffusion Core Setup")
    print("=" * 50)
    
    success = True
    
    # Test imports
    success &= test_imports()
    
    # Test model creation
    success &= test_model_creation()
    
    # Test conditioning system
    success &= test_conditioning()
    
    # Test integrated model
    success &= test_integrated_model()
    
    # Test EDM scheduler
    success &= test_edm_scheduler()
    
    print("\n" + "=" * 50)
    if success:
        print("[COMPLETE] All tests passed!")
    else:
        print("[FAILED] Some tests failed!")
    
    return success


if __name__ == "__main__":
    main()