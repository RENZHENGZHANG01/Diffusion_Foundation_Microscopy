"""
Microscopy Dataset Loaders
==========================
Handles loading and pairing of foundation/degraded microscopy data
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List
import sys
import random
import numpy as np
from PIL import Image

# Add processed dataset path - keep external dependency
sys.path.append('/mnt/g/My Drive/Microscopy_Dataset/fluorescence_datasets/foundation')
from processed_dataset import build_manifest_dataset

# Local imports
from .conditioning import create_microscopy_conditions


class MicroscopyDataset(Dataset):
    """Dataset for microscopy training with phase-aware loading"""
    
    def __init__(self, config: Dict, phase_config: Dict):
        self.config = config
        self.phase_config = phase_config
        self.condition_dropout = phase_config.get('condition_dropout', 0.0)
        
        # Load datasets based on phase
        if phase_config['type'] == 'unconditional':
            self._setup_unconditional()
        else:
            self._setup_conditional()
    
    def _setup_unconditional(self):
        """Setup Phase 1: Foundation data only"""
        self.dataset = build_manifest_dataset(
            roots=[self.config['data']['foundation_path']],
            include_datasets=self.phase_config['datasets']
        )
        self.condition_types = []
        print(f"[DATASET] Phase 1: {len(self.dataset)} foundation samples")
    
    def _setup_conditional(self):
        """Setup Phase 2: Foundation + degraded pairs"""
        
        # Load foundation dataset
        self.foundation_dataset = build_manifest_dataset(
            roots=[self.config['data']['foundation_path']],
            include_datasets=self.phase_config['datasets']
        )
        
        # Load degraded datasets with proper naming
        degraded_datasets = []
        for dataset_name in self.phase_config['datasets']:
            if dataset_name == 'sr_caco2':
                degraded_datasets.append('sr_caco2_lowres')
            elif dataset_name == 'fmd':
                degraded_datasets.append('fmd_noisy')
            else:
                degraded_datasets.append(f"{dataset_name}_degraded")
        
        self.degraded_dataset = build_manifest_dataset(
            roots=[self.config['data']['degraded_path']],
            include_datasets=degraded_datasets
        )
        
        self.condition_types = self.phase_config.get('condition_types', [])
        
        print(f"[DATASET] Phase 2: {len(self.foundation_dataset)} foundation + {len(self.degraded_dataset)} degraded samples")
        print(f"[CONDITIONS] Types: {self.condition_types}")
    
    def __len__(self):
        if hasattr(self, 'foundation_dataset'):
            return len(self.foundation_dataset)
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.phase_config['type'] == 'unconditional':
            return self._get_unconditional_sample(idx)
        else:
            return self._get_conditional_sample(idx)
    
    def _get_unconditional_sample(self, idx):
        """Get Phase 1 sample: just images with descriptions"""
        sample = self.dataset[idx]
        
        # Create descriptive prompt
        description = self._create_description(sample)
        
        return {
            'image': sample['image'],
            'description': description
        }
    
    def _get_conditional_sample(self, idx):
        """Get Phase 2 sample: images + OminiControl-style conditions"""
        
        # Get foundation sample
        foundation_sample = self.foundation_dataset[idx % len(self.foundation_dataset)]
        
        # Apply condition dropout (train unconditional capability)
        use_conditions = random.random() > self.condition_dropout
        
        # Create base return dict
        result = {
            'image': foundation_sample['image'],
            'description': self._create_description(foundation_sample)
        }
        
        if use_conditions:
            # Create conditions using OminiControl-style approach
            try:
                conditions = create_microscopy_conditions(
                    foundation_image=foundation_sample['image'],
                    condition_types=self.condition_types,
                    noise_std=25.0,
                    downsample_factor=4
                )
                
                # Add conditions to result
                for condition_idx, condition_type in enumerate(self.condition_types):
                    if condition_type in conditions:
                        condition_obj = conditions[condition_type]
                        result[f'condition_{condition_idx}'] = condition_obj.condition_image
                        result[f'condition_type_{condition_idx}'] = condition_type
                        result[f'condition_strength_{condition_idx}'] = condition_obj.strength
            
            except Exception as e:
                print(f"[WARNING] Failed to create conditions: {e}")
                # Fallback: try degraded dataset approach
                self._add_degraded_conditions(result, foundation_sample)
        
        return result
    
    def _add_degraded_conditions(self, result, foundation_sample):
        """Fallback: add conditions from degraded dataset"""
        condition_idx = 0
        
        if 'super_resolution' in self.condition_types:
            # Find matching degraded sample (low-res version)
            degraded_sample = self._find_matching_degraded(foundation_sample, 'sr_caco2_lowres')
            if degraded_sample:
                result[f'condition_{condition_idx}'] = degraded_sample['image']
                result[f'condition_type_{condition_idx}'] = 'super_resolution'
                condition_idx += 1
        
        if 'denoising' in self.condition_types:
            # Find matching degraded sample (noisy version)
            degraded_sample = self._find_matching_degraded(foundation_sample, 'fmd_noisy')
            if degraded_sample:
                result[f'condition_{condition_idx}'] = degraded_sample['image']
                result[f'condition_type_{condition_idx}'] = 'denoising'
                condition_idx += 1
    
    def _find_matching_degraded(self, foundation_sample, degraded_dataset_name):
        """Find matching degraded sample for foundation sample"""
        
        # Simple approach: find sample from same base dataset
        foundation_dataset = foundation_sample['dataset']
        
        # Map foundation to degraded dataset
        target_dataset = None
        if foundation_dataset == 'sr_caco2' and degraded_dataset_name == 'sr_caco2_lowres':
            target_dataset = 'sr_caco2_lowres'
        elif foundation_dataset == 'fmd' and degraded_dataset_name == 'fmd_noisy':
            target_dataset = 'fmd_noisy'
        
        if not target_dataset:
            return None
        
        # Find samples from target dataset
        matching_samples = [
            sample for sample in self.degraded_dataset.rows
            if sample.dataset == target_dataset
        ]
        
        if matching_samples:
            # Return random matching sample
            selected = random.choice(matching_samples)
            return {
                'image': torch.load(selected.image) if selected.image.endswith('.pt') else 
                        torch.from_numpy(np.array(Image.open(selected.image))).float() / 255.0,
                'dataset': selected.dataset
            }
        
        return None
    
    def _create_description(self, sample):
        """Create descriptive prompt for sample"""
        
        components = ["high quality microscopy image"]
        
        # Add modality
        if sample.get('modality'):
            components.append(f"{sample['modality'].lower()} microscopy")
        
        # Add specimen info
        if sample.get('specimen'):
            specimen = sample['specimen'].replace('_', ' ').replace('-', ' ')
            components.append(f"of {specimen}")
        
        # Add marker info
        if sample.get('marker'):
            marker = sample['marker'].lower()
            components.append(f"showing {marker}")
        
        # Add task context
        if sample.get('task'):
            task_descriptions = {
                'super_resolution': 'high resolution detail',
                'denoising': 'clear structure',
                'segmentation': 'cellular boundaries',
                'classification': 'morphological features'
            }
            if sample['task'] in task_descriptions:
                components.append(f"with {task_descriptions[sample['task']]}")
        
        return ", ".join(components)


class ValidationDataset(Dataset):
    """Validation dataset for evaluation"""
    
    def __init__(self, foundation_path: str, datasets: List[str], max_samples: int = 100):
        self.dataset = build_manifest_dataset(
            roots=[foundation_path],
            include_datasets=datasets
        )
        
        # Limit validation size
        if len(self.dataset) > max_samples:
            indices = torch.randperm(len(self.dataset))[:max_samples]
            self.dataset.rows = [self.dataset.rows[i] for i in indices]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            'image': sample['image'],
            'description': f"microscopy {sample['dataset']} validation"
        }