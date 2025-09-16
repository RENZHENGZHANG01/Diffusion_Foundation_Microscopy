#!/usr/bin/env python3
"""Simple sampler for Microscopy Diffusion (EDM/DDIM/DDPM)."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml

repo_dir = Path(__file__).parent
sys.path.insert(0, str(repo_dir))
sys.path.insert(0, str(repo_dir / 'core'))

from core.sampling import MicroscopySampler, create_condition_hints


def parse_args():
    p = argparse.ArgumentParser(description="Microscopy Diffusion Sampling")
    p.add_argument('--config', type=str, required=True)
    # sampling
    p.add_argument('--num_samples', type=int)
    # EDM-only
    p.add_argument('--steps', type=int)
    # output
    p.add_argument('--output_dir', type=str)
    p.add_argument('--image_size', type=int, nargs=2)
    p.add_argument('--prefix', type=str, default='sample')
    # conditional
    p.add_argument('--conditional', action='store_true')
    p.add_argument('--condition_types', type=str)
    p.add_argument('--condition_strength', type=float)
    # advanced
    p.add_argument('--seed', type=int)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--device', type=str)
    p.add_argument('--no_progress', action='store_true')
    # model
    p.add_argument('--checkpoint', type=str)
    p.add_argument('--phase', type=str, choices=['unconditional', 'conditional'])
    return p.parse_args()


def apply_overrides(cfg: Dict, a) -> Dict:
    # sampling
    if a.num_samples is not None:
        cfg['output']['num_samples'] = a.num_samples
    if a.steps is not None:
        cfg['sampling']['num_steps'] = a.steps
    # output
    if a.output_dir is not None:
        cfg['output']['save_dir'] = a.output_dir
    if a.image_size is not None:
        cfg['output']['image_size'] = list(a.image_size)
    # conditional
    if a.conditional:
        cfg.setdefault('conditioning', {}).update({'enabled': True})
    if a.condition_types is not None:
        cfg.setdefault('conditioning', {})['condition_types'] = a.condition_types.split(',')
    if a.condition_strength is not None:
        cfg.setdefault('conditioning', {})['condition_strength'] = a.condition_strength
    # advanced
    if a.seed is not None:
        cfg.setdefault('advanced', {})['seed'] = a.seed
    if a.batch_size is not None:
        cfg.setdefault('advanced', {})['batch_size'] = a.batch_size
    if a.device is not None:
        cfg.setdefault('device', {})['device'] = a.device
    if a.no_progress:
        cfg.setdefault('advanced', {})['show_progress'] = False
    # model
    if a.checkpoint is not None:
        cfg.setdefault('model', {})['checkpoint_path'] = a.checkpoint
    if a.phase is not None:
        cfg.setdefault('model', {})['phase'] = a.phase
    return cfg


def main() -> int:
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args)

    seed = cfg.get('advanced', {}).get('seed')
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Write a temp config file to construct the sampler
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    try:
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name
        tmp.close()
        sampler = MicroscopySampler(tmp_path)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    # Optional condition hints
    cond_hints: Optional[Dict] = None
    cond_cfg = cfg.get('conditioning', {})
    if cond_cfg.get('enabled', False):
        ctypes = cond_cfg.get('condition_types', [])
        if ctypes:
            cond_hints = create_condition_hints(
                image_size=tuple(cfg['output']['image_size']),
                condition_types=ctypes,
                hints_config=cond_cfg.get('hints', {})
            )

    # Force EDM method in config
    cfg.setdefault('sampling', {})['method'] = 'edm_euler'

    # Generate (supports batching)
    total = int(cfg['output']['num_samples'])
    bsz = int(cfg.get('advanced', {}).get('batch_size', 1) or 1)
    batches = []
    while len(batches) * bsz < total:
        n = min(bsz, total - len(batches) * bsz)
        batches.append(sampler.generate_samples(num_samples=n, condition_hints=cond_hints))
    samples = torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]

    # Save images and info
    out_dir = cfg['output']['save_dir']
    saved = sampler.save_samples(samples, out_dir, args.prefix)
    info = {
        'num_samples': total,
        'sampling_method': 'edm_euler',
        'num_steps': cfg['sampling']['num_steps'],
        'image_size': cfg['output']['image_size'],
        'conditioning_enabled': bool(cond_cfg.get('enabled', False)),
        'condition_types': cond_cfg.get('condition_types', []),
        'seed': seed,
        'saved_files': saved,
    }
    info_path = Path(out_dir) / f"{args.prefix}_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved {len(saved)} files to {out_dir}\nInfo: {info_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())