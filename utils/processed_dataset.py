#!/usr/bin/env python3
"""Microscopy Foundation Processed Dataset Loader
=================================================

This module provides utilities to load the preprocessed microscopy foundation
datasets produced by the preprocessing pipeline (microscopy_preproc.py + pack_wds.py).

Supports two consumption modes:

1. Manifest mode (default)
   Directory layout per dataset:
       <root>/<dataset>/
          images/*.png | *.npz
          manifests/manifest.csv (columns: id,image,dataset,task,channel_idx,norm_lo,norm_hi,x0,y0,modality,specimen,NA,lambda,um_pixel,marker,...)
          manifests/train.csv / val.csv / test.csv (optional splits)

2. WebDataset shard mode (optional, if you packed shards)
       <root>/shards/shard-00000.tar  ...
       <root>/lists/train.txt (lines of shard paths)

The goal is to give a unified PyTorch iterable returning tensors:
    sample = {
        'image': FloatTensor[C,H,W] in [0,1],
        'id': str,
        'dataset': str,
        'task': str,
        'channel_idx': int | str,
        'shape': (C,H,W),
        'path': str,
        'x0': int,
        'y0': int,
        'modality': str,
        'specimen': str,
        'NA': str,
        'lambda': str,
        'um_pixel': str,
        'marker': str,
        'norm_lo': dict,
        'norm_hi': dict,
        'meta': dict  # any extra fields from manifest
    }

Key Features
------------
* 16-bit PNG auto-converted to float32 / 65535.
* .npy multi-channel (e.g., RxRx1 6 channels) loaded directly.
* Optional transforms (callable taking sample dict and returning modified sample).
* Filtering by dataset names / channel count / custom predicate.
* WebDataset convenience wrapper (only if `webdataset` installed; else raises).

Example (Manifest Mode)
-----------------------
    from processed_dataset import build_manifest_dataset
    ds = build_manifest_dataset(
        roots=['/mnt/data/foundation_preproc_out'],
        splits=['train'],               # or [] to use all manifest rows
        include_datasets=None,          # or list of dataset folder names
        transform=None,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
    for batch in loader:
        imgs = batch['image']  # (B,C,H,W)
        ...

Example (WebDataset Mode)
-------------------------
    from processed_dataset import build_wds_dataloader
    loader = build_wds_dataloader(
        list_file='/mnt/data/foundation_preproc_out/lists/train.txt',
        batch_size=32,
        num_workers=8,
    )
    for sample in loader:
        ...

License: MIT (adjust as needed).
"""
from __future__ import annotations

import csv
import os
import io
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError("Pillow is required to load PNG images. Install via `pip install pillow`." ) from e

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required. Install via `pip install torch`." ) from e

# ---------------------------------------------------------------------------
# Manifest Dataset
# ---------------------------------------------------------------------------

@dataclass
class ManifestRow:
    id: str
    image: str
    dataset: str
    task: str
    channel_idx: Union[int, str]
    norm_lo: str
    norm_hi: str
    x0: int
    y0: int
    modality: str
    specimen: str
    NA: str
    lambda_: str
    um_pixel: str
    marker: str
    extra: Dict[str, Any]


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', newline='') as f:
        rd = csv.DictReader(f)
        return list(rd)


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    """Ensure float32 in [0,1]. Assume input already scaled if <=1 else scale by dtype max."""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # If values appear >1, attempt min-max (avoid changing distribution if already small)
        if arr.max() > 1.0:
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
        return arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return (arr.astype(np.float32) / float(info.max)).clip(0, 1)
    return arr.astype(np.float32)


def _load_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.npy', '.npz'):
        if ext == '.npy':
            arr = np.load(path)
        else:  # .npz
            with np.load(path) as data:
                # Assume the array is stored with key 'image' (from process_foundation.py)
                if 'image' in data:
                    arr = data['image']
                else:
                    # Fallback: take the first array in the file
                    arr = data[data.files[0]]
        
        # Expect (H,W,C) or (C,H,W)
        if arr.ndim == 2:
            arr = arr[..., None]
        elif arr.ndim == 3 and arr.shape[0] <= 6 and arr.shape[0] < arr.shape[-1]:
            # assume (H,W,C); leave
            pass
        elif arr.ndim == 3 and arr.shape[0] > arr.shape[-1]:
            # assume (C,H,W)
            arr = np.moveaxis(arr, 0, -1)
        return _normalize_image(arr)
    # PNG or TIFF (converted to PNG in pipeline). Use PIL.
    with Image.open(path) as im:
        im.load()
        # Convert 16-bit or 8-bit grayscale/RGB to array
        arr = np.array(im)
    if arr.ndim == 2:  # H,W
        arr = arr[:, :, None]
    return _normalize_image(arr)


class PreprocessedMicroscopyDataset(Dataset):
    """PyTorch Dataset over one or multiple preprocessed dataset roots.

    Arguments
    ---------
    roots : list[str]
        Paths each containing one or more dataset directories.
    splits : list[str]
        If provided, attempts to read per-dataset manifests/<split>.csv and merge.
        If empty, uses the base manifest.csv rows.
    include_datasets : list[str] | None
        Restrict to only these dataset directory names.
    transform : callable | None
        Optional function(sample_dict) -> sample_dict after loading image & meta.
    filter_fn : callable | None
        Optional predicate(row_dict) -> bool to keep sample.
    cache_paths : bool
        If True, pre-expand and keep a list; else compute on the fly (list size is fine usually).
    """

    def __init__(
        self,
        roots: Sequence[str],
        splits: Sequence[str] | None = None,
        include_datasets: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        filter_fn: Optional[Callable[[ManifestRow], bool]] = None,
        cache_paths: bool = True,
    ) -> None:
        self.roots = [os.path.abspath(r) for r in roots]
        self.splits = list(splits) if splits else []
        self.include = set(include_datasets) if include_datasets else None
        self.transform = transform
        self.filter_fn = filter_fn
        self.rows: List[ManifestRow] = []
        self._build_index()
        if cache_paths:
            # Force path existence check early (skip for testing)
            missing = [r.image for r in self.rows if not os.path.exists(r.image)]
            if missing:
                print(f"[WARNING] Missing {len(missing)} image files; first: {missing[0]}")
                print("[INFO] Filtering out missing files for testing...")
                self.rows = [r for r in self.rows if os.path.exists(r.image)]
                print(f"[INFO] Using {len(self.rows)} available samples")

    # ------------------------------------------------------------------
    def _gather_dataset_dirs(self) -> List[str]:
        out = []
        for root in self.roots:
            for name in sorted(os.listdir(root)):
                d = os.path.join(root, name)
                if not os.path.isdir(d):
                    continue
                if self.include and name not in self.include:
                    continue
                # must contain manifests/manifest.csv
                if os.path.exists(os.path.join(d, 'manifests', 'manifest.csv')):
                    out.append(d)
        return out

    def _read_split_rows(self, ds_dir: str, ds_name: str) -> List[ManifestRow]:
        mani_dir = os.path.join(ds_dir, 'manifests')
        base_manifest = os.path.join(mani_dir, 'manifest.csv')
        if not self.splits:
            sources = [base_manifest]
        else:
            sources = []
            for sp in self.splits:
                p = os.path.join(mani_dir, f'{sp}.csv')
                if os.path.exists(p):
                    sources.append(p)
            if not sources:  # fallback
                sources = [base_manifest]
        rows: List[ManifestRow] = []
        for src_path in sources:
            print(f"[DEBUG] Reading manifest: {src_path}")
            csv_rows = _read_csv_rows(src_path)
            print(f"[DEBUG] Found {len(csv_rows)} rows in manifest")
            
            for i, rec in enumerate(csv_rows):
                if i % 1000 == 0:
                    print(f"[DEBUG] Processing row {i}/{len(csv_rows)}")
                img_path = rec.get('image') or ''
                if not os.path.isabs(img_path):
                    # allow relative paths inside manifest
                    img_path = os.path.join(ds_dir, img_path)
                
                # Handle case where manifest has .tif but files are .png  
                if not os.path.exists(img_path) and img_path.endswith('.tif'):
                    # Try finding corresponding .png file in images/ directory
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    png_path = os.path.join(ds_dir, 'images', f'{rec.get("id", base_name)}.png')
                    if os.path.exists(png_path):
                        img_path = png_path
                
                # Parse channel_idx - can be empty string or integer
                channel_idx_raw = rec.get('channel_idx', '')
                channel_idx = int(channel_idx_raw) if channel_idx_raw and channel_idx_raw.isdigit() else channel_idx_raw
                
                try:
                    # Core fields that should exist in the manifest
                    core_fields = {'id', 'image', 'dataset', 'task', 'channel_idx', 'norm_lo', 'norm_hi', 'x0', 'y0', 
                                 'modality', 'specimen', 'NA', 'lambda', 'um_pixel', 'marker'}
                    
                    row = ManifestRow(
                        id=str(rec.get('id', '')),
                        image=img_path,
                        dataset=rec.get('dataset', '') or ds_name,
                        task=rec.get('task', ''),
                        channel_idx=channel_idx,
                        norm_lo=rec.get('norm_lo', '{}'),
                        norm_hi=rec.get('norm_hi', '{}'),
                        x0=int(float(rec.get('x0', '0'))),
                        y0=int(float(rec.get('y0', '0'))),
                        modality=rec.get('modality', ''),
                        specimen=rec.get('specimen', ''),
                        NA=rec.get('NA', ''),
                        lambda_=rec.get('lambda', ''),
                        um_pixel=rec.get('um_pixel', ''),
                        marker=rec.get('marker', ''),
                        extra={k: v for k, v in rec.items() if k not in core_fields},
                    )
                except Exception as e:  # pragma: no cover
                    raise ValueError(f"Malformed row in {src_path}: {rec}\nError: {e}")
                if self.filter_fn and not self.filter_fn(row):
                    continue
                rows.append(row)
        return rows

    def _build_index(self) -> None:
        ds_dirs = self._gather_dataset_dirs()
        for d in ds_dirs:
            name = os.path.basename(d)
            self.rows.extend(self._read_split_rows(d, name))
        if not self.rows:
            raise RuntimeError("No samples found. Check roots/splits/include_datasets parameters.")

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        r = self.rows[idx]
        arr = _load_image(r.image)
        # Convert to torch tensor (C,H,W)
        if arr.ndim != 3:
            raise ValueError(f"Loaded array has shape {arr.shape}, expected (H,W,C)")
        arr = np.moveaxis(arr, -1, 0)  # C,H,W
        tensor = torch.from_numpy(arr.astype(np.float32))  # already normalized [0,1]
        
        # Parse normalization parameters from JSON strings if available
        norm_lo_dict = {}
        norm_hi_dict = {}
        try:
            if r.norm_lo:
                norm_lo_dict = json.loads(r.norm_lo)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            if r.norm_hi:
                norm_hi_dict = json.loads(r.norm_hi)
        except (json.JSONDecodeError, TypeError):
            pass
        
        sample: Dict[str, Any] = {
            'image': tensor,
            'id': r.id,
            'dataset': r.dataset,
            'task': r.task,
            'channel_idx': r.channel_idx,
            'shape': tuple(tensor.shape),
            'path': r.image,
            'x0': r.x0,
            'y0': r.y0,
            'modality': r.modality,
            'specimen': r.specimen,
            'NA': r.NA,
            'lambda': r.lambda_,
            'um_pixel': r.um_pixel,
            'marker': r.marker,
            'norm_lo': norm_lo_dict,
            'norm_hi': norm_hi_dict,
            'meta': {**r.extra, 'H': tensor.shape[1], 'W': tensor.shape[2], 'C': tensor.shape[0]},
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def build_manifest_dataset(
    roots: Sequence[str],
    splits: Sequence[str] | None = None,
    include_datasets: Optional[Sequence[str]] = None,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    filter_fn: Optional[Callable[[ManifestRow], bool]] = None,
) -> PreprocessedMicroscopyDataset:
    """Construct a manifest-backed dataset."""
    return PreprocessedMicroscopyDataset(
        roots=roots,
        splits=splits,
        include_datasets=include_datasets,
        transform=transform,
        filter_fn=filter_fn,
    )


# ---------------------------------------------------------------------------
# WebDataset integration (optional)
# ---------------------------------------------------------------------------

def _has_webdataset() -> bool:
    try:
        import webdataset  # noqa: F401
        return True
    except Exception:
        return False


def build_wds_dataloader(
    list_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    nodes_rank_world: Tuple[int, int, int] | None = None,
    decode: str = 'pil',
    handler: Optional[Callable] = None,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
):
    """Build a WebDataset DataLoader from a shard list.

    Parameters
    ----------
    list_file : str
        Path to train/val/test shard list (each line absolute or relative path to *.tar).
    nodes_rank_world : (rank, world_size, workers_per_node) optional
        If doing DDP manual shard partitioning; WebDataset can also infer automatically if env vars set.
    decode : 'pil' | 'torch' | 'numpy'
        Decoding method for image formats (.png). .npy always via numpy.
    transform : callable or None
        Applied per sample after decoding.
    """
    if not _has_webdataset():  # pragma: no cover
        raise ImportError("webdataset not installed. Install via `pip install webdataset`." )

    import webdataset as wds
    import torch

    with open(list_file, 'r') as f:
        shard_paths = [ln.strip() for ln in f if ln.strip()]
    if not shard_paths:
        raise ValueError(f"No shard paths found in {list_file}")

    # Optionally subset for rank
    if nodes_rank_world is not None:
        rank, world_size, _ = nodes_rank_world
        shard_paths = shard_paths[rank::world_size]

    def _decode_sample(sample):
        # sample keys: e.g., {'__key__': 'abc', 'png': bytes, 'json': bytes} or {'npy': bytes}
        meta = json.loads(sample['json'].decode('utf-8')) if 'json' in sample else {}
        key = meta.get('key', sample.get('__key__'))
        if 'png' in sample:
            from PIL import Image
            with Image.open(io.BytesIO(sample['png'])) as im:
                arr = np.array(im)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            arr = _normalize_image(arr)
        elif 'npy' in sample:
            arr = np.load(io.BytesIO(sample['npy']))
            if arr.ndim == 2:
                arr = arr[:, :, None]
            elif arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
                pass
            elif arr.ndim == 3:
                arr = np.moveaxis(arr, 0, -1)
            arr = _normalize_image(arr)
        elif 'npz' in sample:
            with np.load(io.BytesIO(sample['npz'])) as data:
                if 'image' in data:
                    arr = data['image']
                else:
                    arr = data[data.files[0]]
            if arr.ndim == 2:
                arr = arr[:, :, None]
            elif arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
                pass
            elif arr.ndim == 3:
                arr = np.moveaxis(arr, 0, -1)
            arr = _normalize_image(arr)
        else:
            raise ValueError(f"Unsupported sample keys: {list(sample.keys())}")
        arr = np.moveaxis(arr, -1, 0).astype(np.float32)
        tensor = torch.from_numpy(arr)
        out = {
            'image': tensor,
            'id': key,
            'dataset': meta.get('dataset'),
            'source': meta.get('source'),
            'shape': tuple(tensor.shape),
            'path': meta.get('path'),
            'meta': meta,
        }
        if transform:
            out = transform(out)
        return out

    pipeline = (
        wds.WebDataset(shard_paths, handler=handler)
        .decode()  # raw bytes; we'll handle inside
        .map(_decode_sample)
    )

    if shuffle:
        # numbers are heuristics; pipeline-level shuffle + dataloader shuffle (False) recommended
        pipeline = pipeline.shuffle(1000)

    loader = torch.utils.data.DataLoader(
        pipeline,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=_default_collate_dict,
    )
    return loader


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _default_collate_dict(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # images -> (B,C,H,W)
    images = torch.stack([b['image'] for b in batch], dim=0)
    out: Dict[str, Any] = {
        'image': images,
        'id': [b['id'] for b in batch],
        'dataset': [b['dataset'] for b in batch],
        'source': [b['source'] for b in batch],
        'shape': [b['shape'] for b in batch],
        'path': [b['path'] for b in batch],
        'meta': [b['meta'] for b in batch],
    }
    return out


# ---------------------------------------------------------------------------
# CLI helper for quick inspection
# ---------------------------------------------------------------------------

def _cli():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Inspect processed microscopy datasets")
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--splits', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=4)
    ap.add_argument('--mode', choices=['manifest', 'wds'], default='manifest')
    ap.add_argument('--list_file', type=str, help='Shard list file (for wds mode)')
    args = ap.parse_args()

    if args.mode == 'manifest':
        ds = build_manifest_dataset(roots=args.roots, splits=args.splits)
        print(f"Loaded {len(ds)} samples from manifest mode")
        for i in range(min(args.limit, len(ds))):
            s = ds[i]
            print(f"[{i}] id={s['id']} shape={s['shape']} dataset={s['dataset']} path={s['path']}")
    else:
        loader = build_wds_dataloader(args.list_file, batch_size=1, shuffle=False)
        for i, s in enumerate(loader):
            print(f"[{i}] batch shape={s['image'].shape} ids={s['id']}")
            if i + 1 >= args.limit:
                break


if __name__ == '__main__':  # pragma: no cover
    _cli()