"""
Real-LiDAR data loading for the FPS-Voronoi pipeline.

The Phase 1/2/3 engine is dimension-agnostic — `cell_membership`,
`min_pairwise_distance`, `covering_radius`, `cell_occupancy`,
`delaunay_neighbors` and FPS all operate on an (N, D) tensor regardless of D.
So feeding real data is purely a loading concern: read a LiDAR frame into a
point-cloud tensor `P`, optionally keeping full 3-D (x, y, z) or projecting to a
2-D top-down (x, y) view so the existing 2-D visualisations keep working.

Two on-disk formats are supported and auto-detected by extension:

    KITTI    *.bin       float32, reshape(-1, 4)  → x, y, z, intensity
    nuScenes *.pcd.bin   float32, reshape(-1, 5)  → x, y, z, intensity, ring

(Same conventions as extract/extract_kitti.py and extract/extract_nuscenes.py.)
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

# Repo root, so callers can find datasets without hard-coding paths.
_ROOT = os.path.dirname(os.path.abspath(__file__))

_NUSCENES_LIDAR = os.path.join(_ROOT, "data", "nuscenes", "samples", "LIDAR_TOP")
_KITTI_GLOB = os.path.join(
    _ROOT, "data", "kitti", "**", "velodyne_points", "data", "*.bin"
)


def _num_columns(path: str) -> int:
    """Columns in a LiDAR binary, inferred from its extension."""
    name = os.path.basename(path).lower()
    if name.endswith(".pcd.bin"):
        return 5  # nuScenes: x, y, z, intensity, ring
    if name.endswith(".bin"):
        return 4  # KITTI: x, y, z, intensity
    raise ValueError(f"Unrecognised LiDAR file (expected .bin / .pcd.bin): {path}")


def load_lidar_bin(
    path: str,
    dims: int = 3,
    max_range: float | None = None,
    min_z: float | None = None,
) -> torch.Tensor:
    """Load one LiDAR frame into a point-cloud tensor.

    Parameters
    ----------
    path
        A KITTI ``.bin`` or nuScenes ``.pcd.bin`` file.
    dims
        ``3`` → keep (x, y, z); ``2`` → top-down (x, y) projection.
    max_range
        If set, drop points whose horizontal (x, y) radius exceeds this many
        metres. Trims the sparse long-range fringe that bloats covering radii.
    min_z
        If set, drop points with z below this height (rough ground removal).
        Applied on the original 3-D z even when ``dims == 2``.

    Returns
    -------
    torch.Tensor
        ``(N, dims)`` float32, contiguous.
    """
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3, got {dims}")

    ncols = _num_columns(path)
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % ncols != 0:
        raise ValueError(
            f"{path}: {pts.size} floats not divisible by {ncols} columns"
        )
    xyz = pts.reshape(-1, ncols)[:, :3]

    if min_z is not None:
        xyz = xyz[xyz[:, 2] >= min_z]
    if max_range is not None:
        r2 = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        xyz = xyz[r2 <= max_range * max_range]

    out = xyz if dims == 3 else xyz[:, :2]
    return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))


def list_frames(dataset: str) -> list[str]:
    """Sorted list of LiDAR frame paths for ``"nuscenes"`` or ``"kitti"``."""
    d = dataset.lower()
    if d == "nuscenes":
        paths = glob.glob(os.path.join(_NUSCENES_LIDAR, "*.pcd.bin"))
    elif d == "kitti":
        paths = glob.glob(_KITTI_GLOB, recursive=True)
    else:
        raise ValueError(f"Unknown dataset {dataset!r} (use 'nuscenes' or 'kitti')")
    return sorted(paths)


def load_frame(
    dataset: str,
    index: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Convenience: load the ``index``-th frame of a dataset by name.

    Extra keyword arguments are forwarded to :func:`load_lidar_bin`.
    """
    frames = list_frames(dataset)
    if not frames:
        raise FileNotFoundError(f"No LiDAR frames found for dataset {dataset!r}")
    if not -len(frames) <= index < len(frames):
        raise IndexError(
            f"frame index {index} out of range (dataset has {len(frames)} frames)"
        )
    return load_lidar_bin(frames[index], **kwargs)
