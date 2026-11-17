"""Shared helpers for ImplicitMABSA: seeding, device selection, logging, checkpoint I/O."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

DEFAULT_CHECKPOINT_PATH = "./saved_models/implicit_mabsa.pt"
CHECKPOINT_FORMAT_VERSION = 1


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a concise, consistent console logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return the requested device, or the best available one (CUDA > MPS > CPU)."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, enabled: bool) -> torch.autocast:
    """fp16 autocast that is a no-op unless running on CUDA with ``enabled=True``."""
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(enabled and device.type == "cuda"),
    )


def save_checkpoint(payload: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Persist a checkpoint dictionary, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint written by ``train.py`` (uses ``weights_only=True`` for safety)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run `python train.py` first.")
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    version = ckpt.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {version!r}; expected {CHECKPOINT_FORMAT_VERSION}."
        )
    return ckpt
