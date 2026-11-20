from __future__ import annotations

import random
import numpy as np
import torch


def set_reproducibility(seed: int) -> None:
    """Seed the relevant random number generators.

    TODO: complete this function so that Python, NumPy and PyTorch are controlled. Improve CUDA/cuDNN repeatability when CUDA is available.
    """
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # TODO: PyTorch CPU
    torch.manual_seed(seed)
    # TODO: CUDA, if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # TODO: cuDNN deterministic / benchmark flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker seed."""
    # implement worker seeding for NumPy and Python random.
    seed_initialize = torch.initial_seed() % 2**32
    random.seed(seed_initialize)
    np.random.seed(seed_initialize)


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded torch.Generator used for deterministic shuffling."""
    generator = torch.Generator()
    return generator.manual_seed(seed)
