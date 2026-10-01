import gc
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixes random seeds across all core libraries for reproducibility.

    cudnn determinism is deliberately not forced: it slows training and the
    Transformer has no cudnn-heavy ops. Seeds alone give repeatable data order/init.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_memory() -> None:
    """Forces garbage collection and clears CUDA cache to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
