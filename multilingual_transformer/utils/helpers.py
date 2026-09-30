import gc
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Fixes random seeds across all core libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def free_memory() -> None:
    """Forces garbage collection and clears CUDA cache to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()