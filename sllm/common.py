import os

import torch
from platformdirs import user_cache_dir

CACHE_DIR = user_cache_dir("sllm")
WEIGHT_DIR = f"{CACHE_DIR}/weights"
GRADIENT_DIR = f"{CACHE_DIR}/gradient_checkpoints"
os.makedirs(GRADIENT_DIR, exist_ok=True)

MAX_CONCURRENT_THREADS = os.cpu_count() * 2
MINI_BATCH_SIZE = 2

DTYPE = torch.float32
MAX_GRAD_NORM = 0.01

os.environ["TOKENIZERS_PARALLELISM"] = "true"
