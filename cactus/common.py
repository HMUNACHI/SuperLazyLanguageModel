import os

import torch
from platformdirs import user_cache_dir

CACHE_DIR = user_cache_dir("cactus")
WEIGHT_DIR = f"{CACHE_DIR}/weights"
GRADIENT_DIR = f"{CACHE_DIR}/gradient_checkpoints"
os.makedirs(GRADIENT_DIR, exist_ok=True)

MAX_CONCURRENT_THREADS = os.cpu_count() * 2
MINI_BATCH_SIZE = 2

CACTUS_TOKEN = os.getenv("CACTUS_TOKEN")
assert CACTUS_TOKEN is not None, "Please set the CACTUS_TOKEN environment variable."

DTYPE = torch.float32
MAX_GRAD_NORM = 0.01

os.environ["TOKENIZERS_PARALLELISM"] = "true"
