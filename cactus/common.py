import os
from platformdirs import user_cache_dir

CACHE_DIR = user_cache_dir("cactus")
WEIGHT_DIR = f"{CACHE_DIR}/weights"
GRADIENT_DIR = f"{CACHE_DIR}/gradient_checkpoints"
os.makedirs(GRADIENT_DIR, exist_ok=True)

MAX_CONCURRENT_THREADS = os.cpu_count() * 2