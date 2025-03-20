import numpy as np
import jax.numpy as jnp
import torch

def scaled_matmul(matmul_bundles):
    return [A @ B * scale for A, B, scale in matmul_bundles]