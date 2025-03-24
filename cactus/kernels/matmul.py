
def scaled_matmul(matmul_bundles):
    """A & B are numpy arrays, scale is a scalar"""
    return [A @ B * scale for A, B, scale in matmul_bundles]