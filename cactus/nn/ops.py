import torch

from cactus.kernels import scaled_matmul


def cactus_matmul(bundles):
    return scaled_matmul(bundles)


#     numpy_bundles = convert_bundles_to_numpy(bundles)
#     results = scaled_matmul(numpy_bundles)
#     return convert_results_to_torch(results)

# def convert_bundles_to_numpy(bundles):
#     return [(A.detach().numpy(), B.detach().numpy(), scale) for A, B, scale in bundles]

# def convert_results_to_torch(bundles):
#     return [torch.from_numpy(C) for C in bundles]
