import gc
import torch
from concurrent.futures import ThreadPoolExecutor

def scaled_matmul(matmul_bundles):
    results = [] 
    futures_map = {}
    bundles_data = {} 

    with ThreadPoolExecutor() as executor:

        for bundle_idx, bundle in enumerate(matmul_bundles):
            original_A, B, scale = bundle
            A_shape, B_shape = original_A.shape, B.shape

            result_shape = A_shape[:-1] + B_shape[-1:]
            A_reshaped = original_A.reshape(-1, A_shape[-2], A_shape[-1])
            B_reshaped = B.reshape(-1, B_shape[-2], B_shape[-1])
            batch = A_reshaped.shape[0]

            C = torch.empty((batch, A_reshaped.shape[1], B_shape[-1]), dtype=torch.float32) 
            bundles_data[bundle_idx] = (C, result_shape, original_A)

            for batch_idx in range(batch):
                b_tensor = B_reshaped[0] if B_reshaped.shape[0] == 1 else B_reshaped[batch_idx]
                future = executor.submit(remote_matmul, A_reshaped[batch_idx], b_tensor, scale)
                futures_map[future] = (bundle_idx, batch_idx) 

    for future in futures_map:
        bundle_idx, batch_idx = futures_map[future]
        C, _, _ = bundles_data[bundle_idx] 
        try:
            C[batch_idx] = future.result() 
        except Exception as e:
            print(f"Error fetching result for bundle {bundle_idx}, batch {batch_idx}: {e}")
            raise e

    for bundle_idx in sorted(bundles_data.keys()):
        C, result_shape, original_A = bundles_data[bundle_idx]
        final_result = C.reshape(result_shape)
        results.append(final_result)

    gc.collect()
    return results

def remote_matmul(A, B, scale):
    return torch.matmul(A, B) * scale