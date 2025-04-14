import numpy as np
from typing import List, Tuple, Optional
import os
import ctypes
from multiprocessing import cpu_count

# Load the C++ extension
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_matmul_ext.so')
try:
    _matmul_lib = ctypes.CDLL(_lib_path)
    _USING_CPP_EXTENSION = True
except (OSError, ImportError):
    _USING_CPP_EXTENSION = False
    print("Warning: C++ extension for bundled_scaled_matmul not found, falling back to Python implementation")

def bundled_scaled_matmul(
    matrix_pairs: List[Tuple[np.ndarray, np.ndarray]], 
    scales: Optional[List[float]] = None
) -> List[np.ndarray]:
    """
    Perform bundled scaled matrix multiplications in parallel using C++ extension.
    
    This function takes pairs of matrices and optional scaling factors,
    computes A @ B * scale for each triplet (A, B, scale),
    and returns the list of results. C++ extension bypasses GIL and utilizes
    multiple cores for parallel execution.
    
    Args:
        matrix_pairs: List of tuples, each containing two matrices (A, B) to multiply
        scales: Optional list of scaling factors for each multiplication.
                If None, scaling factor of 1.0 is used for all multiplications.
    
    Returns:
        List of matrix multiplication results
    """
    if scales is None:
        scales = [1.0] * len(matrix_pairs)
    
    assert len(matrix_pairs) == len(scales), "Number of matrix pairs must match number of scaling factors"
    
    # If the C++ extension is available, use it for parallel execution
    if _USING_CPP_EXTENSION:
        return _bundled_matmul_cpp(matrix_pairs, scales)
    else:
        # Fallback to Python implementation
        return _bundled_matmul_python(matrix_pairs, scales)

def _bundled_matmul_cpp(matrix_pairs, scales):
    """Use the C++ extension to perform bundled scaled matrix multiplication."""
    # Convert all matrices to contiguous float32 arrays for C++ compatibility
    contiguous_pairs = []
    for (A, B) in matrix_pairs:
        A_cont = np.ascontiguousarray(A, dtype=np.float32)
        B_cont = np.ascontiguousarray(B, dtype=np.float32)
        contiguous_pairs.append((A_cont, B_cont))
    
    # Prepare data for C++ function
    results = []
    for (A, B), scale in zip(contiguous_pairs, scales):
        # Ensure matrices can be multiplied
        assert A.shape[1] == B.shape[0], f"Matrix shapes incompatible for multiplication: {A.shape} and {B.shape}"
        
        # Create output array
        C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
        
        # Call C++ function
        _matmul_lib.scaled_matmul(
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(A.shape[0]),
            ctypes.c_int(A.shape[1]),
            ctypes.c_int(B.shape[1]),
            ctypes.c_float(scale)
        )
        
        results.append(C)
    
    return results

def _bundled_matmul_python(matrix_pairs, scales):
    """Python implementation as fallback."""
    results = []
    for (A, B), scale in zip(matrix_pairs, scales):
        # Ensure matrices can be multiplied
        assert A.shape[1] == B.shape[0], f"Matrix shapes incompatible for multiplication: {A.shape} and {B.shape}"
        result = A @ B * scale
        results.append(result)
    
    return results