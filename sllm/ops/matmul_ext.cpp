#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <cmath>
#include <vector>

// Function to perform a single scaled matrix multiplication: C = A @ B * scale
static void matmul_scaled(const float* A, const float* B, float* C, 
                          int m, int k, int n, float scale) {
    // Initialize C to zeros
    for (int i = 0; i < m * n; ++i) {
        C[i] = 0.0f;
    }
    
    // Perform matrix multiplication with OpenMP parallelization
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum * scale;
        }
    }
}

// Exposed C function for Python to call
extern "C" {
    void scaled_matmul(float* A, float* B, float* C, int m, int k, int n, float scale) {
        matmul_scaled(A, B, C, m, k, n, scale);
    }
    
    // Bundled version to process multiple matrices at once
    void bundled_scaled_matmul(int num_pairs, float** A_ptrs, float** B_ptrs, float** C_ptrs, 
                              int* m_dims, int* k_dims, int* n_dims, float* scales) {
        #pragma omp parallel for
        for (int i = 0; i < num_pairs; ++i) {
            matmul_scaled(A_ptrs[i], B_ptrs[i], C_ptrs[i], 
                          m_dims[i], k_dims[i], n_dims[i], scales[i]);
        }
    }
}