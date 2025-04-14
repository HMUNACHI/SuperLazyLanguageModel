/**
 * @file matmul.cpp
 * @brief PyTorch extension module for performing scaled matrix multiplication.
 *
 * This file implements functions to perform matrix multiplications with a scaling factor.
 * It provides both a simple operation and a bundled version that can execute multiple operations
 * concurrently using OpenMP for batch processing. The module is exposed to Python via PyBind11.
 */

 #include <torch/extension.h>
 #include <pybind11/pybind11.h>
 #include <vector>
 #ifdef _OPENMP
     // OpenMP header included when available.
 #endif
 
 namespace py = pybind11;
 
 /**
  * @brief Performs scaled matrix multiplication.
  *
  * Computes the matrix product of A and B using torch::matmul and then scales the resulting tensor
  * by the provided scale factor.
  *
  * @param A A torch::Tensor representing the first matrix.
  * @param B A torch::Tensor representing the second matrix.
  * @param scale A double representing the scaling factor to be applied to the product.
  * @return torch::Tensor The scaled result of the matrix multiplication (i.e., (A * B) * scale).
  */
 torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B, double scale) {
     return torch::matmul(A, B) * scale;
 }
 
 /**
  * @brief Executes a set of scaled matrix multiplications bundled together.
  *
  * Processes a vector of tuples where each tuple (referred to as a "bundle") contains three elements:
  *  - The first element is a torch::Tensor representing the matrix A.
  *  - The second element is a torch::Tensor representing the matrix B.
  *  - The third element is a double specifying the scaling factor for the multiplication.
  *
  * The function reshapes each input tensor to separate batch dimensions from the matrix dimensions,
  * then performs the scaled matrix multiplication over the batch elements. When B_reshaped has a single
  * batch element, it is broadcasted across the entire batch of A_reshaped. OpenMP is used for parallelization
  * over the batch dimension if it is available.
  *
  * The final resulting tensor is reshaped to match the original input dimensions (except for the last dimension
  * which becomes the last dimension of B). The function returns a vector of results with each result corresponding
  * to a bundle in the input.
  *
  * @param matmul_bundles A vector of py::tuple objects, each containing:
  *        - A torch::Tensor for matrix A,
  *        - A torch::Tensor for matrix B,
  *        - A double value for the scaling factor.
  * @return std::vector<torch::Tensor> A vector containing the resulting tensors after performing the
  *         bundled scaled matrix multiplications.
  */
 std::vector<torch::Tensor> bundled_scaled_matmul(const std::vector<py::tuple>& matmul_bundles) {
     std::vector<torch::Tensor> results;
 
     for (const auto& bundle : matmul_bundles) {
         auto original_A = bundle[0].cast<torch::Tensor>();
         auto B = bundle[1].cast<torch::Tensor>();
         double scale = bundle[2].cast<double>();
 
         auto A_shape = original_A.sizes();
         auto B_shape = B.sizes();
 
         // Reshape to isolate the batch dimension from matrix dimensions.
         auto A_reshaped = original_A.reshape({-1, A_shape[A_shape.size()-2], A_shape[A_shape.size()-1]});
         auto B_reshaped = B.reshape({-1, B_shape[B_shape.size()-2], B_shape[B_shape.size()-1]});
         int64_t batch = A_reshaped.size(0);
         auto C = torch::empty({batch, A_reshaped.size(1), B_reshaped.size(2)}, torch::kFloat32);
 
         // Parallelize over the batch dimension using OpenMP.
         #pragma omp parallel for
         for (int64_t i = 0; i < batch; ++i) {
             auto A_i = A_reshaped[i];
             // Use the same B for all if B_reshaped only has one element; otherwise, index accordingly.
             auto b_tensor = (B_reshaped.size(0) == 1) ? B_reshaped[0] : B_reshaped[i];
             C[i] = matmul(A_i, b_tensor, scale);
         }
 
         // Restore the original batch shape with the new matrix multiplication dimensions.
         std::vector<int64_t> result_shape;
         for (size_t j = 0; j < A_shape.size()-1; ++j) {
             result_shape.push_back(A_shape[j]);
         }
         result_shape.push_back(B_shape[B_shape.size()-1]);
         results.push_back(C.reshape(result_shape));
     }
     return results;
 }
 
 /**
  * @brief PyBind11 module definition for the matrix multiplication extension.
  *
  * Exposes the bundled_scaled_matmul function to Python, allowing it to be called as a regular Python function.
  * The module is named "matmul".
  */
 PYBIND11_MODULE(matmul, m) {
     m.def("bundled_scaled_matmul", &bundled_scaled_matmul, "Distributes bundles of scaled matrix multiplication operations to remote devices concurrently");
 }
 