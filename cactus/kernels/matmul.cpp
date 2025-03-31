#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#ifdef _OPENMP
#endif

namespace py = pybind11;

torch::Tensor remote_matmul(const torch::Tensor& A, const torch::Tensor& B, double scale) {
    return torch::matmul(A, B) * scale;
}

std::vector<torch::Tensor> scaled_matmul(const std::vector<py::tuple>& matmul_bundles) {
    std::vector<torch::Tensor> results;

    for (const auto& bundle : matmul_bundles) {
        auto original_A = bundle[0].cast<torch::Tensor>();
        auto B = bundle[1].cast<torch::Tensor>();
        double scale = bundle[2].cast<double>();

        auto A_shape = original_A.sizes();
        auto B_shape = B.sizes();

        auto A_reshaped = original_A.reshape({-1, A_shape[A_shape.size()-2], A_shape[A_shape.size()-1]});
        auto B_reshaped = B.reshape({-1, B_shape[B_shape.size()-2], B_shape[B_shape.size()-1]});
        int64_t batch = A_reshaped.size(0);
        auto C = torch::empty({batch, A_reshaped.size(1), B_reshaped.size(2)}, torch::kFloat32);

        #pragma omp parallel for
        for (int64_t i = 0; i < batch; ++i) {
            auto A_i = A_reshaped[i];
            auto b_tensor = (B_reshaped.size(0) == 1) ? B_reshaped[0] : B_reshaped[i];
            C[i] = remote_matmul(A_i, b_tensor, scale);
        }

        std::vector<int64_t> result_shape;
        for (size_t j = 0; j < A_shape.size()-1; ++j) {
            result_shape.push_back(A_shape[j]);
        }
        result_shape.push_back(B_shape[B_shape.size()-1]);
        results.push_back(C.reshape(result_shape));
    }
    return results;
}

PYBIND11_MODULE(matmul, m) {
    m.def("scaled_matmul", &scaled_matmul, "Distributes bundels of matmul operations to remote devices concurrently");
}
