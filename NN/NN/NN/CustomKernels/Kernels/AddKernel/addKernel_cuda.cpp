#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> addKernel_cuda(
    torch::Tensor inputA,
    torch::Tensor inputB);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> addKernel(
    torch::Tensor inputA,
    torch::Tensor inputB)
{
    CHECK_INPUT(inputA);
    CHECK_INPUT(inputB);
    return addKernel_cuda(inputA, inputB);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("addKernel", &addKernel, "Add Kernel (CUDA)");
}
