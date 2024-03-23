#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace 
{
    template <typename scalar_t>
    __device__ __forceinline__ scalar_t add(scalar_t a, scalar_t b) {
        return a + b;
    }

    template <typename scalar_t>
    __global__ void addKernel_cuda_kernel(
        const   at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, size_t> inputA,
        const   at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, size_t> inputB,
                at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, size_t> resultTensor)
    {
        //batch index
        //const int n = blockIdx.y;
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;

        resultTensor[0][0][c] = add(inputA[0][0][c], inputB[0][0][c]);
    }

} // namespace

std::vector<at::Tensor> addKernel_cuda(
    at::Tensor inputA,
    at::Tensor inputB)
{
    //Check which tensor is bigger
    const auto maxLength = inputA.size(-1) > inputB.size(-1) ? inputA.size(-1) : inputB.size(-1);

    //Result tensor initialization
    auto resultTensor = inputA.size(-1) > inputB.size(-1) ? at::zeros_like(inputA) : at::zeros_like(inputB);

    //Dispatch
    const int threads = 1024;
    const dim3 blocks((resultTensor.size(-1) + threads - 1) / threads, 1);

    //Create dispatch
    AT_DISPATCH_FLOATING_TYPES(resultTensor.type(), "addKernel_cuda", ([&] {
        addKernel_cuda_kernel<scalar_t> << <blocks, threads >> > (
            inputA.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, size_t>(),
            inputB.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, size_t>(),
            resultTensor.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, size_t>());
    }));

    return { resultTensor };
}