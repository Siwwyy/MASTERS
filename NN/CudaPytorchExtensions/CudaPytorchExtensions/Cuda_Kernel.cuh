#ifndef _CUDA_KERNEL_CUH_
#define _CUDA_KERNEL_CUH_
#pragma once

#include <iostream>
#include <cstdint>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define HostToDevice	cudaMemcpyHostToDevice
#define DeviceToHost	cudaMemcpyDeviceToHost

namespace Cuda_Kernel
{
    template<class _Ty>
    __global__ void vector_addition_kernel(_Ty* A, _Ty* B, _Ty* C, const unsigned int *arraySize)
    {
        int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    	
        if (threadID < *arraySize)
        {
            C[threadID] = A[threadID] + B[threadID];
        }
    }

    template<typename Callable>
    __global__ void test_global(Callable const & callable)
    {
        callable();
    }

	template<typename Callable>
	__host__ void test_host(Callable const& callable)
    {
        callable();
    }

    template<typename Callable>
    __device__ void test_device(Callable const& callable)
    {
        callable();
    }
	
}

#endif /* _CUDA_KERNEL_CUH_ */