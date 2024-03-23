

#include "Kernel.cuh"



void Cuda_Kernel::kernel_double(double* A, double* B, double* C, const std::size_t& array_size)
{
	//kernel<double>(A, B, C, array_size);

    double* d_A, * d_B, * d_C;
    unsigned int* d_ArraySize;


    cudaMalloc((void**)&d_A        , array_size * sizeof(double));
    cudaMalloc((void**)&d_B        , array_size * sizeof(double));
    cudaMalloc((void**)&d_C        , array_size * sizeof(double));
    cudaMalloc((void**)&d_ArraySize, sizeof(unsigned int));


    cudaMemcpy(d_A        , A, array_size * sizeof(double), HostToDevice);
    cudaMemcpy(d_B        , B, array_size * sizeof(double), HostToDevice);
    cudaMemcpy(d_ArraySize, &array_size, sizeof(unsigned int), HostToDevice);


    dim3 blockSize(512, 1, 1);
    dim3 gridSize(512 / array_size + 1, 1);

    vector_addition_kernel<double> <<<10, 10, 10>>> (d_A, d_B, d_C, d_ArraySize);


    cudaMemcpy(C, d_C, array_size * sizeof(double), DeviceToHost);

    //Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_ArraySize);
}

//template<typename Runnable>
//void Cuda_Kernel::test_function(const Runnable& runnable)
//{
//	
//}