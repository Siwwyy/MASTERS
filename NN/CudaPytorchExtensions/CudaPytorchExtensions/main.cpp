#include <iostream>

#include "Kernel.cuh"

template<bool on_cuda, typename Runnable>
void run_code(Runnable const& runnable)
{
	if constexpr (on_cuda) {
		Cuda_Kernel::test_function<Runnable, device_type::host>(runnable);
	}
	else {
		runnable();
	}
}


int main(int argc, char* argv[])
{

	double A[3]{};
	double B[3]{};
	double C[3]{};

	// Populate arrays A and B.
	A[0] = 5; A[1] = 8; A[2] = 3;
	B[0] = 7; B[1] = 6; B[2] = 4;

	// Sum array elements across ( C[0] = A[0] + B[0] ) into array C using CUDA.
	//kernel(A, B, C, 3);
	Cuda_Kernel::kernel_double(A, B, C, 3);
	
	// Print out result.
	std::cout << "C = " << C[0] << ", " << C[1] << ", " << C[2] << std::endl;


	run_code<true>([]() { printf("Called from CPP!\n");  });

	system("pause");
	return EXIT_SUCCESS;
}
