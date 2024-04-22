# from torch.utils.cpp_extension import load

# addKernel_cuda = load(
#     'addKernel_cuda', ['addKernel_cuda.cpp', 'addKernel_cuda_kernel.cu'], verbose=True)

# help(addKernel_cuda)


import torch
import math

# import addKernel_cuda

torch.manual_seed(42)

# a = torch.zeros(1, 1, 10, device="cuda:0")
# b = torch.ones(1, 1, 10, device="cuda:0")
# b[0, 0, 5] = 50
# print(addKernel_cuda.addKernel(a, b))
