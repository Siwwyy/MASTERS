from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# # using a different name to prevent the interpreter from picking up based on directory name
# setup(
#     name="NN_Extension",
#     version="0.1",
#     install_requires=["torch", "ninja"],
#     packages=["nn_extension"],
#     package_dir={
#         "nn_extensions": "src/nn_extensions",
#     },
#     ext_modules=[
#         cpp_extension.CUDAExtension("fud_extension", ["csrc/filter_kernel.cu", "csrc/upscale.cu", "csrc/py_module.cu"]),
#     ],
#     cmdclass={"build_ext": cpp_extension.BuildExtension},
# )

setup(
    name="addKernel_cuda",
    ext_modules=[
        CUDAExtension(
            "addKernel_cuda",
            [
                "Kernels/AddKernel/addKernel_cuda.cpp",
                "Kernels/AddKernel/addKernel_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
