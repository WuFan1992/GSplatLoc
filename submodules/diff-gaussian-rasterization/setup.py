#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    version='3.2',
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            #extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            ### To fix illegal memory issue
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"), 
                                         "-I" + "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/include/", 
                                                                                              "-I" + "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt/", 
                                                                                              "-I" + "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/um/", 
                                                                                              "-I" + "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/shared/"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
