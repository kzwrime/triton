from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_moe_kernel_self_test',
    ext_modules=[
        CUDAExtension(
            name='fused_moe_kernel_self_test',
            sources=['wrapper_for_torch.cu'],
            # library_dirs=['/lib/x86_64-linux-gnu'],
            # libraries=['cuda'],  # 链接CUDA库
            extra_objects=[
                './build/libfused_moe_kernel_fp8.a',
            ],
            extra_compile_args={'cxx': [], 'nvcc': ['-O2']},
            extra_link_args=['-lcuda', "-mcmodel=medium"],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)