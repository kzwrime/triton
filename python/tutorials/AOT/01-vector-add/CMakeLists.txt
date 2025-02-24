cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(AddKernelFp32 VERSION 1.0 LANGUAGES C CXX CUDA)

# 设置C和CUDA的标准
set(CMAKE_C_STANDARD 99)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit REQUIRED)
add_compile_options(-fPIC)
# add_compile_options(-rdc=true)

# 自动扫描 fused_moe_kernel_fp8 目录下的源文件和头文件
file(GLOB SOURCES "kernels/*.c" "kernels/*.cu")
# file(GLOB HEADERS "kernels/fused_moe_kernel_fp8/*.h")

# 打印找到的文件，用于调试
message(STATUS "Sources: ${SOURCES}")
# message(STATUS "Headers: ${HEADERS}")

# 创建一个库目标
add_library(add_kernel_fp32
            STATIC
            # SHARED
            ${SOURCES} 
            add_kernel.c
            # ${HEADERS}
)

# 指定库的头文件目录
target_include_directories(add_kernel_fp32 PUBLIC 
                           ${CMAKE_CURRENT_SOURCE_DIR}/kernels
                           ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接CUDA库
# target_link_libraries(fused_moe_kernel_fp8 PRIVATE CUDA)
target_link_libraries(add_kernel_fp32 PRIVATE CUDA::cudart)

# 如果你的库需要其他CUDA库，可以在这里添加
# target_link_libraries(fused_moe_kernel_fp8 PRIVATE CUDA::cublas CUDA::curand)

# 如果你的项目有可执行文件，可以在这里添加
# add_executable(main main.c)
# target_link_libraries(main PRIVATE fused_moe_kernel_fp8)