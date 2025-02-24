// nvcc test_add_kernel.cu ./add_kernel.c ./kernels/*.c -lcuda

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <typeinfo>

#include "add_kernel.h"

#define CUDA_CHECK(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char *err_str; \
            cuGetErrorString(err, &err_str); \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << err_str << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template <typename T>
__global__ void vec_add_kernel(const T* vecA, const T* vecB, T* vecC, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vecC[idx] = vecA[idx] + vecB[idx];
    }
}

template <typename T>
void print_vector(const std::vector<T>& vec, const std::string& name, int head = 5, int tail = 5) {
    std::cout << name << " (first " << head << " and last " << tail << " elements):" << std::endl;
    for (int i = 0; i < head && i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << "... ";
    for (int i = std::max(static_cast<int>(vec.size()) - tail, 0); i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void vec_add(int n) {
    std::cout << "Initializing vectors of size " << n << " with type " << typeid(T).name() << std::endl;

    // Allocate host memory
    std::vector<T> h_vecA(n), h_vecB(n), h_vecC(n), h_vecC_ref(n);
    for (int i = 0; i < n; ++i) {
        h_vecA[i] = static_cast<T>(i);
        h_vecB[i] = static_cast<T>(i * 2);
        h_vecC_ref[i] = h_vecA[i] + h_vecB[i];  // Precompute reference result on CPU
    }

    // Print input vectors
    print_vector(h_vecA, "A");
    print_vector(h_vecB, "B");

    // Allocate device memory
    std::cout << "Allocating device memory..." << std::endl;
    CUdeviceptr d_vecA, d_vecB, d_vecC;
    CUDA_CHECK(cuMemAlloc(&d_vecA, n * sizeof(T)));
    CUDA_CHECK(cuMemAlloc(&d_vecB, n * sizeof(T)));
    CUDA_CHECK(cuMemAlloc(&d_vecC, n * sizeof(T)));

    // Copy data to device
    std::cout << "Copying data to device..." << std::endl;
    CUDA_CHECK(cuMemcpyHtoD(d_vecA, h_vecA.data(), n * sizeof(T)));
    CUDA_CHECK(cuMemcpyHtoD(d_vecB, h_vecB.data(), n * sizeof(T)));

    // Launch kernel
    std::cout << "Launching kernel..." << std::endl;
    int threadsPerBlock = 128;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    // vec_add_kernel<T><<<blocksPerGrid, threadsPerBlock>>>((const T*)d_vecA, (const T*)d_vecB, (T*)d_vecC, n);
    CUDA_CHECK(add_kernel_dispatcher_with_grid(
        cudaStreamPerThread, d_vecA, d_vecB, d_vecC, n, threadsPerBlock, 4, 2, blocksPerGrid, 1, 1)
    );

    // Check for kernel launch errors
    CUDA_CHECK(cuCtxSynchronize());
    std::cout << "Kernel execution completed." << std::endl;

    // Copy result back to host
    std::cout << "Copying result back to host..." << std::endl;
    CUDA_CHECK(cuMemcpyDtoH(h_vecC.data(), d_vecC, n * sizeof(T)));

    // Print reference and CUDA results
    print_vector(h_vecC_ref, "Reference result (A + B)");
    print_vector(h_vecC, "CUDA result (A + B)");

    // Verify the result
    std::cout << "Verifying result..." << std::endl;
    for (int i = 0; i < n; ++i) {
        if (h_vecC[i] != h_vecC_ref[i]) {
            std::cerr << "Error at index " << i << ": expected " << h_vecC_ref[i] << ", got " << h_vecC[i] << std::endl;
            std::cerr << "A[" << i << "] = " << h_vecA[i] << ", B[" << i << "] = " << h_vecB[i] << std::endl;
            break;
        }
    }
    std::cout << "Result verification completed." << std::endl;

    // Free device memory
    std::cout << "Freeing device memory..." << std::endl;
    CUDA_CHECK(cuMemFree(d_vecA));
    CUDA_CHECK(cuMemFree(d_vecB));
    CUDA_CHECK(cuMemFree(d_vecC));
}

int main() {
    // Initialize CUDA
    std::cout << "Initializing CUDA..." << std::endl;
    CUDA_CHECK(cuInit(0));

    // Create CUDA context
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, 0, device));

    // Print the selected device
    char selectedDeviceName[256];
    CUDA_CHECK(cuDeviceGetName(selectedDeviceName, sizeof(selectedDeviceName), device));
    std::cout << "Selected Device: " << selectedDeviceName << " (Device ID: 0)" << std::endl;

    std::cout << "load_vector_add..." << std::endl;
    load_add_kernel();
    std::cout << "load_vector_add finished" << std::endl;

    // Set vector length and type
    int n = 100000;
    vec_add<float>(n);  // Example with float type

    // Destroy CUDA context
    std::cout << "Destroying CUDA context..." << std::endl;
    CUDA_CHECK(cuCtxDestroy(context));

    std::cout << "Program completed successfully." << std::endl;
    return 0;
}