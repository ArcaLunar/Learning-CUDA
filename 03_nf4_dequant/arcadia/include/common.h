#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>

// CUDA error checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// CUDA kernel launch error checking
#define CHECK_KERNEL()                                                        \
    do {                                                                      \
        cudaError_t error = cudaGetLastError();                               \
        if (error != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA kernel error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(error));           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Type aliases for clarity
using fp16 = __half;
using bf16 = __nv_bfloat16;

// Utility: Convert uint16_t (bit representation) to half
inline __host__ __device__ __half uint16_to_half(uint16_t val) {
    __half h;
    *reinterpret_cast<uint16_t*>(&h) = val;
    return h;
}

// Utility: Convert half to uint16_t (bit representation)
inline __host__ __device__ uint16_t half_to_uint16(__half h) {
    return *reinterpret_cast<uint16_t*>(&h);
}

#endif // COMMON_H
