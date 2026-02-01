#pragma once
#include <cuda_runtime.h>

#define cudaCheck(x)                                                           \
  do {                                                                         \
    cudaError_t err = x;                                                       \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * @brief Performs warp-level reduction to compute the sum of values.
 */
template <typename T> __inline__ __device__ T warp_reduce_sum(T v) {
#pragma unroll
  for (int i = 16; i > 0; i >>= 1)
    v += __shfl_down_sync(0xffffffff, v, i);

  return v;
}

template <typename T> __inline__ __device__ T warp_reduce_max(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}