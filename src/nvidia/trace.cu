#pragma once
#include "helper.cu"
#include <cuda_runtime.h>

template <typename T>
__global__ void trace_kernel(const T *__restrict__ input,
                             T *__restrict__ output, int n, int cols) {
  T sum{0};

  // grid-stride loop, add partial diagonal sums
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid, len = cols + 1; i < n; i += stride)
    sum += input[i * len];

  // warp-level reduction
  sum = warp_reduce_sum(sum);

  // store warp sums in shared memory
  __shared__ T shared[32];
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  if (lane == 0)
    shared[warp_id] = sum;
  __syncthreads();

  // block-level warp reduction
  if (warp_id == 0) {
    sum = (lane < (blockDim.x + 31) / 32) ? shared[lane] : T{0};
    sum = warp_reduce_sum(sum);
    if (lane == 0)
      atomicAdd(output, sum);
  }
}