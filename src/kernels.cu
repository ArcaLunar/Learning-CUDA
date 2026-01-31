#include <cassert>
#include <cuda_fp16.h>
#include <vector>

#include "../tester/utils.h"

/**
 * @brief Performs warp-level reduction to compute the sum of values.
 */
template <typename T> __inline__ __device__ T warp_reduce_sum(T v) {
#pragma unroll
  for (int i = 16; i > 0; i >>= 1)
    v += __shfl_down_sync(0xffffffff, v, i);

  return v;
}

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

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T> &h_input, size_t rows, size_t cols) {
  size_t n = (rows < cols) ? rows : cols;

  // std::vector<T> diag(n);
  // for (size_t i = 0, stride = cols + 1; i < n; ++i)
  //   diag[i] = h_input[i * stride];

  int block = 256;
  int grid = (n + block - 1) / block;

  T *d_input;
  T *d_output;
  T h_output = 0;
  cudaMalloc(&d_input, sizeof(T) * rows * cols);
  cudaMalloc(&d_output, sizeof(T));

  cudaMemcpy(d_input, h_input.data(), sizeof(T) * rows * cols,
             cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, sizeof(T));

  trace_kernel<<<grid, block>>>(d_input, d_output, n, cols);
  cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
  return h_output;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads,
 * head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len,
 * query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query
 * attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int> &, size_t, size_t);
template float trace<float>(const std::vector<float> &, size_t, size_t);
template void flashAttention<float>(const std::vector<float> &,
                                    const std::vector<float> &,
                                    const std::vector<float> &,
                                    std::vector<float> &, int, int, int, int,
                                    int, int, bool);
template void flashAttention<half>(const std::vector<half> &,
                                   const std::vector<half> &,
                                   const std::vector<half> &,
                                   std::vector<half> &, int, int, int, int, int,
                                   int, bool);
