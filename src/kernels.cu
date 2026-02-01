#include "../tester/utils.h"
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

#include "nvidia/attn.cu"
#include "nvidia/helper.cu"
#include "nvidia/trace.cu"

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
  auto qsize = batch_size * target_seq_len * query_heads * head_dim;
  auto ksize = batch_size * src_seq_len * kv_heads * head_dim;
  auto vsize = batch_size * src_seq_len * kv_heads * head_dim;
  auto osize = batch_size * target_seq_len * query_heads * head_dim;

  h_o.resize(osize);

  T *d_q;
  T *d_k;
  T *d_v;
  T *d_o;

  cudaCheck(cudaMalloc(&d_q, qsize * sizeof(T)));
  cudaCheck(cudaMalloc(&d_k, ksize * sizeof(T)));
  cudaCheck(cudaMalloc(&d_v, vsize * sizeof(T)));
  cudaCheck(cudaMalloc(&d_o, osize * sizeof(T)));

  cudaCheck(
      cudaMemcpy(d_q, h_q.data(), qsize * sizeof(T), cudaMemcpyHostToDevice));
  cudaCheck(
      cudaMemcpy(d_k, h_k.data(), ksize * sizeof(T), cudaMemcpyHostToDevice));
  cudaCheck(
      cudaMemcpy(d_v, h_v.data(), vsize * sizeof(T), cudaMemcpyHostToDevice));

  // Launch the flash attention kernel here (not implemented in this snippet)
  dim3 grid(batch_size, query_heads, target_seq_len);
  int threads = 128;

  constexpr int TILE_N = 32;
  constexpr int MAX_D = 128;
  assert(head_dim <= MAX_D);

  auto smem_bytes = 2 * TILE_N * MAX_D * sizeof(T);

  flashattn<T, TILE_N, MAX_D><<<grid, threads, smem_bytes>>>(
      d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len, query_heads,
      kv_heads, head_dim, is_causal);

  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());

  cudaCheck(
      cudaMemcpy(h_o.data(), d_o, osize * sizeof(T), cudaMemcpyDeviceToHost));

  cudaCheck(cudaFree(d_q));
  cudaCheck(cudaFree(d_k));
  cudaCheck(cudaFree(d_v));
  cudaCheck(cudaFree(d_o));
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
