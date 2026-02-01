#pragma once

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T> __device__ __forceinline__ float val_to_float(T val);

template <> __device__ __forceinline__ float val_to_float<float>(float val) {
  return val;
}

template <> __device__ __forceinline__ float val_to_float<half>(half val) {
  return __half2float(val);
}

template <typename T> __device__ __forceinline__ T float_to_val(float val);

template <> __device__ __forceinline__ float float_to_val<float>(float val) {
  return val;
}

template <> __device__ __forceinline__ half float_to_val<half>(float val) {
  return __float2half(val);
}

template <typename T, int Tr, int MAX_D>
__global__ void flashattn(const T *__restrict__ Q, const T *__restrict__ K,
                          const T *__restrict__ V, T *__restrict__ O,
                          int batch_size, int target_seq_len, int src_seq_len,
                          int query_heads, int kv_heads, int head_dim,
                          bool is_causal) {
  // Tile size for K/V. Assumed 32 to match host code smem allocation.
  const int Tc = 32;

  // Shared memory for K and V tiles
  // Size: 2 * Tc * MAX_D * sizeof(float)
  extern __shared__ float smem[];
  float *K_smem = smem;
  float *V_smem = smem + Tc * MAX_D;

  int tx = threadIdx.x; // 0..Tr-1
  int bx = blockIdx.x;
  int by = blockIdx.y; // Q tile index

  // Dimensions
  int b = bx / query_heads;
  int h = bx % query_heads;

  // KV head
  int kv_group_size = query_heads / kv_heads;
  int h_kv = h / kv_group_size;

  // Global strides
  int stride_Q_S = query_heads * head_dim;
  int stride_Q_B = target_seq_len * stride_Q_S;

  int stride_K_S = kv_heads * head_dim;
  int stride_K_B = src_seq_len * stride_K_S; // V shares same layout

  // Base pointers
  const T *q_ptr_base = Q + b * stride_Q_B + h * head_dim;
  const T *k_ptr_base = K + b * stride_K_B + h_kv * head_dim;
  const T *v_ptr_base = V + b * stride_K_B + h_kv * head_dim;
  T *o_ptr_base = O + b * stride_Q_B + h * head_dim;

  // Q row for this thread
  int q_row_global = by * Tr + tx;
  bool valid_q = q_row_global < target_seq_len;

  // Load Q row into registers
  float q_reg[MAX_D]; // Register usage might be high
  if (valid_q) {
    for (int d = 0; d < head_dim; ++d) {
      q_reg[d] = val_to_float(*(q_ptr_base + q_row_global * stride_Q_S + d));
    }
  } else {
    for (int d = 0; d < MAX_D; ++d)
      q_reg[d] = 0.0f;
  }

  // Accumulators
  float o_reg[MAX_D];
  for (int d = 0; d < MAX_D; ++d)
    o_reg[d] = 0.0f;

  float m = -INFINITY;
  float l = 0.0f;

  float scale = 1.0f / sqrtf((float)head_dim);

  // Loop over K/V blocks
  for (int j = 0; j < src_seq_len; j += Tc) {
    int valid_wc = min(Tc, src_seq_len - j);

    // Cooperative Load K and V into Smem
    // Each thread loads one row of K/V tile (since blockDim.x == Tc == 32)
    // Row `tx` in tile corresponds to `j + tx` in global
    int k_row_global = j + tx;
    bool valid_k_load = (tx < valid_wc);

    if (valid_k_load) {
      for (int d = 0; d < head_dim; ++d) {
        K_smem[tx * MAX_D + d] =
            val_to_float(*(k_ptr_base + k_row_global * stride_K_S + d));
        V_smem[tx * MAX_D + d] =
            val_to_float(*(v_ptr_base + k_row_global * stride_K_S + d));
      }
    }
    __syncthreads();

    // Compute Attention
    if (valid_q) {
      float scores[Tc];
      float local_max = -INFINITY;

      for (int k = 0; k < valid_wc; ++k) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          score += q_reg[d] * K_smem[k * MAX_D + d];
        }
        score *= scale;

        // Causal Mask
        if (is_causal) {
          if (j + k > q_row_global) {
            score = -INFINITY;
          }
        }
        scores[k] = score;
        local_max = fmaxf(local_max, score);
      }

      // Online Softmax update
      float m_new = fmaxf(m, local_max);

      float exp_m_diff = expf(m - m_new);

      float block_sum = 0.0f;
      for (int k = 0; k < valid_wc; ++k) {
        float val = scores[k];
        float p = expf(val - m_new);
        scores[k] = p; // Store P_ij
        block_sum += p;
      }

      // Update l
      l = l * exp_m_diff + block_sum;

      // Update O
      for (int d = 0; d < head_dim; ++d) {
        float pv = 0.0f;
        for (int k = 0; k < valid_wc; ++k) {
          pv += scores[k] * V_smem[k * MAX_D + d];
        }
        o_reg[d] = o_reg[d] * exp_m_diff + pv;
      }

      m = m_new;
    }
    __syncthreads();
  }

  // Write O to global
  if (valid_q) {
    for (int d = 0; d < head_dim; ++d) {
      float res = (l > 0.0f) ? (o_reg[d] / l) : 0.0f;
      *(o_ptr_base + q_row_global * stride_Q_S + d) = float_to_val<T>(res);
    }
  }
}
