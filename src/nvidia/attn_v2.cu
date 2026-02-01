#pragma once

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template <typename T>
__device__ __forceinline__ float val_to_float(T val);

template <>
__device__ __forceinline__ float val_to_float<float>(float val) {
  return val;
}

template <>
__device__ __forceinline__ float val_to_float<half>(half val) {
  return __half2float(val);
}

template <typename T>
__device__ __forceinline__ T float_to_val(float val);

template <>
__device__ __forceinline__ float float_to_val<float>(float val) {
  return val;
}

template <>
__device__ __forceinline__ half float_to_val<half>(float val) {
  return __float2half_rn(val);
}

template <typename T, int Tr, int MAX_D>
__global__ void flashattn(const T *__restrict__ Q, const T *__restrict__ K,
                          const T *__restrict__ V, T *__restrict__ O,
                          int batch_size, int target_seq_len, int src_seq_len,
                          int query_heads, int kv_heads, int head_dim,
                          bool is_causal) {
  const int Tc = 32;

  extern __shared__ float smem[];
  float *K_smem = smem;
  float *V_smem = smem + Tc * MAX_D;

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int b = bx / query_heads;
  int h = bx % query_heads;
  
  int kv_group_size = query_heads / kv_heads;
  int h_kv = h / kv_group_size;

  int stride_Q_S = query_heads * head_dim;
  int stride_Q_B = target_seq_len * stride_Q_S;
  
  int stride_K_S = kv_heads * head_dim;
  int stride_K_B = src_seq_len * stride_K_S;

  const T *q_ptr_base = Q + b * stride_Q_B + h * head_dim;
  const T *k_ptr_base = K + b * stride_K_B + h_kv * head_dim;
  const T *v_ptr_base = V + b * stride_K_B + h_kv * head_dim;
  T *o_ptr_base = O + b * stride_Q_B + h * head_dim;

  int q_row_global = by * Tr + tx;
  bool valid_q = q_row_global < target_seq_len;

  float q_reg[MAX_D];
  if (valid_q) {
    for (int d = 0; d < head_dim; ++d) {
      q_reg[d] = val_to_float(*(q_ptr_base + q_row_global * stride_Q_S + d));
    }
  } else {
      for (int d = 0; d < MAX_D; ++d) q_reg[d] = 0.0f;
  }

  double o_accum[MAX_D];
  for (int d = 0; d < MAX_D; ++d) o_accum[d] = 0.0;
  
  double m = -INFINITY;
  double l = 0.0;
  
  double sqrt_d = sqrt((double)head_dim);

  for (int j = 0; j < src_seq_len; j += Tc) {
    int valid_wc = min(Tc, src_seq_len - j);
    int k_row_global = j + tx;
    bool valid_k_load = (tx < valid_wc); 

    if (valid_k_load) {
        for (int d = 0; d < head_dim; ++d) {
            K_smem[tx * MAX_D + d] = val_to_float(*(k_ptr_base + k_row_global * stride_K_S + d));
            V_smem[tx * MAX_D + d] = val_to_float(*(v_ptr_base + k_row_global * stride_K_S + d));
        }
    }
    __syncthreads();

    if (valid_q) {
        double scores[Tc];
        double local_max = -INFINITY;

        for (int k = 0; k < valid_wc; ++k) {
             double score = 0.0;
             for (int d = 0; d < head_dim; ++d) {
                 score += (double)q_reg[d] * (double)K_smem[k * MAX_D + d];
             }
             score /= sqrt_d;

             if (is_causal) {
                 if (j + k > q_row_global) {
                     score = -INFINITY;
                 }
             }
             scores[k] = score;
             local_max = fmax(local_max, score);
        }

        if (local_max != -INFINITY) {
            double m_new = fmax(m, local_max);
            double exp_m_diff = exp(m - m_new); 
            
            double block_sum = 0.0;
            for (int k = 0; k < valid_wc; ++k) {
                double p = 0.0;
                if (scores[k] != -INFINITY) {
                    p = exp(scores[k] - m_new);
                }
                scores[k] = p;
                block_sum += p;
            }

            l = l * exp_m_diff + block_sum;
            
            for (int d = 0; d < head_dim; ++d) {
                 double pv = 0.0;
                 for (int k = 0; k < valid_wc; ++k) {
                     pv += scores[k] * (double)V_smem[k * MAX_D + d];
                 }
                 o_accum[d] = o_accum[d] * exp_m_diff + pv;
            }
            m = m_new;
        }
    }
    __syncthreads();
  }

  if (valid_q) {
      for (int d = 0; d < head_dim; ++d) {
          double res = (l > 0.0) ? (o_accum[d] / l) : 0.0;
          *(o_ptr_base + q_row_global * stride_Q_S + d) = float_to_val<T>((float)res);
      }
  }
}