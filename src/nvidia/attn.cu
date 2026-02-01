#pragma once
#include "helper.cu"
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ===============================
// Utils
// ===============================

__device__ __forceinline__ float to_float(half val) {
  return __half2float(val);
}
__device__ __forceinline__ float to_float(float val) { return val; }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) {
  return x;
}
template <> __device__ __forceinline__ half from_float<half>(float x) {
  return __float2half_rn(x);
}

__host__ __device__ __forceinline__ int idx4(int b, int s, int h, int d, int S,
                                             int H, int D) {
  return ((b * S + s) * H + h) * D + d;
}

// ===============================
// Block-level reduction (warp level is in helper.cu)
// ===============================

__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float smem[32];
  __shared__ float out_s;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  // first reduction
  v = warp_reduce_max(v);
  if (lane == 0)
    smem[warp] = v;
  __syncthreads();

  // final reduction
  float out = -INFINITY;
  if (warp == 0) {
    out = lane < (blockDim.x >> 5) ? smem[lane] : -INFINITY;
    out = warp_reduce_max(out);
    if (lane == 0)
      out_s = out;
  }
  __syncthreads();
  return out_s;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float smem[32];
  __shared__ float out_s;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  // first reduction
  v = warp_reduce_sum(v);
  if (lane == 0)
    smem[warp] = v;
  __syncthreads();

  // final reduction
  float out = 0.0f;
  if (warp == 0) {
    out = lane < (blockDim.x >> 5) ? smem[lane] : 0.0f;
    out = warp_reduce_sum(out);
    if (lane == 0)
      out_s = out;
  }
  __syncthreads();
  return out_s;
}

// ===============================
// Flash Attention Kernel
// ===============================

template <typename T, int TILE_N, int MAX_D>
__global__ void
flashattn(const T *__restrict__ q, // [batch, qlen, qhead, head_dim]
          const T *__restrict__ k, // [batch, klen, kvhead, head_dim]
          const T *__restrict__ v, // [batch, klen, kvhead, head_dim]
          T *__restrict__ o,       // [batch, qlen, qhead, head_dim]
          int batch, int qlen, int kvlen, int qheads, int kvheads, int headdim,
          bool is_causal) {
  // grid: [batch, qhead, qlen]
  int b = blockIdx.x;
  int qh = blockIdx.y;
  int m = blockIdx.z;

  float scale = 1.0 / sqrtf((float)headdim);

  // GQA mapping
  int kvh = (kvheads == qheads) ? qh : qh * kvheads / qheads;

  // causal
  int k_max = kvlen;
  if (is_causal)
    k_max = min(m + 1, kvlen);

  // shared memory for k,v tile
  extern __shared__ unsigned char sram[];
  T *sk = reinterpret_cast<T *>(sram); // [BLOCK_K, head_dim]
  T *sv = sk + TILE_N * MAX_D;         // [BLOCK_K, head_dim]

  // online softmax stats
  float m_i = -INFINITY;
  float l_i = 0.0f;

  constexpr int MAX_STRIPE = (MAX_D + 127) / 128;
  float o_stripe[MAX_STRIPE], tile_stripe[MAX_STRIPE];
#pragma unroll
  for (int i = 0; i < MAX_STRIPE; i++)
    o_stripe[i] = 0.0f;

  // load Q (b, m, qh)
  for (int n0 = 0; n0 < k_max; n0 += TILE_N) {
    int tile_len = min(TILE_N, k_max - n0);

    // load K, V
    for (int t = threadIdx.x; t < tile_len * headdim; t += blockDim.x) {
      int j = t / headdim;
      int d = t - j * headdim;
      int k_idx = idx4(b, n0 + j, kvh, d, kvlen, kvheads, headdim);
      sk[j * MAX_D + d] = k[k_idx];
      sv[j * MAX_D + d] = v[k_idx];
    }
    __syncthreads();

    // 1. compute max of qk^T * scale
    float tile_max = -INFINITY;
    for (int j = 0; j < tile_len; j++) {
      float partial = 0.0;
      // dot over headdim
      for (int d0 = threadIdx.x; d0 < headdim; d0 += blockDim.x) {
        int q_idx = idx4(b, m, qh, d0, qlen, qheads, headdim);
        float q_val = to_float(q[q_idx]);
        float k_val = to_float(sk[j * MAX_D + d0]);
        partial += q_val * k_val;
      }

      // block-reduce partial
      float dot = block_reduce_sum(partial);
      float s = dot * scale;
      tile_max = fmaxf(tile_max, s);
      __syncthreads();
    }

    float m_new = fmaxf(m_i, tile_max);

// reset tiles
#pragma unroll
    for (int i = 0; i < MAX_STRIPE; i++)
      tile_stripe[i] = 0.0f;

    // 2. exp sums
    float l_tile = 0.0f;
    for (int j = 0; j < tile_len; ++j) {
      float partial = 0.0f;
      for (int d0 = threadIdx.x; d0 < headdim; d0 += blockDim.x) {
        int q_idx = idx4(b, m, qh, d0, qlen, qheads, headdim);
        float q_val = to_float(q[q_idx]);
        float k_val = to_float(sk[j * MAX_D + d0]);
        partial += q_val * k_val;
      }

      float dot = block_reduce_sum(partial);
      float s = dot * scale;
      float p = expf(s - m_new);

      if (threadIdx.x == 0)
        l_tile += p;
      __syncthreads();

// tile_stripe += p * vj
#pragma unroll
      for (int i = 0; i < MAX_STRIPE; i++) {
        int d = threadIdx.x + i * blockDim.x;
        if (d < headdim)
          tile_stripe[i] += p * to_float(sv[j * MAX_D + d]);
      }
      __syncthreads();
    }

    // broadcast l_tile
    __shared__ float l_tile_shared;
    if (threadIdx.x == 0)
      l_tile_shared = l_tile;
    __syncthreads();
    l_tile = l_tile_shared;

    float alpha = expf(m_i - m_new);
    float l_new = l_i * alpha + l_tile;

    float old_scale = (l_i == 0.0f) ? 0.0f : (alpha * l_i) / l_new;
    float new_scale = 1.0 / l_new;

    // update o_stripe
#pragma unroll
    for (int i = 0; i < MAX_STRIPE; i++)
      o_stripe[i] = o_stripe[i] * old_scale + tile_stripe[i] * new_scale;

    // update m_i, l_i
    m_i = m_new;
    l_i = l_new;

    __syncthreads();
  }

// write output
#pragma unroll
  for (int i = 0; i < MAX_STRIPE; i++) {
    int d = threadIdx.x + i * blockDim.x;
    if (d < headdim) {
      int o_idx = idx4(b, m, qh, d, qlen, qheads, headdim);
      o[o_idx] = from_float<T>(o_stripe[i]);
    }
  }
}