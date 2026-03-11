#include "../include/common.h"
#include "../include/nf4_constants.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

constexpr int BLOCK_SIZE = 256; // Number of threads per block
constexpr int BYTES_PER_THREAD = 4;

#ifndef NF4_USE_CODE2_CONST
#define NF4_USE_CODE2_CONST 1
#endif

#ifndef NF4_DYNAMIC_SCHEDULE
#define NF4_DYNAMIC_SCHEDULE 1
#endif

#ifndef NF4_CONTIGUOUS_THRESHOLD_BYTES
#define NF4_CONTIGUOUS_THRESHOLD_BYTES 1048576
#endif

#if NF4_USE_CODE2_CONST
constexpr int CODE2_SIZE = 256;
__constant__ __half CODE2_CONST[CODE2_SIZE];
#endif

// ---------------------------------------------------------------------------
// Conversion helpers: float -> output type T
// __float2half  is available on all sm
// __float2bfloat16 is available on all sm (only *arithmetic* needs sm_80+)
// ---------------------------------------------------------------------------
template <typename T> __device__ __forceinline__ T float_to_output(float x);

template <typename T> __device__ __forceinline__ uint16_t output_to_u16(T x) {
  union {
    T value;
    uint16_t bits;
  } cvt;
  cvt.value = x;
  return cvt.bits;
}

template <> __device__ __forceinline__ __half float_to_output<__half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16
float_to_output<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

template <typename T>
__device__ __forceinline__ void store_output_pair(T *__restrict__ output,
                                                  int64_t elem_idx_even,
                                                  T out_even, T out_odd) {
  const ushort2 packed =
      make_ushort2(output_to_u16<T>(out_even), output_to_u16<T>(out_odd));
  *reinterpret_cast<ushort2 *>(&output[elem_idx_even]) = packed;
}

template <typename T, int BLOCK_SHIFT>
__device__ __forceinline__ void process_packed_index(
    int64_t packed_idx, const uint8_t *__restrict__ packed_weights,
    const uint8_t *__restrict__ absmax_q, const __half *__restrict__ absmax2,
    const __half *__restrict__ d_code2, float offset,
    T *__restrict__ output) {
  uint8_t packed = packed_weights[packed_idx];
  uint8_t idx_even = (packed >> 4) & 0x0F;
  uint8_t idx_odd = packed & 0x0F;

  int64_t elem_idx_even = packed_idx * 2;
  float base_val_even = NF4_LUT[idx_even];
  float base_val_odd = NF4_LUT[idx_odd];

  int64_t block_idx = elem_idx_even >> BLOCK_SHIFT;
  int64_t group_idx = block_idx >> 8;

#if NF4_USE_CODE2_CONST
  float block_scale =
      __half2float(CODE2_CONST[absmax_q[block_idx]]) *
          __half2float(absmax2[group_idx]) +
      offset;
#else
  float block_scale =
      __half2float(d_code2[absmax_q[block_idx]]) *
          __half2float(absmax2[group_idx]) +
      offset;
#endif

  float dequant_even = base_val_even * block_scale;
  float dequant_odd = base_val_odd * block_scale;
  T out_even = float_to_output<T>(dequant_even);
  T out_odd = float_to_output<T>(dequant_odd);
  store_output_pair(output, elem_idx_even, out_even, out_odd);
}

// NF4 dequantization kernel - templated on output type T (__half or
// __nv_bfloat16). BLOCK_SHIFT specializes block index math to compile-time
// shifts to avoid expensive runtime integer division in the hot path.
template <typename T, int BLOCK_SHIFT>
__global__ void nf4_dequantize_kernel_strided(
    const uint8_t *__restrict__ packed_weights, // 4-bit indices packed as uint8
                                                // (2 per byte)
    const uint8_t *__restrict__ absmax_q, // Level-1 quantized scale factors
    const __half *__restrict__ absmax2,   // Level-2 scale factors (fp16)
    const __half *__restrict__ d_code2,   // Codebook fallback in global memory
    float offset,                         // Quantization offset
    T *__restrict__ output, // Output dequantized weights (fp16 or bf16)
    int64_t num_rows, int64_t num_cols) {
  // Global thread index
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t grid_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  // Total number of bytes in packed_weights (each byte contains 2 indices)
  int64_t total_elements = num_rows * num_cols;
  int64_t total_bytes = total_elements >> 1;

  // Grid-stride with small unroll to amortize index math and improve
  // throughput.
  for (int64_t base = tid; base < total_bytes;
       base += grid_stride * BYTES_PER_THREAD) {
#pragma unroll
    for (int i = 0; i < BYTES_PER_THREAD; ++i) {
      int64_t packed_idx = base + i * grid_stride;
      if (packed_idx >= total_bytes)
        break;
      process_packed_index<T, BLOCK_SHIFT>(packed_idx, packed_weights, absmax_q,
                                           absmax2, d_code2, offset, output);
    }
  }
}

template <typename T, int BLOCK_SHIFT>
__global__ void nf4_dequantize_kernel_contiguous(
    const uint8_t *__restrict__ packed_weights,
    const uint8_t *__restrict__ absmax_q, const __half *__restrict__ absmax2,
    const __half *__restrict__ d_code2, float offset, T *__restrict__ output,
    int64_t num_rows, int64_t num_cols) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t tiles_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * BYTES_PER_THREAD;

  int64_t total_elements = num_rows * num_cols;
  int64_t total_bytes = total_elements >> 1;

  for (int64_t base = tid * BYTES_PER_THREAD; base < total_bytes;
       base += tiles_stride) {
#pragma unroll
    for (int i = 0; i < BYTES_PER_THREAD; ++i) {
      int64_t packed_idx = base + i;
      if (packed_idx >= total_bytes)
        break;

      process_packed_index<T, BLOCK_SHIFT>(packed_idx, packed_weights, absmax_q,
                                           absmax2, d_code2, offset, output);
    }
  }
}

template <typename T>
inline void launch_nf4_dequantize_kernel_for_blocksize_strided(
    int32_t blocksize, int num_blocks, cudaStream_t stream,
    const uint8_t *d_packed_weights, const uint8_t *d_absmax_q,
    const __half *d_absmax2, const __half *d_code2, float offset, T *d_output,
    int64_t num_rows, int64_t num_cols) {
  switch (blocksize) {
  case 64:
    nf4_dequantize_kernel_strided<T, 6><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_packed_weights, d_absmax_q, d_absmax2, d_code2, offset, d_output,
        num_rows, num_cols);
    break;
  case 128:
    nf4_dequantize_kernel_strided<T, 7><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_packed_weights, d_absmax_q, d_absmax2, d_code2, offset, d_output,
        num_rows, num_cols);
    break;
  case 256:
    nf4_dequantize_kernel_strided<T, 8><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_packed_weights, d_absmax_q, d_absmax2, d_code2, offset, d_output,
        num_rows, num_cols);
    break;
  default:
    fprintf(stderr,
            "Unsupported blocksize %d (expected one of: 64, 128, 256)\n",
            static_cast<int>(blocksize));
    exit(EXIT_FAILURE);
  }
}

template <typename T>
inline void launch_nf4_dequantize_kernel_for_blocksize_contiguous(
    int32_t blocksize, int num_blocks, cudaStream_t stream,
    const uint8_t *d_packed_weights, const uint8_t *d_absmax_q,
    const __half *d_absmax2, const __half *d_code2, float offset, T *d_output,
    int64_t num_rows, int64_t num_cols) {
  switch (blocksize) {
  case 64:
    nf4_dequantize_kernel_contiguous<T, 6><<<num_blocks, BLOCK_SIZE, 0,
                                             stream>>>(d_packed_weights,
                                                       d_absmax_q, d_absmax2,
                                                       d_code2, offset,
                                                       d_output, num_rows,
                                                       num_cols);
    break;
  case 128:
    nf4_dequantize_kernel_contiguous<T, 7><<<num_blocks, BLOCK_SIZE, 0,
                                             stream>>>(d_packed_weights,
                                                       d_absmax_q, d_absmax2,
                                                       d_code2, offset,
                                                       d_output, num_rows,
                                                       num_cols);
    break;
  case 256:
    nf4_dequantize_kernel_contiguous<T, 8><<<num_blocks, BLOCK_SIZE, 0,
                                             stream>>>(d_packed_weights,
                                                       d_absmax_q, d_absmax2,
                                                       d_code2, offset,
                                                       d_output, num_rows,
                                                       num_cols);
    break;
  default:
    fprintf(stderr,
            "Unsupported blocksize %d (expected one of: 64, 128, 256)\n",
            static_cast<int>(blocksize));
    exit(EXIT_FAILURE);
  }
}

template <typename T>
inline void launch_nf4_dequantize_kernel_for_blocksize(
    bool use_contiguous, int32_t blocksize, int num_blocks, cudaStream_t stream,
    const uint8_t *d_packed_weights, const uint8_t *d_absmax_q,
    const __half *d_absmax2, const __half *d_code2, float offset, T *d_output,
    int64_t num_rows, int64_t num_cols) {
  if (use_contiguous) {
    launch_nf4_dequantize_kernel_for_blocksize_contiguous(
        blocksize, num_blocks, stream, d_packed_weights, d_absmax_q, d_absmax2,
        d_code2, offset, d_output, num_rows, num_cols);
  } else {
    launch_nf4_dequantize_kernel_for_blocksize_strided(
        blocksize, num_blocks, stream, d_packed_weights, d_absmax_q, d_absmax2,
        d_code2, offset, d_output, num_rows, num_cols);
  }
}

// Host function to launch the kernel
// use_bf16=false → output as __half (fp16)
// use_bf16=true  → output as __nv_bfloat16 (bf16, requires CUDA 11+ headers;
// native math needs sm_80+)
void launch_nf4_dequantize(const uint8_t *d_packed_weights,
                           const uint8_t *d_absmax_q, const __half *d_absmax2,
                           const __half *d_code2, float offset, void *d_output,
                           int64_t num_rows, int64_t num_cols,
                           int32_t blocksize, bool use_bf16,
                           cudaStream_t stream) {
  // Calculate kernel launch configuration
  int64_t total_elements = num_rows * num_cols;
  int64_t total_bytes =
      total_elements / 2; // Each thread processes 1 byte (2 elements)

  // Choose block size (256 threads per block is common for memory-bound
  // kernels) const int threads_per_block = 256;
  int num_blocks = (total_bytes + BLOCK_SIZE * BYTES_PER_THREAD - 1) /
                   (BLOCK_SIZE * BYTES_PER_THREAD);

  // Optional constant-memory path for code2 LUT.
#if NF4_USE_CODE2_CONST
  CHECK_CUDA(cudaMemcpyToSymbolAsync(CODE2_CONST, d_code2,
                                     CODE2_SIZE * sizeof(__half), 0,
                                     cudaMemcpyDeviceToDevice, stream));
#endif

  // Choose schedule based on total_bytes.
#if NF4_DYNAMIC_SCHEDULE
  bool use_contiguous = (total_bytes <= NF4_CONTIGUOUS_THRESHOLD_BYTES);
#else
  bool use_contiguous = false;
#endif

  // Dispatch to the appropriate template instantiation
  if (use_bf16) {
    launch_nf4_dequantize_kernel_for_blocksize<__nv_bfloat16>(
        use_contiguous, blocksize, num_blocks, stream, d_packed_weights,
        d_absmax_q, d_absmax2, d_code2, offset,
        reinterpret_cast<__nv_bfloat16 *>(d_output), num_rows, num_cols);
  } else {
    launch_nf4_dequantize_kernel_for_blocksize<__half>(
        use_contiguous, blocksize, num_blocks, stream, d_packed_weights,
        d_absmax_q, d_absmax2, d_code2, offset,
        reinterpret_cast<__half *>(d_output), num_rows, num_cols);
  }

  CHECK_KERNEL();
}
