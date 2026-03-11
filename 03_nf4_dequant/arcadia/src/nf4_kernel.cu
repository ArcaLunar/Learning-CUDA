#include "../include/common.h"
#include "../include/nf4_constants.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

constexpr int BLOCK_SIZE = 256; // Number of threads per block

// ---------------------------------------------------------------------------
// Conversion helpers: float -> output type T
// __float2half  is available on all sm
// __float2bfloat16 is available on all sm (only *arithmetic* needs sm_80+)
// ---------------------------------------------------------------------------
template <typename T> __device__ __forceinline__ T float_to_output(float x);

template <> __device__ __forceinline__ __half float_to_output<__half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16
float_to_output<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

// NF4 dequantization kernel - templated on output type T (__half or
// __nv_bfloat16). BLOCK_SHIFT specializes block index math to compile-time
// shifts to avoid expensive runtime integer division in the hot path.
template <typename T, int BLOCK_SHIFT>
__global__ void nf4_dequantize_kernel(
    const uint8_t *__restrict__ packed_weights, // 4-bit indices packed as uint8
                                                // (2 per byte)
    const uint8_t *__restrict__ absmax_q, // Level-1 quantized scale factors
    const __half *__restrict__ absmax2,   // Level-2 scale factors (fp16)
    const __half
        *__restrict__ code2, // Codebook for absmax_q (256 entries, fp16)
    float offset,            // Quantization offset
    T *__restrict__ output,  // Output dequantized weights (fp16 or bf16)
    int64_t num_rows, int64_t num_cols, int32_t blocksize) {
  // Global thread index
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Total number of bytes in packed_weights (each byte contains 2 indices)
  int64_t total_elements = num_rows * num_cols;
  int64_t total_bytes = total_elements >> 1;

  // Each thread processes one byte (2 elements)
  if (tid >= total_bytes)
    return;

  // Read packed byte containing 2x 4-bit indices
  uint8_t packed = packed_weights[tid];
  // bitsandbytes packing convention:
  //   HIGH 4 bits (bits 7-4) → element at even position (2k)
  //   LOW  4 bits (bits 3-0) → element at odd  position (2k+1)
  uint8_t idx_even = (packed >> 4) & 0x0F; // High nibble → even index (first)
  uint8_t idx_odd = packed & 0x0F;         // Low  nibble → odd  index (second)

  // Calculate element index for the pair of values
  int64_t elem_idx_even = tid * 2;

  // Lookup NF4 base values from constant memory
  float base_val_even = NF4_LUT[idx_even];
  float base_val_odd = NF4_LUT[idx_odd];

  // For supported even block sizes, both elements map to the same block.
  int64_t block_idx = elem_idx_even >> BLOCK_SHIFT;

  // Calculate group index (256 blocks per group).
  int64_t group_idx = block_idx >> 8;

  // Reconstruct block scale via two-level nested dequantization:
  //   block_scale = nested_quant_map[absmax_q[block]] * nested_absmax[group] +
  //   nested_offset
  // Note: offset (nested_offset) is added INSIDE the block scale, not to the
  // final output.
  float block_scale = __half2float(code2[absmax_q[block_idx]]) *
                          __half2float(absmax2[group_idx]) +
                      offset;

  // Apply NF4 dequantization: output = NF4_LUT[idx] * block_scale
  float dequant_even = base_val_even * block_scale;
  float dequant_odd = base_val_odd * block_scale;

  // Convert to output type (T = __half or __nv_bfloat16, both 16-bit)
  T out_even = float_to_output<T>(dequant_even);
  T out_odd = float_to_output<T>(dequant_odd);

  // Vectorized memory write: Pack 2x 16-bit values into uint32_t and write once
  // Works for both __half and __nv_bfloat16 since both are 16-bit types
  uint32_t packed_output;
  *reinterpret_cast<T *>(&packed_output) = out_even;
  *reinterpret_cast<T *>(reinterpret_cast<uint16_t *>(&packed_output) + 1) =
      out_odd;
  *reinterpret_cast<uint32_t *>(&output[elem_idx_even]) = packed_output;
}

template <typename T>
inline void launch_nf4_dequantize_kernel_for_blocksize(
    int32_t blocksize, int num_blocks, cudaStream_t stream,
    const uint8_t *d_packed_weights, const uint8_t *d_absmax_q,
    const __half *d_absmax2, const __half *d_code2, float offset, T *d_output,
    int64_t num_rows, int64_t num_cols) {
  switch (blocksize) {
  case 64:
    nf4_dequantize_kernel<T, 6><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_packed_weights, d_absmax_q, d_absmax2, d_code2, offset, d_output,
        num_rows, num_cols, blocksize);
    break;
  case 128:
    nf4_dequantize_kernel<T, 7><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_packed_weights, d_absmax_q, d_absmax2, d_code2, offset, d_output,
        num_rows, num_cols, blocksize);
    break;
  case 256:
    nf4_dequantize_kernel<T, 8><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_packed_weights, d_absmax_q, d_absmax2, d_code2, offset, d_output,
        num_rows, num_cols, blocksize);
    break;
  default:
    fprintf(stderr,
            "Unsupported blocksize %d (expected one of: 64, 128, 256)\n",
            static_cast<int>(blocksize));
    exit(EXIT_FAILURE);
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
  int num_blocks = (total_bytes + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Dispatch to the appropriate template instantiation
  if (use_bf16) {
    launch_nf4_dequantize_kernel_for_blocksize<__nv_bfloat16>(
        blocksize, num_blocks, stream, d_packed_weights, d_absmax_q, d_absmax2,
        d_code2, offset, reinterpret_cast<__nv_bfloat16 *>(d_output), num_rows,
        num_cols);
  } else {
    launch_nf4_dequantize_kernel_for_blocksize<__half>(
        blocksize, num_blocks, stream, d_packed_weights, d_absmax_q, d_absmax2,
        d_code2, offset, reinterpret_cast<__half *>(d_output), num_rows,
        num_cols);
  }

  CHECK_KERNEL();
}
