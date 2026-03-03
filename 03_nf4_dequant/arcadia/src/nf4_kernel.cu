#include "../include/common.h"
#include "../include/nf4_constants.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Conversion helpers: float -> output type T
// __float2half  is available on all sm
// __float2bfloat16 is available on all sm (only *arithmetic* needs sm_80+)
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T float_to_output(float x);

template <>
__device__ __forceinline__ __half float_to_output<__half>(float x) {
    return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_output<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// NF4 dequantization kernel - templated on output type T (__half or __nv_bfloat16)
// Each thread processes 2 elements (packed in 1 byte) and writes them as uint32_t for vectorization
template <typename T>
__global__ void nf4_dequantize_kernel(
    const uint8_t* __restrict__ packed_weights,   // 4-bit indices packed as uint8 (2 per byte)
    const uint8_t* __restrict__ absmax_q,         // Level-1 quantized scale factors
    const __half* __restrict__ absmax2,           // Level-2 scale factors (fp16)
    const __half* __restrict__ code2,             // Codebook for absmax_q (256 entries, fp16)
    float offset,                                  // Quantization offset
    T* __restrict__ output,                       // Output dequantized weights (fp16 or bf16)
    int64_t num_rows,
    int64_t num_cols,
    int32_t blocksize
) {
    // Global thread index
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of bytes in packed_weights (each byte contains 2 indices)
    int64_t total_elements = num_rows * num_cols;
    int64_t total_bytes = total_elements / 2;
    
    // Each thread processes one byte (2 elements)
    if (tid >= total_bytes) return;
    
    // Read packed byte containing 2x 4-bit indices
    uint8_t packed = packed_weights[tid];
    // bitsandbytes packing convention:
    //   HIGH 4 bits (bits 7-4) → element at even position (2k)
    //   LOW  4 bits (bits 3-0) → element at odd  position (2k+1)
    uint8_t idx_even = (packed >> 4) & 0x0F; // High nibble → even index (first)
    uint8_t idx_odd  =  packed       & 0x0F; // Low  nibble → odd  index (second)
    
    // Calculate element indices for the two values
    int64_t elem_idx_even = tid * 2;
    int64_t elem_idx_odd  = tid * 2 + 1;
    
    // Check boundary for the second element
    bool write_odd = (elem_idx_odd < total_elements);
    
    // Lookup NF4 base values from constant memory
    float base_val_even = NF4_LUT[idx_even];
    float base_val_odd  = NF4_LUT[idx_odd];
    
    // Calculate block indices for both elements
    int64_t block_idx_even = elem_idx_even / blocksize;
    int64_t block_idx_odd  = elem_idx_odd  / blocksize;
    
    // Calculate group indices (256 blocks per group)
    int64_t group_idx_even = block_idx_even / 256;
    int64_t group_idx_odd  = block_idx_odd  / 256;
    
    // Reconstruct block scale via two-level nested dequantization:
    //   block_scale = nested_quant_map[absmax_q[block]] * nested_absmax[group] + nested_offset
    // Note: offset (nested_offset) is added INSIDE the block scale, not to the final output.
    float block_scale_even = __half2float(code2[absmax_q[block_idx_even]])
                           * __half2float(absmax2[group_idx_even])
                           + offset;
    float block_scale_odd  = __half2float(code2[absmax_q[block_idx_odd]])
                           * __half2float(absmax2[group_idx_odd])
                           + offset;
    
    // Apply NF4 dequantization: output = NF4_LUT[idx] * block_scale
    float dequant_even = base_val_even * block_scale_even;
    float dequant_odd  = base_val_odd  * block_scale_odd;
    
    // Convert to output type (T = __half or __nv_bfloat16, both 16-bit)
    T out_even = float_to_output<T>(dequant_even);
    T out_odd  = float_to_output<T>(dequant_odd);
    
    // Vectorized memory write: Pack 2x 16-bit values into uint32_t and write once
    // Works for both __half and __nv_bfloat16 since both are 16-bit types
    if (write_odd) {
        uint32_t packed_output;
        *reinterpret_cast<T*>(&packed_output) = out_even;
        *reinterpret_cast<T*>(reinterpret_cast<uint16_t*>(&packed_output) + 1) = out_odd;
        *reinterpret_cast<uint32_t*>(&output[elem_idx_even]) = packed_output;
    } else {
        output[elem_idx_even] = out_even;
    }
}

// Host function to launch the kernel
// use_bf16=false → output as __half (fp16)
// use_bf16=true  → output as __nv_bfloat16 (bf16, requires CUDA 11+ headers; native math needs sm_80+)
void launch_nf4_dequantize(
    const uint8_t* d_packed_weights,
    const uint8_t* d_absmax_q,
    const __half* d_absmax2,
    const __half* d_code2,
    float offset,
    void* d_output,
    int64_t num_rows,
    int64_t num_cols,
    int32_t blocksize,
    bool use_bf16,
    cudaStream_t stream
) {
    // Calculate kernel launch configuration
    int64_t total_elements = num_rows * num_cols;
    int64_t total_bytes = total_elements / 2;  // Each thread processes 1 byte (2 elements)
    
    // Choose block size (256 threads per block is common for memory-bound kernels)
    const int threads_per_block = 256;
    int num_blocks = (total_bytes + threads_per_block - 1) / threads_per_block;
    
    // Dispatch to the appropriate template instantiation
    if (use_bf16) {
        nf4_dequantize_kernel<__nv_bfloat16><<<num_blocks, threads_per_block, 0, stream>>>(
            d_packed_weights, d_absmax_q, d_absmax2, d_code2, offset,
            reinterpret_cast<__nv_bfloat16*>(d_output),
            num_rows, num_cols, blocksize
        );
    } else {
        nf4_dequantize_kernel<__half><<<num_blocks, threads_per_block, 0, stream>>>(
            d_packed_weights, d_absmax_q, d_absmax2, d_code2, offset,
            reinterpret_cast<__half*>(d_output),
            num_rows, num_cols, blocksize
        );
    }
    
    CHECK_KERNEL();
}
