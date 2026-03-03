#include "../include/common.h"
#include "../include/nf4_constants.cuh"
#include <cuda_fp16.h>

// NF4 dequantization kernel - single kernel implementation
// Each thread processes 2 elements (packed in 1 byte) and writes them as uint32_t for vectorization
__global__ void nf4_dequantize_kernel(
    const uint8_t* __restrict__ packed_weights,   // 4-bit indices packed as uint8 (2 per byte)
    const uint8_t* __restrict__ absmax_q,         // Level-1 quantized scale factors
    const __half* __restrict__ absmax2,           // Level-2 scale factors (fp16)
    const __half* __restrict__ code2,             // Codebook for absmax_q (256 entries, fp16)
    float offset,                                  // Quantization offset
    __half* __restrict__ output,                  // Output dequantized weights (fp16)
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
    
    // Convert to half precision
    __half out_even = __float2half(dequant_even);
    __half out_odd  = __float2half(dequant_odd);
    
    // Vectorized memory write: Pack 2x fp16 values into uint32_t and write once
    // This improves memory bandwidth utilization
    if (write_odd) {
        // Both elements are valid - use packed write
        uint32_t packed_output;
        *reinterpret_cast<__half*>(&packed_output) = out_even;
        *reinterpret_cast<__half*>(reinterpret_cast<uint16_t*>(&packed_output) + 1) = out_odd;
        *reinterpret_cast<uint32_t*>(&output[elem_idx_even]) = packed_output;
    } else {
        // Only first element is valid (boundary case)
        output[elem_idx_even] = out_even;
    }
}

// Host function to launch the kernel
void launch_nf4_dequantize(
    const uint8_t* d_packed_weights,
    const uint8_t* d_absmax_q,
    const __half* d_absmax2,
    const __half* d_code2,
    float offset,
    __half* d_output,
    int64_t num_rows,
    int64_t num_cols,
    int32_t blocksize,
    cudaStream_t stream
) {
    // Calculate kernel launch configuration
    int64_t total_elements = num_rows * num_cols;
    int64_t total_bytes = total_elements / 2;  // Each thread processes 1 byte (2 elements)
    
    // Choose block size (256 threads per block is common for memory-bound kernels)
    const int threads_per_block = 256;
    int num_blocks = (total_bytes + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    nf4_dequantize_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_packed_weights,
        d_absmax_q,
        d_absmax2,
        d_code2,
        offset,
        d_output,
        num_rows,
        num_cols,
        blocksize
    );
    
    CHECK_KERNEL();
}
