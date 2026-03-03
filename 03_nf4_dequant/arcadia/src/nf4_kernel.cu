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
    uint8_t idx_low = packed & 0x0F;         // Low 4 bits (first element)
    uint8_t idx_high = (packed >> 4) & 0x0F; // High 4 bits (second element)
    
    // Calculate element indices for the two values
    int64_t elem_idx_low = tid * 2;
    int64_t elem_idx_high = tid * 2 + 1;
    
    // Check boundary for the second element
    bool write_high = (elem_idx_high < total_elements);
    
    // Lookup NF4 base values from constant memory
    float base_val_low = NF4_LUT[idx_low];
    float base_val_high = NF4_LUT[idx_high];
    
    // Calculate block indices for both elements
    int64_t block_idx_low = elem_idx_low / blocksize;
    int64_t block_idx_high = elem_idx_high / blocksize;
    
    // Calculate group indices (256 blocks per group)
    int64_t group_idx_low = block_idx_low / 256;
    int64_t group_idx_high = block_idx_high / 256;
    
    // Level-1 scaling: Get quantized scale and decode via code2 lookup table
    __half scale1_low = code2[absmax_q[block_idx_low]];
    __half scale1_high = code2[absmax_q[block_idx_high]];
    
    // Level-2 scaling
    __half scale2_low = absmax2[group_idx_low];
    __half scale2_high = absmax2[group_idx_high];
    
    // Apply two-level dequantization: output = base_val * scale1 * scale2 + offset
    float dequant_low = base_val_low * __half2float(scale1_low) * __half2float(scale2_low) + offset;
    float dequant_high = base_val_high * __half2float(scale1_high) * __half2float(scale2_high) + offset;
    
    // Convert to half precision
    __half out_low = __float2half(dequant_low);
    __half out_high = __float2half(dequant_high);
    
    // Vectorized memory write: Pack 2x fp16 values into uint32_t and write once
    // This improves memory bandwidth utilization
    if (write_high) {
        // Both elements are valid - use packed write
        uint32_t packed_output;
        *reinterpret_cast<__half*>(&packed_output) = out_low;
        *reinterpret_cast<__half*>(reinterpret_cast<uint16_t*>(&packed_output) + 1) = out_high;
        *reinterpret_cast<uint32_t*>(&output[elem_idx_low]) = packed_output;
    } else {
        // Only first element is valid (boundary case)
        output[elem_idx_low] = out_low;
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
