#ifndef NF4_KERNEL_CUH
#define NF4_KERNEL_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Launch the NF4 dequantization kernel
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
    cudaStream_t stream = 0
);

#endif // NF4_KERNEL_CUH
