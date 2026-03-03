#ifndef NF4_CONSTANTS_CUH
#define NF4_CONSTANTS_CUH

#include <cuda_fp16.h>

// NF4 Lookup Table: 16 quantization values from the quantiles of N(0,1)
// These values divide the standard normal distribution CDF into 16 equal-probability bins
__constant__ __device__ float NF4_LUT[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

#endif // NF4_CONSTANTS_CUH
