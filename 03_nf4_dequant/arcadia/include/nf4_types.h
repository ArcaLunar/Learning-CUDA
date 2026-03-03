#ifndef NF4_TYPES_H
#define NF4_TYPES_H

#include <cstdint>
#include <vector>
#include <string>

// Metadata structure for NF4 quantized weights
struct WeightMetadata {
    int64_t num_rows;      // Number of rows in the weight matrix
    int64_t num_cols;      // Number of columns in the weight matrix
    int32_t blocksize;     // Block size for quantization (typically 64 or 128)
    int64_t num_blocks;    // Total number of blocks (num_rows * num_cols / blocksize)
    int64_t num_groups;    // Number of groups (num_blocks / 256)
    int64_t total_elements; // Total number of elements (num_rows * num_cols)
    
    WeightMetadata() : num_rows(0), num_cols(0), blocksize(0), 
                       num_blocks(0), num_groups(0), total_elements(0) {}
};

// Structure to hold all NF4 quantized data
struct NF4QuantizedWeights {
    WeightMetadata metadata;
    
    std::vector<uint8_t> packed_weights;  // 4-bit indices packed as uint8 (2 per byte)
    std::vector<uint8_t> absmax_q;        // Level-1 quantized scale factors (one per block)
    std::vector<uint16_t> absmax2;        // Level-2 scale factors (one per group), stored as fp16
    std::vector<uint16_t> code2;          // Codebook for absmax_q (256 entries), stored as fp16
    float offset;                          // Quantization offset
    
    NF4QuantizedWeights() : offset(0.0f) {}
};

// Configuration parameters
struct NF4Config {
    int32_t blocksize;
    std::string compute_type;  // "bf16" or "fp16"
    std::string target_gpu;
    
    NF4Config() : blocksize(64), compute_type("bf16"), target_gpu("T4") {}
};

#endif // NF4_TYPES_H
