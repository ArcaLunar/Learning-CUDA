#include "../include/nf4_types.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

// Helper to convert fp16 (as uint16_t) to float for CPU-side comparison
inline float fp16_to_float(uint16_t h) {
    union {
        uint32_t u;
        float f;
    } u;
    
    uint16_t sign = (h >> 15) & 0x1;
    uint16_t exponent = (h >> 10) & 0x1F;
    uint16_t fraction = h & 0x3FF;
    
    if (exponent == 0) {
        if (fraction == 0) {
            // Zero
            u.u = sign << 31;
        } else {
            // Subnormal
            exponent = 127 - 14;
            while ((fraction & 0x400) == 0) {
                fraction <<= 1;
                exponent--;
            }
            fraction &= 0x3FF;
            u.u = (sign << 31) | (exponent << 23) | (fraction << 13);
        }
    } else if (exponent == 0x1F) {
        // Inf or NaN
        u.u = (sign << 31) | (0xFF << 23) | (fraction << 13);
    } else {
        // Normal
        u.u = (sign << 31) | ((exponent + (127 - 15)) << 23) | (fraction << 13);
    }
    
    return u.f;
}

// BF16 is simply the top 16 bits of a float32 (1 sign + 8 exp + 7 mantissa).
// To convert: zero-extend the 16-bit value into the upper half of a uint32 and reinterpret.
inline float bf16_to_float(uint16_t b) {
    union { uint32_t u; float f; } u;
    u.u = static_cast<uint32_t>(b) << 16;
    return u.f;
}

bool write_output(const std::string& output_file, const void* data, 
                  size_t size_bytes, const std::string& data_type) {
    std::ofstream file(output_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create output file: " << output_file << std::endl;
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(data), size_bytes);
    file.close();
    
    std::cout << "Wrote " << size_bytes << " bytes (" << data_type 
              << ") to " << output_file << std::endl;
    
    return true;
}

bool write_performance_log(const std::string& log_file, 
                           float kernel_time_ms,
                           float bandwidth_gb_s,
                           float speedup,
                           float mae,
                           const WeightMetadata& metadata) {
    std::ofstream file(log_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create log file: " << log_file << std::endl;
        return false;
    }
    
    // Write performance metrics in JSON format
    file << "{\n";
    file << "  \"matrix_shape\": [" << metadata.num_rows << ", " << metadata.num_cols << "],\n";
    file << "  \"blocksize\": " << metadata.blocksize << ",\n";
    file << "  \"total_elements\": " << metadata.total_elements << ",\n";
    file << "  \"kernel_time_ms\": " << std::fixed << std::setprecision(4) << kernel_time_ms << ",\n";
    file << "  \"bandwidth_gb_s\": " << std::fixed << std::setprecision(2) << bandwidth_gb_s << ",\n";
    file << "  \"speedup_vs_bitsandbytes\": " << std::fixed << std::setprecision(2) << speedup << ",\n";
    file << "  \"mae_vs_reference\": " << std::scientific << std::setprecision(6) << mae << ",\n";
    file << "  \"validation_passed\": " << (mae < 1e-2 ? "true" : "false") << "\n";
    file << "}\n";
    
    file.close();
    
    std::cout << "\n=== Performance Report ===" << std::endl;
    std::cout << "Matrix shape: " << metadata.num_rows << " x " << metadata.num_cols << std::endl;
    std::cout << "Kernel time: " << std::fixed << std::setprecision(4) << kernel_time_ms << " ms" << std::endl;
    std::cout << "Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "MAE: " << std::scientific << std::setprecision(6) << mae << std::endl;
    std::cout << "Validation: " << (mae < 1e-2 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "=========================" << std::endl;
    
    return true;
}

// Compute MAE between output and reference.
// is_bf16: when true BOTH arrays are decoded as bfloat16; otherwise both as fp16.
// datagen.py saves reference.bin in the same dtype as the expected kernel output,
// so decoding both with the same function gives an apples-to-apples comparison.
float compute_mae(const uint16_t* output, const uint16_t* reference,
                  size_t num_elements, bool is_bf16) {
    double sum_abs_error = 0.0;
    auto decode = is_bf16 ? bf16_to_float : fp16_to_float;
    
    for (size_t i = 0; i < num_elements; ++i) {
        sum_abs_error += std::abs(decode(output[i]) - decode(reference[i]));
    }
    
    return static_cast<float>(sum_abs_error / num_elements);
}

// Load reference data from binary file
bool load_reference(const std::string& filename, std::vector<uint16_t>& reference, 
                    size_t expected_elements) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open reference file: " << filename << std::endl;
        return false;
    }
    
    reference.resize(expected_elements);
    file.read(reinterpret_cast<char*>(reference.data()), expected_elements * sizeof(uint16_t));
    file.close();
    
    std::cout << "Loaded reference data: " << expected_elements << " fp16 values" << std::endl;
    
    return true;
}
