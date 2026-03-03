#ifndef NF4_IO_H
#define NF4_IO_H

#include "nf4_types.h"
#include <string>
#include <vector>

// Read configuration parameters from text file
bool read_params(const std::string& filename, NF4Config& config);

// Read quantized weights from binary file
bool read_weights(const std::string& filename, NF4QuantizedWeights& weights);

// Write output data to binary file
bool write_output(const std::string& output_file, const void* data, 
                  size_t size_bytes, const std::string& data_type);

// Write performance log to file
bool write_performance_log(const std::string& log_file, 
                           float kernel_time_ms,
                           float bandwidth_gb_s,
                           float speedup,
                           float mae,
                           const WeightMetadata& metadata);

// Compute mean absolute error
float compute_mae(const uint16_t* output, const uint16_t* reference, size_t num_elements);

// Load reference data from binary file
bool load_reference(const std::string& filename, std::vector<uint16_t>& reference, 
                    size_t expected_elements);

#endif // NF4_IO_H
