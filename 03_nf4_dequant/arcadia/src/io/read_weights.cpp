#include "../include/nf4_types.h"
#include <fstream>
#include <iostream>
#include <cstring>

bool read_weights(const std::string& filename, NF4QuantizedWeights& weights) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open weights file: " << filename << std::endl;
        return false;
    }
    
    // Read header (20 bytes total)
    file.read(reinterpret_cast<char*>(&weights.metadata.num_rows), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&weights.metadata.num_cols), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&weights.metadata.blocksize), sizeof(int32_t));
    
    // Calculate derived metadata
    weights.metadata.total_elements = weights.metadata.num_rows * weights.metadata.num_cols;
    weights.metadata.num_blocks = (weights.metadata.total_elements + weights.metadata.blocksize - 1) 
                                   / weights.metadata.blocksize;
    weights.metadata.num_groups = (weights.metadata.num_blocks + 255) / 256;  // Ceiling division by 256
    
    std::cout << "Weight matrix dimensions: " << weights.metadata.num_rows 
              << " x " << weights.metadata.num_cols << std::endl;
    std::cout << "Blocksize: " << weights.metadata.blocksize << std::endl;
    std::cout << "Number of blocks: " << weights.metadata.num_blocks << std::endl;
    std::cout << "Number of groups: " << weights.metadata.num_groups << std::endl;
    
    // Calculate sizes for each data section
    size_t packed_weights_size = weights.metadata.total_elements / 2;  // 2 indices per byte
    size_t absmax_q_size = weights.metadata.num_blocks;
    size_t absmax2_size = weights.metadata.num_groups;
    size_t code2_size = 256;
    
    // Allocate and read packed_weights
    weights.packed_weights.resize(packed_weights_size);
    file.read(reinterpret_cast<char*>(weights.packed_weights.data()), packed_weights_size);
    
    // Allocate and read absmax_q
    weights.absmax_q.resize(absmax_q_size);
    file.read(reinterpret_cast<char*>(weights.absmax_q.data()), absmax_q_size);
    
    // Allocate and read absmax2 (stored as uint16_t, represents fp16)
    weights.absmax2.resize(absmax2_size);
    file.read(reinterpret_cast<char*>(weights.absmax2.data()), absmax2_size * sizeof(uint16_t));
    
    // Allocate and read code2 (stored as uint16_t, represents fp16)
    weights.code2.resize(code2_size);
    file.read(reinterpret_cast<char*>(weights.code2.data()), code2_size * sizeof(uint16_t));
    
    // Read offset
    file.read(reinterpret_cast<char*>(&weights.offset), sizeof(float));
    
    file.close();
    
    std::cout << "Successfully loaded quantized weights" << std::endl;
    std::cout << "  Packed weights: " << packed_weights_size << " bytes" << std::endl;
    std::cout << "  Absmax_q: " << absmax_q_size << " bytes" << std::endl;
    std::cout << "  Absmax2: " << absmax2_size << " fp16 values" << std::endl;
    std::cout << "  Code2: " << code2_size << " fp16 values" << std::endl;
    std::cout << "  Offset: " << weights.offset << std::endl;
    
    return true;
}
