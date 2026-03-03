#include "../include/nf4_types.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// Helper function to trim whitespace from string
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

// Helper function to remove quotes from string
static std::string remove_quotes(const std::string& str) {
    std::string result = str;
    result.erase(std::remove(result.begin(), result.end(), '\"'), result.end());
    return result;
}

bool read_params(const std::string& filename, NF4Config& config) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open params file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        
        // Parse key = value
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));
        value = remove_quotes(value);
        
        if (key == "blocksize") {
            config.blocksize = std::stoi(value);
        } else if (key == "compute_type") {
            config.compute_type = value;
        } else if (key == "target_gpu") {
            config.target_gpu = value;
        }
    }
    
    file.close();
    
    std::cout << "Loaded config: blocksize=" << config.blocksize 
              << ", compute_type=" << config.compute_type
              << ", target_gpu=" << config.target_gpu << std::endl;
    
    return true;
}
