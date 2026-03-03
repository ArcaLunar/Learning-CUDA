cmake_minimum_required(VERSION 3.14)
project(NF4Dequant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CUDA architecture (adjust based on your GPU)
# sm_70: Volta (V100), sm_75: Turing (T4), sm_80: Ampere (A100), sm_86: Ampere (RTX 30xx)
set(CMAKE_CUDA_ARCHITECTURES 75)  # T4 as specified in params.txt

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")  # Print register usage

# C++ compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Source files
set(SOURCES
    src/main.cpp
    src/nf4_kernel.cu
    src/io/read_params.cpp
    src/io/read_weights.cpp
    src/io/write_output.cpp
)

# Create executable
add_executable(nf4_dequant ${SOURCES})

# Link CUDA libraries
target_link_libraries(nf4_dequant cudart)

# Set output directory
set_target_properties(nf4_dequant PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")