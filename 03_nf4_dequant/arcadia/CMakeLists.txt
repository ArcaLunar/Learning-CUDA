cmake_minimum_required(VERSION 3.14)
project(NF4Dequant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NF4_USE_CODE2_CONST "Use constant memory for code2 LUT" ON)
set(NF4_SCHEDULE_MODE "dynamic" CACHE STRING "Kernel schedule mode: dynamic, force_strided, force_contiguous")
set_property(CACHE NF4_SCHEDULE_MODE PROPERTY STRINGS dynamic force_strided force_contiguous)
set(NF4_CONTIGUOUS_THRESHOLD_BYTES 1048576 CACHE STRING "Byte threshold for contiguous schedule when dynamic scheduling is enabled")
set(NF4_CONTIGUOUS_SMEM_LEVEL 0 CACHE STRING "Contiguous kernel shared-memory level: 0=off, 1=absmax2, 2=absmax2+absmax_q")

if(NOT NF4_SCHEDULE_MODE MATCHES "^(dynamic|force_strided|force_contiguous)$")
    message(FATAL_ERROR "NF4_SCHEDULE_MODE must be one of: dynamic, force_strided, force_contiguous")
endif()

if(NOT NF4_CONTIGUOUS_SMEM_LEVEL MATCHES "^[012]$")
    message(FATAL_ERROR "NF4_CONTIGUOUS_SMEM_LEVEL must be one of: 0, 1, 2")
endif()

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
    src/main.cu
    src/nf4_kernel.cu
    src/io/read_params.cpp
    src/io/read_weights.cpp
    src/io/write_output.cpp
)

# Create executable
add_executable(nf4_dequant ${SOURCES})

if(NF4_USE_CODE2_CONST)
    target_compile_definitions(nf4_dequant PRIVATE NF4_USE_CODE2_CONST=1)
else()
    target_compile_definitions(nf4_dequant PRIVATE NF4_USE_CODE2_CONST=0)
endif()

if(NF4_SCHEDULE_MODE STREQUAL "dynamic")
    target_compile_definitions(nf4_dequant PRIVATE NF4_SCHEDULE_MODE=0)
elseif(NF4_SCHEDULE_MODE STREQUAL "force_strided")
    target_compile_definitions(nf4_dequant PRIVATE NF4_SCHEDULE_MODE=1)
elseif(NF4_SCHEDULE_MODE STREQUAL "force_contiguous")
    target_compile_definitions(nf4_dequant PRIVATE NF4_SCHEDULE_MODE=2)
endif()

target_compile_definitions(nf4_dequant PRIVATE NF4_CONTIGUOUS_THRESHOLD_BYTES=${NF4_CONTIGUOUS_THRESHOLD_BYTES})
target_compile_definitions(nf4_dequant PRIVATE NF4_CONTIGUOUS_SMEM_LEVEL=${NF4_CONTIGUOUS_SMEM_LEVEL})

# Link CUDA libraries
target_link_libraries(nf4_dequant cudart)

# Set output directory
set_target_properties(nf4_dequant PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "NF4_USE_CODE2_CONST: ${NF4_USE_CODE2_CONST}")
message(STATUS "NF4_SCHEDULE_MODE: ${NF4_SCHEDULE_MODE}")
message(STATUS "NF4_CONTIGUOUS_THRESHOLD_BYTES: ${NF4_CONTIGUOUS_THRESHOLD_BYTES}")
message(STATUS "NF4_CONTIGUOUS_SMEM_LEVEL: ${NF4_CONTIGUOUS_SMEM_LEVEL}")