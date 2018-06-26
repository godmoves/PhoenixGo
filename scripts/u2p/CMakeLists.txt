cmake_minimum_required(VERSION 3.1)
project(trt_image_classification)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

set(CUDA_NVCC_FLAGS --std=c++11)
set(CMAKE_CXX_STANDARD 11)

add_executable(uff_to_plan uff_to_plan.cpp)
target_link_libraries(uff_to_plan nvinfer nvparsers)
