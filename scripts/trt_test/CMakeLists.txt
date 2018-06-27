cmake_minimum_required(VERSION 3.1)
project(trt_test)

find_package(CUDA REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(${CMAKE_SOURCE_DIR})

set(CUDA_NVCC_FLAGS --std=c++11)
set(CMAKE_CXX_STANDARD 11)

add_executable(trt_test trt_test.cc)
target_link_libraries(trt_test nvinfer nvparsers cudart)
