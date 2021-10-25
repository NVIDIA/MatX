# This is a sample application to show how to include matx into your project using CMake

cmake_minimum_required(VERSION 3.18)

project(SAMPLE_MATX LANGUAGES CUDA CXX)
find_package(CUDAToolkit 11.5 REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 70)

find_package(matx CONFIG REQUIRED)

add_executable(sample_matx main.cu)
target_link_libraries(sample_matx PRIVATE matx::matx)
