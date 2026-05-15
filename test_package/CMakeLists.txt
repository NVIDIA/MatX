cmake_minimum_required(VERSION 3.25.2)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    message(STATUS "Using native GPU architecture since CMAKE_CUDA_ARCHITECTURES not defined")
endif()

project(test_package LANGUAGES CXX CUDA)

find_package(matx CONFIG REQUIRED)

add_executable(${PROJECT_NAME} test_package.cu)

target_link_libraries(${PROJECT_NAME} PRIVATE matx::matx)
