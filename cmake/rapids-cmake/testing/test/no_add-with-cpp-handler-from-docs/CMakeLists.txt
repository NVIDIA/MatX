# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.20)
project(rapids-test-project LANGUAGES CXX CUDA)

include(${rapids-cmake-dir}/rapids-test.cmake)

rapids_test_init()

find_package(CUDAToolkit REQUIRED)
add_executable(verify_docs main.cpp)
target_compile_features(verify_docs PRIVATE cxx_std_17)

target_include_directories(verify_docs PRIVATE "${rapids-cmake-dir}/../docs/cpp_code_snippets/")
target_link_libraries(verify_docs PRIVATE CUDA::cudart)

enable_testing()
add_test(NAME simple_test COMMAND verify_docs)
rapids_test_gpu_requirements(simple_test GPUS 1 PERCENT 25)
