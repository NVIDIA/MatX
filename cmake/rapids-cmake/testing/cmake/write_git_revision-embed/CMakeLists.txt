# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/write_git_revision_file.cmake)

cmake_minimum_required(VERSION 3.30.4)

project(DEMO VERSION 2.4 LANGUAGES CXX)
rapids_cmake_write_git_revision_file(git_generated_header
                                     "${CMAKE_CURRENT_BINARY_DIR}/demo/git_version.hpp")

add_executable(embed_git_version main.cpp)
target_link_libraries(embed_git_version PRIVATE git_generated_header)
target_compile_features(embed_git_version PRIVATE cxx_std_14)

# verify that the git strings have been embedded.

find_package(Git QUIET)

add_custom_command(TARGET embed_git_version
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -DWORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}
                           -DEXECUTABLE=$<TARGET_FILE:embed_git_version>
                           -DGIT_EXECUTABLE=${GIT_EXECUTABLE} -P
                           ${CMAKE_CURRENT_SOURCE_DIR}/verify_embedding.cmake)
