# ////////////////////////////////////////////////////////////////////////////////
# // BSD 3-Clause License
# //
# // Copyright (c) 2021, NVIDIA Corporation
# // All rights reserved.
# //
# // Redistribution and use in source and binary forms, with or without
# // modification, are permitted provided that the following conditions are met:
# //
# // 1. Redistributions of source code must retain the above copyright notice, this
# //    list of conditions and the following disclaimer.
# //
# // 2. Redistributions in binary form must reproduce the above copyright notice,
# //    this list of conditions and the following disclaimer in the documentation
# //    and/or other materials provided with the distribution.
# //
# // 3. Neither the name of the copyright holder nor the names of its
# //    contributors may be used to endorse or promote products derived from
# //    this software without specific prior written permission.
# //
# // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# /////////////////////////////////////////////////////////////////////////////////
# Taken from cuDF cmake
function(find_and_configure_gtest VERSION)

    if(TARGET GTest::gtest)
        return()
    endif()

    # Set GoogleTest-specific options
    set(GTEST_OPTIONS "INSTALL_GTEST ON")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        # Force PIC for GoogleTest on ARM64 to avoid relocation issues
        list(APPEND GTEST_OPTIONS "CMAKE_POSITION_INDEPENDENT_CODE ON")
        message(STATUS "Enabling PIC for GoogleTest on ARM64")
    endif()

    # Find or install GoogleTest
    CPMFindPackage(NAME GTest
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/google/googletest.git
        GIT_TAG         v${VERSION}
        GIT_SHALLOW     TRUE
        OPTIONS         ${GTEST_OPTIONS}
        # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
        FIND_PACKAGE_ARGUMENTS "CONFIG")
    # Add GTest aliases if they don't already exist.
    # Assumes if GTest::gtest doesn't exist, the others don't either.
    # TODO: Is this always a valid assumption?
    if(NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
        add_library(GTest::gmock ALIAS gmock)
        add_library(GTest::gtest_main ALIAS gtest_main)
        add_library(GTest::gmock_main ALIAS gmock_main)
    endif()

    # Make sure consumers of cudf can also see GTest::* targets
    set(GTest::gtest GTest::gtest PARENT_SCOPE)
    set(GTest::gmock GTest::gmock PARENT_SCOPE)
    set(GTest::gtest_main GTest::gtest_main PARENT_SCOPE)
    set(GTest::gmock_main GTest::gmock_main PARENT_SCOPE)
    # fix_cmake_global_defaults(GTest::gtest)
    # fix_cmake_global_defaults(GTest::gmock)
    # fix_cmake_global_defaults(GTest::gtest_main)
    # fix_cmake_global_defaults(GTest::gmock_main)
endfunction()

set(CUDF_MIN_VERSION_GTest 1.17.0)

find_and_configure_gtest(${CUDF_MIN_VERSION_GTest})
