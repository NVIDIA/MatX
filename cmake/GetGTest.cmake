# /////////////////////////////////////////////////////////////////////////////////////////
# // This code contains NVIDIA Confidential Information and is disclosed
# // under the Mutual Non-Disclosure Agreement.
# //
# // Notice
# // ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
# // NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
# // THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
# // MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# //
# // NVIDIA Corporation assumes no responsibility for the consequences of use of such
# // information or for any infringement of patents or other rights of third parties that may
# // result from its use. No license is granted by implication or otherwise under any patent
# // or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
# // expressly authorized by NVIDIA. Details are subject to change without notice.
# // This code supersedes and replaces all information previously supplied.
# // NVIDIA Corporation products are not authorized for use as critical
# // components in life support devices or systems without express written approval of
# // NVIDIA Corporation.
# //
# // Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# //
# // NVIDIA Corporation and its licensors retain all intellectual property and proprietary
# // rights in and to this software and related documentation and any modifications thereto.
# // Any use, reproduction, disclosure or distribution of this software and related
# // documentation without an express license agreement from NVIDIA Corporation is
# // strictly prohibited.
# //
# /////////////////////////////////////////////////////////////////////////////////////////
# Taken from cuDF cmake
function(find_and_configure_gtest VERSION)

    if(TARGET GTest::gtest)
        return()
    endif()

    # Find or install GoogleTest
    CPMFindPackage(NAME GTest
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/google/googletest.git
        GIT_TAG         release-${VERSION}
        GIT_SHALLOW     TRUE
        OPTIONS         "INSTALL_GTEST ON"
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

set(CUDF_MIN_VERSION_GTest 1.10.0)

find_and_configure_gtest(${CUDF_MIN_VERSION_GTest})