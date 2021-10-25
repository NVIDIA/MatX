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

#Look for an executable called sphinx-build
find_program(SPHINX_EXECUTABLE
             NAMES sphinx-build
             DOC "Path to sphinx-build executable")

include(FindPackageHandleStandardArgs)

#Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(Sphinx
                                  "Failed to find sphinx-build executable"
                                  SPHINX_EXECUTABLE)