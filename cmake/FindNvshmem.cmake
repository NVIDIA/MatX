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

# Looks for nvshmem in the usual locations and sets the following variables:
# NVSHMEM_INCLUDE_DIRS -- Include directories
# NVSHMEM_LIBRARY_PATH -- Library path
# NVSHMEM_LIBRARY -- Library to link against

set(HINT_PATH /usr/local/nvshmem/ ${NSHMEM_DIR})

find_path(NVSHMEM_INCLUDE_DIR nvshmem.h HINTS ${HINT_PATH} PATH_SUFFIXES include)
find_path(NVSHMEM_LIBRARY_PATH libnvshmem.a HINTS ${HINT_PATH} PATH_SUFFIXES lib)
find_library(NVSHMEM_LIBRARY NAMES libnvshmem.a HINTS ${NVSHMEM_LIBRARY_PATH} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nvshmem  DEFAULT_MSG NVSHMEM_LIBRARY NVSHMEM_INCLUDE_DIR NVSHMEM_LIBRARY_PATH)
mark_as_advanced(NVSHMEM_LIBRARY NVSHMEM_INCLUDE_DIR NVSHMEM_LIBRARY_PATH)
set(NVSHMEM_LIBRARIES ${NVSHMEM_LIBRARY} )
set(NVSHMEM_INCLUDE_DIRS ${NVSHMEM_INCLUDE_DIR} )
