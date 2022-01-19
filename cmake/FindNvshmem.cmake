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

# Looks for nvshmem in the usual locations and sets the following variables:
# NVSHMEM_INCLUDE_DIRS -- Include directories
# NVSHMEM_LIBRARY_PATH -- Library path
# NVSHMEM_LIBRARY -- Library to link against

set(HINT_PATH /usr/local/nvshmem/ ${NVSHMEM_DIR})

find_path(NVSHMEM_INCLUDE_DIR nvshmem.h HINTS ${HINT_PATH} PATH_SUFFIXES include)
find_path(NVSHMEM_LIBRARY_PATH libnvshmem.a HINTS ${HINT_PATH} PATH_SUFFIXES lib)
find_library(NVSHMEM_LIBRARY NAMES libnvshmem.a HINTS ${NVSHMEM_LIBRARY_PATH} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nvshmem  DEFAULT_MSG NVSHMEM_LIBRARY NVSHMEM_INCLUDE_DIR NVSHMEM_LIBRARY_PATH)
mark_as_advanced(NVSHMEM_LIBRARY NVSHMEM_INCLUDE_DIR NVSHMEM_LIBRARY_PATH)
set(NVSHMEM_LIBRARIES ${NVSHMEM_LIBRARY} )
set(NVSHMEM_INCLUDE_DIRS ${NVSHMEM_INCLUDE_DIR} )
