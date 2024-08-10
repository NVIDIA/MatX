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

set(OPENBLAS_PATH_SUFFIXES "cmake/openblas")

find_package(OpenBLAS CONFIG QUIET 
  HINTS ${blas_DIR}
  PATH_SUFFIXES ${OPENBLAS_PATH_SUFFIXES}
  REQUIRED
)

find_path(LAPACK_INCLUDE_DIR
  NAMES lapack.h
  HINTS ${OpenBLAS_INCLUDE_DIRS}
  QUIET
)

if(LAPACK_INCLUDE_DIR)
  message(STATUS "Enabling LAPACK with OpenBLAS")
  target_compile_definitions(matx INTERFACE LAPACK_COMPLEX_CUSTOM)
  target_compile_definitions(matx INTERFACE MATX_EN_OPENBLAS_LAPACK)
  include(CheckSymbolExists)
  check_symbol_exists(OPENBLAS_USE64BITINT "${OpenBLAS_INCLUDE_DIRS}/openblas_config.h" OPENBLAS_USE64BITINT_FOUND)
  if (OPENBLAS_USE64BITINT_FOUND)
    target_compile_definitions(matx INTERFACE MATX_OPENBLAS_64BITINT)
  endif()
else()
  message(STATUS "Could not find lapack.h. No LAPACK support is enabled.")
endif()


if(NOT TARGET OpenBLAS::OpenBLAS)
  add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)
  set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${OpenBLAS_LIBRARIES}"
  )
endif()