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

# Try using the .pc file first in case it was installed from source
find_package(PkgConfig)

if(PkgConfig_FOUND)
  set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${blas_DIR}/share/pkgconfig")
  pkg_check_modules(BLIS QUIET blis)
  if(BLIS_FOUND)
    set(BLIS_LIBRARIES ${pkgcfg_lib_BLIS_blis})
  endif()
endif()

# If not found, search for the BLIS library directly
if(NOT BLIS_FOUND)
  find_library(BLIS_LIBRARIES NAMES blis64 blis HINTS ${blas_DIR}/lib)

  if(BLIS_LIBRARIES)
    if(BLIS_LIBRARIES MATCHES ".*blis64.*")
      # If the 64-bit index version is installed using a package manager like apt,
      # the header files are blis64.h and cblas64.h.
      set(BLIS_INCLUDE_NAME blis64.h)
      set(BLIS_64_HEADER TRUE)
    else()
      set(BLIS_INCLUDE_NAME blis.h)
    endif()

    find_path(BLIS_INCLUDE_DIRS 
      NAMES ${BLIS_INCLUDE_NAME}
      HINTS ${blas_DIR}/include
      REQUIRED
    )

    set(BLIS_FOUND TRUE)
  endif()
endif()

if(NOT BLIS_FOUND)
    message(FATAL_ERROR "BLIS not found")
endif()

if(NOT TARGET BLIS::BLIS)
  add_library(BLIS::BLIS INTERFACE IMPORTED)
  set_target_properties(BLIS::BLIS PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${BLIS_LIBRARIES}"
  )
  if(BLIS_64_HEADER)
    target_compile_definitions(matx INTERFACE MATX_BLIS_64_HEADER=1)
  endif()
endif()