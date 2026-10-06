# Find the separately distributed cuFFTMp package.

find_path(cuFFTMp_INCLUDE_DIR
  NAMES cufftMp.h
  HINTS ${cufftmp_DIR} ENV CUFFTMP_HOME ENV CUFFT_INC
  PATH_SUFFIXES include include/cufftmp math_libs/include/cufftmp)

find_library(cuFFTMp_LIBRARY
  NAMES cufftMp
  HINTS ${cufftmp_DIR} ENV CUFFTMP_HOME ENV CUFFT_LIB
  PATH_SUFFIXES lib lib64 math_libs/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuFFTMp
  REQUIRED_VARS cuFFTMp_INCLUDE_DIR cuFFTMp_LIBRARY)

if(cuFFTMp_FOUND AND NOT TARGET cuFFTMp::cuFFTMp)
  add_library(cuFFTMp::cuFFTMp UNKNOWN IMPORTED)
  set_target_properties(cuFFTMp::cuFFTMp PROPERTIES
    IMPORTED_LOCATION "${cuFFTMp_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${cuFFTMp_INCLUDE_DIR}")
endif()

mark_as_advanced(cuFFTMp_INCLUDE_DIR cuFFTMp_LIBRARY)
