# Find the separately distributed cuSOLVERMp package.

find_path(cuSOLVERMp_INCLUDE_DIR
  NAMES cusolverMp.h
  HINTS ${cusolvermp_DIR} ENV CUSOLVERMP_HOME
  PATH_SUFFIXES include)

find_library(cuSOLVERMp_LIBRARY
  NAMES cusolverMp cusolvermp
  HINTS ${cusolvermp_DIR} ENV CUSOLVERMP_HOME
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuSOLVERMp
  REQUIRED_VARS cuSOLVERMp_INCLUDE_DIR cuSOLVERMp_LIBRARY)

if(cuSOLVERMp_FOUND AND NOT TARGET cuSOLVERMp::cuSOLVERMp)
  add_library(cuSOLVERMp::cuSOLVERMp UNKNOWN IMPORTED)
  set_target_properties(cuSOLVERMp::cuSOLVERMp PROPERTIES
    IMPORTED_LOCATION "${cuSOLVERMp_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${cuSOLVERMp_INCLUDE_DIR}")
endif()

mark_as_advanced(cuSOLVERMp_INCLUDE_DIR cuSOLVERMp_LIBRARY)
