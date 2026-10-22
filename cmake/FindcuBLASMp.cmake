# Find the separately distributed cuBLASMp package.

find_path(cuBLASMp_INCLUDE_DIR
  NAMES cublasMp.h cublasmp.h
  HINTS ${cublasmp_DIR} ENV CUBLASMP_HOME
  PATH_SUFFIXES include)

find_library(cuBLASMp_LIBRARY
  NAMES cublasmp
  HINTS ${cublasmp_DIR} ENV CUBLASMP_HOME
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuBLASMp
  REQUIRED_VARS cuBLASMp_INCLUDE_DIR cuBLASMp_LIBRARY)

if(cuBLASMp_FOUND AND NOT TARGET cuBLASMp::cuBLASMp)
  add_library(cuBLASMp::cuBLASMp UNKNOWN IMPORTED)
  set_target_properties(cuBLASMp::cuBLASMp PROPERTIES
    IMPORTED_LOCATION "${cuBLASMp_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${cuBLASMp_INCLUDE_DIR}")
endif()

mark_as_advanced(cuBLASMp_INCLUDE_DIR cuBLASMp_LIBRARY)
