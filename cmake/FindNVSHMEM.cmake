# Find NVSHMEM host and device libraries for cuFFTMp.

find_package(NVSHMEM CONFIG QUIET
  HINTS ${nvshmem_DIR} ENV NVSHMEM_PREFIX ENV NVSHMEM_HOME)
if(TARGET nvshmem::nvshmem_host AND TARGET nvshmem::nvshmem_device)
  set(NVSHMEM_FOUND TRUE)
  return()
endif()

find_path(NVSHMEM_INCLUDE_DIR
  NAMES nvshmem.h
  HINTS ${nvshmem_DIR} ENV NVSHMEM_PREFIX ENV NVSHMEM_HOME ENV NVSHMEM_INC
  PATH_SUFFIXES include)

find_library(NVSHMEM_HOST_LIBRARY
  NAMES nvshmem_host
  HINTS ${nvshmem_DIR} ENV NVSHMEM_PREFIX ENV NVSHMEM_HOME ENV NVSHMEM_LIB
  PATH_SUFFIXES lib lib64)

find_library(NVSHMEM_DEVICE_LIBRARY
  NAMES nvshmem_device
  HINTS ${nvshmem_DIR} ENV NVSHMEM_PREFIX ENV NVSHMEM_HOME ENV NVSHMEM_LIB
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVSHMEM
  REQUIRED_VARS
    NVSHMEM_INCLUDE_DIR
    NVSHMEM_HOST_LIBRARY
    NVSHMEM_DEVICE_LIBRARY)

if(NVSHMEM_FOUND)
  if(NOT TARGET nvshmem::nvshmem_host)
    add_library(nvshmem::nvshmem_host UNKNOWN IMPORTED)
    set_target_properties(nvshmem::nvshmem_host PROPERTIES
      IMPORTED_LOCATION "${NVSHMEM_HOST_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}")
  endif()

  if(NOT TARGET nvshmem::nvshmem_device)
    add_library(nvshmem::nvshmem_device UNKNOWN IMPORTED)
    set_target_properties(nvshmem::nvshmem_device PROPERTIES
      IMPORTED_LOCATION "${NVSHMEM_DEVICE_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(
  NVSHMEM_INCLUDE_DIR
  NVSHMEM_HOST_LIBRARY
  NVSHMEM_DEVICE_LIBRARY)
