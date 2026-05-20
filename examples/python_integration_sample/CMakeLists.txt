# This is a cmake project showing how to build a python importable library
# using pybind11, how to pass tensors between MatX and python, and
# how to call MatX operators from python

cmake_minimum_required(VERSION 3.26)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  message(WARNING "CMAKE_BUILD_TYPE not defined.  Defaulting to release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug;Release;MinSizeRel;RelWithDebInfo")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(WARNING "CMAKE_CUDA_ARCHITECTURES not defined.  Defaulting to 70")
  set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "Select compile target CUDA Compute Capabilities")
endif()

option(MATX_FETCH_REMOTE "Fetch MatX from GitHub instead of using the local repository" OFF)

project(SAMPLE_MATX_PYTHON LANGUAGES CUDA CXX)
find_package(CUDAToolkit 12.2 REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Must enable pybind11 support
set(MATX_EN_PYBIND11 ON)

# Use this section if you want to configure other MatX options
#set(MATX_EN_VISUALIZATION ON)  # Uncomment to enable visualizations
#set(MATX_EN_FILEIO ON)         # Uncomment to file IO

# Skip recursive MatX fetch
if(MATX_BUILD_EXAMPLES)
else()
  if(MATX_FETCH_REMOTE)
    include(FetchContent)
    FetchContent_Declare(
      MatX
      GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
      GIT_TAG main
    )
  else()
    include(FetchContent)
    FetchContent_Declare(
      MatX
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../
    )
  endif()
  FetchContent_MakeAvailable(MatX)
endif()

add_library(matxutil MODULE matxutil.cu)
target_link_libraries(matxutil PRIVATE matx::matx CUDA::cudart)
set_target_properties(matxutil PROPERTIES SUFFIX ".so" PREFIX "")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/mypythonlib.py
	${CMAKE_BINARY_DIR}
  COPYONLY
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/example_matxutil.py
	${CMAKE_BINARY_DIR}
  COPYONLY
)
