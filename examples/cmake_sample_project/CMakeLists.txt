# This is a sample application to show how to include matx into your project using CMake

cmake_minimum_required(VERSION 3.18)

project(SAMPLE_MATX LANGUAGES CUDA CXX)
find_package(CUDAToolkit 11.5 REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 70)

# Use this section if you're importing MatX as a CMake subdirectory
#set(MATX_EN_VISUALIZATION ON)  # Uncomment to enable visualizations
#set(MATX_EN_FILEIO ON)         # Uncomment to file IO 

# The expected use pattern is to copy the "cmake_sample_project directory" to
# a new development location and then checkout MatX within the copied folder 
# 
# alternatively you can update "MatX" below with the path to your MatX checkout
add_subdirectory(MatX)

# Use this section if you installed MatX using "make install"
#find_package(matx CONFIG REQUIRED)

add_executable(sample_matx main.cu)
target_link_libraries(sample_matx PRIVATE matx::matx)
