cmake_minimum_required(VERSION 3.18)

set (test_sources
    00_tensor/BasicTensorTests.cu
    00_tensor/CUBTests.cu
    00_tensor/ViewTests.cu
    00_tensor/VizTests.cu
    00_tensor/TensorCreationTests.cu
    00_tensor/EinsumTests.cu
    00_operators/OperatorTests.cu
    00_operators/GeneratorTests.cu
    00_operators/ReductionTests.cu
    00_transform/ConvCorr.cu
    00_transform/MatMul.cu
    00_transform/Copy.cu
    00_transform/Cov.cu
    00_transform/FFT.cu
    00_transform/Solve.cu
    00_solver/Cholesky.cu
    00_solver/LU.cu
    00_solver/QR.cu
    00_solver/QR2.cu
    00_solver/SVD.cu
    00_solver/Eigen.cu
    00_solver/Det.cu
    00_solver/Inverse.cu
    00_operators/PythonEmbed.cu
    00_io/FileIOTests.cu
    01_radar/MultiChannelRadarPipeline.cu
    01_radar/MVDRBeamformer.cu
    01_radar/ambgfun.cu
    01_radar/dct.cu
    main.cu
)

# Find proprietary parameters
file (GLOB_RECURSE proprietary_sources ../proprietary/*/tests/*.cu)
foreach (ptest ${proprietary_sources})
    get_filename_component(incdir ${ptest} DIRECTORY)
    list(APPEND proprietary_inc_list ${incdir}/../examples)
endforeach()

set(target_inc ${CMAKE_SOURCE_DIR}/test/include ${CMAKE_SOURCE_DIR}/examples/ ${proprietary_inc_list})
set(system_inc ${CUTLASS_INC} ${GTEST_INC_DIRS} ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

set(all_test_srcs ${test_sources} ${proprietary_sources})

add_executable(matx_test main.cu ${all_test_srcs})

# Set all the flags/other properties
set_property(TARGET matx_test PROPERTY ENABLE_EXPORTS 1)

if (MSVC)
    target_compile_options(matx_test  PRIVATE /W4 /WX)
else()
    target_compile_options(matx_test PRIVATE ${WARN_FLAGS})
    #target_compile_options(matx_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${WARN_FLAGS}>)
    target_compile_options(matx_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MATX_CUDA_FLAGS}>)
endif()

target_include_directories(matx_test PRIVATE "${target_inc}")
target_include_directories(matx_test SYSTEM PRIVATE "${system_inc}")
target_link_libraries(matx_test PRIVATE matx::matx) # Transitive properties
target_link_libraries(matx_test PRIVATE ${NVSHMEM_LIBRARY} gtest)

add_custom_target(test
    DEPENDS matx_test
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/matx_test)




