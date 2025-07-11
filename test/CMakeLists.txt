cmake_minimum_required(VERSION 3.18)

include(00_operators/CMakeLists.txt)
list(TRANSFORM OPERATOR_TEST_FILES PREPEND "00_operators/")

set (test_sources
    00_tensor/BasicTensorTests.cu
    00_tensor/CUBTests.cu
    00_tensor/ViewTests.cu
    00_tensor/VizTests.cu
    00_tensor/TensorCreationTests.cu
    00_tensor/EinsumTests.cu
    ${OPERATOR_TEST_FILES}
    00_operators/GeneratorTests.cu
    00_operators/PWelch.cu
    00_operators/ReductionTests.cu
    00_transform/ConvCorr.cu
    00_transform/MatMul.cu
    00_transform/ChannelizePoly.cu
    00_transform/Copy.cu
    00_transform/Cov.cu
    00_transform/FFT.cu
    00_transform/Norm.cu
    00_transform/ResamplePoly.cu
    00_transform/Solve.cu
    00_solver/Cholesky.cu
    00_solver/LU.cu
    00_solver/QR.cu
    00_solver/QR2.cu
    00_solver/QREcon.cu
    00_solver/SVD.cu
    00_solver/Eigen.cu
    00_solver/Det.cu
    00_solver/Inverse.cu
    00_solver/Pinv.cu
    00_operators/PythonEmbed.cu
    00_io/FileIOTests.cu
    00_io/PrintTests.cu
    00_io/NvtxTests.cu
    01_radar/MultiChannelRadarPipeline.cu
    01_radar/MVDRBeamformer.cu
    01_radar/ambgfun.cu
    01_radar/dct.cu
    00_sparse/Basic.cu
    00_sparse/Convert.cu
    00_sparse/Dia.cu
    00_sparse/Matmul.cu
    00_sparse/Matvec.cu
    00_sparse/Solve.cu
    main.cu
)

# Some of <00_io> tests need csv files and binaries which all
# are located under 'CMAKE_CURRENT_SOURCE_DIR/00_io'. When calling the test
# executable <matx_test> from its location in 'CMAKE_BINARY_DIR/test' the
# search paths according <FileIOTests.cu> are
# '../test/00_io/small_csv_comma_nh.csv' and
# '../test/00_io/small_csv_complex_comma_nh.csv' respectively. Therefore
# they must be copied to the correct location:
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test/00_io)
file(COPY
	${CMAKE_CURRENT_SOURCE_DIR}/00_io/small_csv_comma_nh.csv
	DESTINATION ${CMAKE_BINARY_DIR}/test/00_io
)
file(COPY
	${CMAKE_CURRENT_SOURCE_DIR}/00_io/small_csv_complex_comma_nh.csv
	DESTINATION ${CMAKE_BINARY_DIR}/test/00_io
)
file(COPY
	${CMAKE_CURRENT_SOURCE_DIR}/00_io/test.mat
	DESTINATION ${CMAKE_BINARY_DIR}/test/00_io
)
file(COPY
	${CMAKE_CURRENT_SOURCE_DIR}/00_io/test.npy
	DESTINATION ${CMAKE_BINARY_DIR}/test/00_io
)

# Find proprietary parameters
file (GLOB_RECURSE proprietary_sources ../proprietary/*/tests/*.cu)
foreach (ptest ${proprietary_sources})
    get_filename_component(incdir ${ptest} DIRECTORY)
    list(APPEND proprietary_inc_list ${incdir}/../examples)
endforeach()

set(target_inc ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../examples/ ${proprietary_inc_list})
set(system_inc ${CUTLASS_INC} ${GTEST_INC_DIRS} ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

set(all_test_srcs ${test_sources} ${proprietary_sources})

add_executable(matx_test main.cu ${all_test_srcs})

# Set all the flags/other properties
set_property(TARGET matx_test PROPERTY ENABLE_EXPORTS 1)

if (DEFINED cupy_PYTHON_PACKAGE)
    target_compile_definitions(matx_test PRIVATE CUPY_INSTALLED)
endif()

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

# Enable position independent code to fix relocation issues on ARM64
set_property(TARGET matx_test PROPERTY POSITION_INDEPENDENT_CODE ON)

# For ARM64, prefer dynamic linking to avoid relocation issues
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set_property(TARGET matx_test PROPERTY LINK_PREFERENCE_PUBLIC SHARED)
endif()

add_custom_target(test
    DEPENDS matx_test
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/matx_test)
