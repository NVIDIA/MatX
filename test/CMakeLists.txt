cmake_minimum_required(VERSION 3.18)

include(00_operators/CMakeLists.txt)
list(TRANSFORM OPERATOR_TEST_FILES PREPEND "00_operators/")

set (test_sources
    00_misc/AllocatorTests.cu
    00_misc/ProfilingTests.cu
    00_tensor/BasicTensorTests.cu
    00_tensor/CUBTests.cu
    00_tensor/Storage.cu
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

# Function to create individual test executables
function(create_test_executable test_file)
    # Get the filename without path and extension for the target name
    get_filename_component(test_name ${test_file} NAME_WE)
    # Get the directory to make target names unique
    get_filename_component(test_dir ${test_file} DIRECTORY)
    string(REPLACE "/" "_" test_dir_clean ${test_dir})
    set(target_name "test_${test_dir_clean}_${test_name}")

    # Create executable with main.cu and the specific test file
    add_executable(${target_name} main.cu ${test_file})

    # Set all the flags/other properties
    set_property(TARGET ${target_name} PROPERTY ENABLE_EXPORTS 1)

    if (DEFINED cupy_PYTHON_PACKAGE)
        target_compile_definitions(${target_name} PRIVATE CUPY_INSTALLED)
    endif()

    if (MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /WX)
    else()
        target_compile_options(${target_name} PRIVATE ${WARN_FLAGS})
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MATX_CUDA_FLAGS}>)
    endif()

    target_include_directories(${target_name} PRIVATE "${target_inc}")
    target_include_directories(${target_name} SYSTEM PRIVATE "${system_inc}")
    target_link_libraries(${target_name} PRIVATE matx::matx) # Transitive properties
    target_link_libraries(${target_name} PRIVATE ${NVSHMEM_LIBRARY} gtest)

    # Register the test with CTest
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endfunction()

# Enable CTest
enable_testing()

# Create individual executables for each test file
foreach(test_file ${test_sources})
    create_test_executable(${test_file})
endforeach()

# Create individual executables for proprietary tests
foreach(test_file ${proprietary_sources})
    create_test_executable(${test_file})
endforeach()

# Number of test jobs to run in parallel
set(CTEST_PARALLEL_JOBS 4)

# Create a legacy matx_test script for CI compatibility
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/matx_test.sh
    ${CMAKE_CURRENT_BINARY_DIR}/matx_test
    COPYONLY
)
# Make the script executable
file(CHMOD ${CMAKE_CURRENT_BINARY_DIR}/matx_test
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

# Add a custom target to run CTest from the main build directory
add_custom_target(test
    COMMAND ${CMAKE_CTEST_COMMAND} -j${CTEST_PARALLEL_JOBS} --output-on-failure
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all MatX tests in parallel (${CTEST_PARALLEL_JOBS} cores)"
    VERBATIM
)
