enable_testing()

set(bench_sources
    00_misc/fltflt_arithmetic.cu
    00_transform/matmul.cu
    00_transform/fft.cu
    00_transform/conv.cu
    00_transform/cub.cu
    00_transform/einsum.cu
    00_transform/sarbp.cu
    00_transform/svd_power.cu
    00_transform/qr.cu
    00_operators/operators.cu
    00_operators/reduction.cu
    01_radar/SingleChanSimplePipeline.cu
    00_sparse/SpMM.cu
)

set(target_inc  ${CMAKE_SOURCE_DIR}/test/include
                ${CMAKE_SOURCE_DIR}/examples/)
set(system_inc  ${CUTLASS_INC}
                ${pybind11_INCLUDE_DIR}
                ${PYTHON_INCLUDE_DIRS})

function(create_bench_executable bench_file)
    get_filename_component(bench_name ${bench_file} NAME_WE)
    get_filename_component(bench_dir ${bench_file} DIRECTORY)
    string(REPLACE "/" "_" bench_dir_clean ${bench_dir})
    set(target_name "bench_${bench_dir_clean}_${bench_name}")

    add_executable(${target_name} ${bench_file})

    set_property(TARGET ${target_name} PROPERTY ENABLE_EXPORTS 1)

    if (MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /WX)
    else()
        target_compile_options(${target_name} PRIVATE ${WARN_FLAGS})
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MATX_CUDA_FLAGS}>)
    endif()

    target_include_directories(${target_name} PRIVATE "${target_inc}")
    target_include_directories(${target_name} SYSTEM PRIVATE "${system_inc}")
    target_include_directories(${target_name} SYSTEM PRIVATE "${pybind11_INCLUDE_DIR}" "${PYTHON_INCLUDE_DIRS}")
    target_link_libraries(${target_name} PRIVATE nvbench::main matx::matx)

    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RUN_SERIAL TRUE
        LABELS bench
    )
endfunction()

foreach(bench_file ${bench_sources})
    create_bench_executable(${bench_file})
endforeach()

set(bench_targets "")
foreach(bench_file ${bench_sources})
    get_filename_component(_bn ${bench_file} NAME_WE)
    get_filename_component(_bd ${bench_file} DIRECTORY)
    string(REPLACE "/" "_" _bdc ${_bd})
    list(APPEND bench_targets "bench_${_bdc}_${_bn}")
endforeach()

set(BENCH_CTEST_PARALLEL_JOBS 4)

add_custom_target(bench
    DEPENDS ${bench_targets}
    COMMAND ${CMAKE_CTEST_COMMAND} -j${BENCH_CTEST_PARALLEL_JOBS} -R "^bench_" --output-on-failure
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all MatX benchmarks via CTest (${BENCH_CTEST_PARALLEL_JOBS} parallel jobs; each bench test is RUN_SERIAL)"
    VERBATIM
)
