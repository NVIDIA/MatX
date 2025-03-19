set (bench_sources
    00_transform/matmul.cu
    00_transform/fft.cu
    00_transform/conv.cu
    00_transform/cub.cu
    00_transform/einsum.cu
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

# Compile all the unit tests into an object first since pybind needs to use its own version of
# add_module/library
add_executable(matx_bench ${bench_sources})

target_link_libraries(matx_bench PRIVATE nvbench::main)
target_link_libraries(matx_bench PRIVATE matx::matx)

# Set all the flags/other properties
set_property(TARGET matx_bench PROPERTY ENABLE_EXPORTS 1)

if (MSVC)
    target_compile_options(matx_bench PRIVATE /W4 /WX)
else()
    target_compile_options(matx_bench PRIVATE ${WARN_FLAGS})
    target_compile_options(matx_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MATX_CUDA_FLAGS}>)
endif()

target_include_directories(matx_bench PRIVATE "${target_inc}")
target_include_directories(matx_bench SYSTEM PRIVATE "${system_inc}")

target_include_directories(matx_bench SYSTEM PRIVATE "${pybind11_INCLUDE_DIR}" "${PYTHON_INCLUDE_DIRS}")
target_link_libraries(  matx_bench PRIVATE matx::matx)

add_custom_target(bench
    DEPENDS matx_bench
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/matx_bench)


