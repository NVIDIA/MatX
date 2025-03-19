set(examples
    simple_radar_pipeline
    recursive_filter
    channelize_poly_bench
    convolution
    conv2d
    cgsolve
    eigenExample
    fft_conv
    resample
    mvdr_beamformer
    pwelch
    resample_poly_bench
    sparse_tensor
    simple_stf_test
    spectrogram
    spectrogram_graph
    spherical_harmonics
    svd_power
    qr
    black_scholes
    print_styles)

add_library(example_lib INTERFACE)
target_include_directories(example_lib SYSTEM INTERFACE ${CUTLASS_INC} ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

target_link_libraries(example_lib INTERFACE matx::matx) # Transitive properties

set_property(TARGET example_lib PROPERTY ENABLE_EXPORTS 1)

if(eigen_DIR)
    include_directories(SYSTEM ${eigen_DIR})
    add_definitions(-DUSE_EIGEN)
    target_compile_definitions(example_lib INTERFACE USE_EIGEN)
endif()

if (MSVC)
    target_compile_options(example_lib INTERFACE /W4 /WX)
else()
    target_compile_options(example_lib INTERFACE ${WARN_FLAGS})
    target_compile_options(example_lib INTERFACE ${MATX_CUDA_FLAGS})
endif()

if (MULTI_GPU)
    set_target_properties(example_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()


foreach( example ${examples} )
    string( CONCAT file ${example} ".cu" )
    add_executable( ${example} ${file} )
    target_link_libraries(${example} example_lib)
endforeach()

# Compile all examples with CUDASTF and append their name with _stf
foreach( example ${examples} )
    string( CONCAT file ${example} ".cu" )

    set(output_name "${example}_stf")
    add_executable( ${output_name} ${file} )

    # Add the -DUSE_STF compilation flag
    target_compile_definitions(${output_name} PRIVATE USE_STF)

    target_link_libraries(${output_name} example_lib)
endforeach()



# Build proprietary examples
file (GLOB_RECURSE proprietary_sources CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/proprietary/*/examples/*.cu)
foreach (pexample ${proprietary_sources})
    get_filename_component(exename ${pexample} NAME_WE)
    add_executable(${exename} ${pexample})
    target_link_libraries(${exename} example_lib)
endforeach()

# Add host-compiler only example program to catch missing ifdef __CUDACC__ guards
add_executable(test_host test_host.cpp)
target_link_libraries(test_host matx::matx)

if(MATX_EN_PYBIND11)
    add_subdirectory(python_integration_sample)
endif()
