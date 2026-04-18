set(examples
    simple_radar_pipeline
    kernel_fusion
    recursive_filter
    channelize_poly_bench
    convolution
    conv2d
    cgsolve
    eigenExample
    fft_conv
    interpolate
    resample
    mvdr_beamformer
    pwelch
    resample_poly_bench
    sparse_tensor
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

# Examples in subdirectories
add_executable(sarbp sarbp/sarbp.cu)
target_link_libraries(sarbp example_lib)

# Copy sarbp helper Python scripts next to the built executable so users can
# run them from the build/examples directory alongside sarbp.
add_custom_command(TARGET sarbp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/sarbp/cphd_to_sarbp_input.py
        $<TARGET_FILE_DIR:sarbp>/cphd_to_sarbp_input.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/sarbp/view_sarbp_image.py
        $<TARGET_FILE_DIR:sarbp>/view_sarbp_image.py
    COMMENT "Copying sarbp helper Python scripts to build directory")

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
target_compile_options(test_host PRIVATE -Wall -Werror)

if(MATX_EN_PYBIND11)
    add_subdirectory(python_integration_sample)
endif()
