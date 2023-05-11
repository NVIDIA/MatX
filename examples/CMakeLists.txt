set(examples 
    simple_radar_pipeline 
    recursive_filter 
    convolution 
    conv2d
    cgsolve 
    fft_conv 
    resample 
    mvdr_beamformer 
    spectrogram 
    spectrogram_graph
    spherical_harmonics 
    svd_power
    qr
    black_scholes)

add_library(example_lib INTERFACE)
target_include_directories(example_lib SYSTEM INTERFACE ${CUTLASS_INC} ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS}) 

target_link_libraries(example_lib INTERFACE matx::matx) # Transitive properties                        

set_property(TARGET example_lib PROPERTY ENABLE_EXPORTS 1)

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

# Build proprietary examples
file (GLOB_RECURSE proprietary_sources CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/proprietary/*/examples/*.cu)
foreach (pexample ${proprietary_sources})
    get_filename_component(exename ${pexample} NAME_WE)
    add_executable(${exename} ${pexample})    
    target_link_libraries(${exename} example_lib)
endforeach()



