# This test can also be built without the GPU test suite:
# cmake -S test/driverless -B build-driverless -DCMAKE_CUDA_ARCHITECTURES=80
# cmake --build build-driverless && ctest --test-dir build-driverless
cmake_minimum_required(VERSION 3.30.4)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(MatXDriverlessTest LANGUAGES CUDA CXX)
  enable_testing()
  add_subdirectory(../.. matx)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(matx_driverless_consumer SHARED consumer.cu)
  target_link_libraries(matx_driverless_consumer PRIVATE matx::matx)
  # Reject unresolved driver references even when this shared library is never run.
  target_link_options(matx_driverless_consumer PRIVATE "LINKER:--no-undefined")

  add_executable(matx_driverless_entry_points entry_points.cpp)
  target_link_libraries(matx_driverless_entry_points PRIVATE matx::matx)
  add_test(NAME matx_driverless_entry_points COMMAND matx_driverless_entry_points)

  add_executable(matx_driverless_loader loader.cpp)
  target_link_libraries(matx_driverless_loader PRIVATE ${CMAKE_DL_LIBS})
  add_test(NAME matx_driverless_load
    COMMAND matx_driverless_loader $<TARGET_FILE:matx_driverless_consumer>)
  add_test(NAME matx_driverless_dependencies
    COMMAND ${CMAKE_COMMAND} -DREADELF=${CMAKE_READELF}
      -DLIBRARY=$<TARGET_FILE:matx_driverless_consumer>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_dependencies.cmake)
endif()
