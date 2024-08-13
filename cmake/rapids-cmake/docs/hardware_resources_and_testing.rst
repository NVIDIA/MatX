
.. _rapids_resource_allocation:

Hardware Resources and Testing
##############################


Resource Allocation
*******************

CTest resource allocation frameworks allow tests to specify which hardware resources that they need, and for projects to specify the specific local/machine resources available.
Combined together this ensures that tests are told which specific resources they should use, and ensures over-subscription won't occur no matter the requested testing parallel level.

To get CTest resource allocation used by tests the following components are needed.

  - A JSON per-machine resource specification file
  - The :cmake:variable:`CTEST_RESOURCE_SPEC_FILE` points to the JSON file
  - Each :cmake:command:`add_test` records what resources it requires via test properties
  - Each test reads the relevant environment variables to determine
    what specific resources it should use


These are steep requirements that require large amounts of infrastructure setup for each project.
In addition the CTest resource allocation specification is very relaxed, allowing it to represent arbitrary requirements such as CPUs, GPUs, and ASICs.

rapids_test
***********

To help RAPIDS projects utilize all GPUs on a machine when running tests, the ``rapids-cmake`` project offers a suite of commands to simplify the process.
These commands simplify GPU detection, setting up resource specification files, specifying test requirements, and setting the active CUDA GPU.

Machine GPU Detection
*********************

The key component of CTest resource allocation is having an accurate representation of the hardware that exists on the developer's machine.
The :cmake:command:`rapids_test_init` function will do system introspection to determine the number of GPUs on the current machine and generate a resource allocation JSON file representing these GPUs.

.. code-block:: cmake

  include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

  include(rapids-test)

  enable_testing()
  rapids_test_init()

The CTest resource allocation specification isn't limited to representing GPUs as a single unit.
Instead it allows the JSON file to specify the capacity (slots) that each GPU has.
In the case of rapids-cmake we always represent each GPU as having 100 slots allowing projects to think in total percentages when calculating requirements.


Specifying Tests GPU Requirements
*********************************
As discussed above, each CMake test needs to specify the GPU resources they require to allow CTest to properly partition GPUs given the CTest parallel level.
The easiest path for for developers is to use the :cmake:command:`rapids_test_add` which wraps each execution in a wrapper script that sets the CUDA visible devices, making tests only see the allocated device(s).

For example below we have three tests, two which can run concurrently on the same GPU and one that requires a full GPU.
This specification will allow all three tests to run concurrently when a machine has 2+ GPUs with no modification of the tests!

.. code-block:: cmake

  include(rapids-test)

  enable_testing()
  rapids_test_init()

  add_executable( cuda_test test.cu )
  rapids_test_add(NAME test_small_alloc COMMAND cuda_test 50 GPUS 1 PERCENT 10)
  rapids_test_add(NAME test_medium_alloc COMMAND cuda_test 100 GPUS 1 PERCENT 20)
  rapids_test_add(NAME test_very_large_alloc COMMAND cuda_test 10000 GPUS 1)


Multi GPU Tests
***************
The :cmake:command:`rapids_test_add` command also supports tests that require multiple GPU bindings.
In that case you will need to request two (or more) GPUs with a full allocation like this:

.. code-block:: cmake

  include(rapids-test)

  enable_testing()
  rapids_test_init()

  add_executable( cuda_test test.cu )
  rapids_test_add(NAME multi_gpu COMMAND cuda_test GPUS 3)

Due to how CTest does allocations if you need distinct GPUs you need to request a percentage of 51% or higher.
Otherwise you have a chance for multiple allocations to be placed on the same GPU.

When rapids-cmake test wrapper is insufficient
**********************************************

At times the approach of using wrapper scripts is insufficient, usually due to using existing test wrappers.

As discussed above, each CMake test still needs to specify the GPU resources they require to allow CTest to properly partition GPUs given the CTest parallel level.
But in those cases the tests themselves will need to parse the CTest environment variables to extract what GPUs they should run on.

For the CMake side you can use :cmake:command:`rapids_test_gpu_requirements` to specify the requirements:

.. code-block:: cmake

  include(rapids-test)

  enable_testing()
  rapids_test_init()

  add_executable( cuda_test test.cu )
  target_link_libraries( cuda_test PRIVATE RAPIDS::test )

  add_test(NAME test_small_alloc COMMAND cuda_test 50)
  rapids_test_gpu_requirements(test_small_alloc GPUS 1 PERCENT 10)

Now in the C++ you need to parse the relevant ``CTEST_RESOURCE_GROUP`` environment variables.
To simplify the process, here is some helper C++ code that will do the heavy lifting for you:

.. literalinclude:: cpp_code_snippets/rapids_cmake_ctest_allocation.hpp
  :language: cpp

.. literalinclude:: cpp_code_snippets/rapids_cmake_ctest_allocation.cpp
  :language: cpp
