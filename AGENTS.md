## Overview
MatX is a C++20 numerical computing library targeting both CPUs and NVIDIA GPUs. It tries to use syntax familiar to both Python and MATLAB users.

## Environment Setup
MatX is a header-only library, but building examples, tests, and benchmarks requires a build environment with CMake, CUDA Toolkit, and a supported C++ host compiler. 
The current versions required for each dependency are listed in @docs_input/build.rst. For a reproducible build environment, users can build the Docker container in 
the @.devcontainer directory using the create_base_container.sh script. The python library `hpccm` is required to create the Dockerfile.

## Compiling and Running
For use in external projects, MatX simply needs to be included by `#include <matx.h>`. To build unit tests the use the CMake option `MATX_BUILD_TESTS=ON`, benchmarks 
`MATX_BUILD_BENCHMARKS=ON`, and examples `MATX_BUILD_EXAMPLES=ON`. Individual tests can be compiled separately via different targets output from CMake.

## Development Expectations
Every new public function, operator, transform, or backend path should include accompanying unit tests and documentation updates. Tests should cover the supported
types, ranks, batching behavior, and important error or fallback cases for the new behavior. Documentation should describe how to use the feature, required build
options or dependencies, and any limitations that would affect users or future maintainers. Any change to operators should also verify
@docs_input/executor_compatibility.rst and update it if executor support, limitations, or backend requirements changed.

When adding or changing accelerated backends such as CUDA, cuBLAS, cuSolver, cuFFT, or MathDx paths, preserve the existing non-accelerated behavior unless the task
explicitly calls for a breaking change. Prefer focused tests that compare the new backend against an existing trusted MatX path, and include negative tests for
unsupported shape, dtype, rank, or configuration combinations when applicable.
