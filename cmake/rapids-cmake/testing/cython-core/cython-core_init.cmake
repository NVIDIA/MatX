# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cython-core/init.cmake)

# Test that rapids_cython_init initializes the expected variables.
rapids_cython_init()
if(NOT DEFINED RAPIDS_CYTHON_INITIALIZED)
  message(FATAL_ERROR "rapids_cython_init didn't correctly set RAPIDS_CYTHON_INITIALIZED")
endif()

string(REGEX MATCHALL ".*--directive.*" matches "${CYTHON_FLAGS}")
list(LENGTH matches num_directives)

if(NOT CYTHON_FLAGS OR NOT num_directives EQUAL 1)
  message(FATAL_ERROR "rapids_cython_init didn't correctly set CYTHON_FLAGS")
endif()

# Test that rapids_cython_init is idempotent.
rapids_cython_init()
string(REGEX MATCHALL ".*--directive.*" matches "${CYTHON_FLAGS}")
list(LENGTH matches num_directives)

if(NOT num_directives EQUAL 1)
  message(FATAL_ERROR "rapids_cython_init is not idempotent, num_directives = ${num_directives}")
endif()

# Unset the cached CYTHON_FLAGS variable for future runs of the test to behave as expected.
unset(CYTHON_FLAGS CACHE)
