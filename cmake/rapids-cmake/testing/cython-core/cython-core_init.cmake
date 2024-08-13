#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
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
