#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

function(find_and_configure_libcudacxx version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "libcudacxx")

  rapids_cpm_find(libcudacxx ${version}
    GLOBAL_TARGETS
      libcudacxx libcudacxx::libcudacxx cxx_static
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-exports
    CPM_ARGS
      GIT_REPOSITORY  https://github.com/NVIDIA/libcudacxx.git
      GIT_TAG         ${version}
      GIT_SHALLOW     TRUE
      DOWNLOAD_ONLY   TRUE
      OPTIONS         "LIBCXX_INCLUDE_BENCHMARKS OFF"
                      "LIBCXX_INCLUDE_TESTS OFF"
                      "LIBCXX_ENABLE_SHARED OFF"
                      "LIBCXX_ENABLE_STATIC OFF"
  )

  set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include" PARENT_SCOPE)
  set(LIBCXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/libcxx/include" PARENT_SCOPE)

endfunction()

find_and_configure_libcudacxx(${LIBCUDACXX_VERSION})
