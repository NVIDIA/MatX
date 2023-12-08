#=============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/test/init.cmake)
include(${rapids-cmake-dir}/test/install_relocatable.cmake)

enable_language(CUDA)
rapids_test_init()

rapids_test_install_relocatable(INSTALL_COMPONENT_SET wrong_component
                                DESTINATION bin/testing)
rapids_test_install_relocatable(INSTALL_COMPONENT_SET another_wrong_component
                                DESTINATION bin/testing)
