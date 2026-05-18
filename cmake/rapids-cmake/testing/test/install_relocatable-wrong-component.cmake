# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/test/init.cmake)
include(${rapids-cmake-dir}/test/install_relocatable.cmake)

enable_language(CUDA)
rapids_test_init()

rapids_test_install_relocatable(INSTALL_COMPONENT_SET wrong_component DESTINATION bin/testing)
rapids_test_install_relocatable(INSTALL_COMPONENT_SET another_wrong_component
                                DESTINATION bin/testing)
