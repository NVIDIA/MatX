# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
# can't have an include guard on this file that breaks its usage by cpm/detail/package_details

if(NOT DEFINED rapids-cmake-checkout-tag)
  file(READ "${CMAKE_CURRENT_LIST_DIR}/../VERSION" _rapids_version)
  if(_rapids_version MATCHES [[^([0-9][0-9])\.([0-9][0-9])\.([0-9][0-9])(a?)]])
    set(RAPIDS_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(RAPIDS_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(RAPIDS_VERSION_PATCH "${CMAKE_MATCH_3}")
    set(RAPIDS_USE_MAIN "${CMAKE_MATCH_4}")
    set(RAPIDS_VERSION_MAJOR_MINOR "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}")
    set(RAPIDS_VERSION "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}.${RAPIDS_VERSION_PATCH}")
  else()
    string(REPLACE "\n" "\n  " _rapids_version_formatted "  ${_rapids_version}")
    message(FATAL_ERROR "Could not determine RAPIDS version. Contents of VERSION file:\n${_rapids_version_formatted}"
    )
  endif()

  set(rapids-cmake-version ${RAPIDS_VERSION_MAJOR_MINOR})
endif()

if(NOT DEFINED rapids-cmake-checkout-tag)
  # Use STRINGS to trim whitespace/newlines
  file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../RAPIDS_BRANCH" _rapids_checkout)
  if(NOT _rapids_checkout)
    message(FATAL_ERROR "Could not determine branch name to use for checking out rapids-cmake. The file \"${CMAKE_CURRENT_LIST_DIR}/../RAPIDS_BRANCH\" is missing."
    )
  endif()
  set(rapids-cmake-checkout-tag "${_rapids_checkout}")
endif()
