# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cpm_convert_patch_json
-------------------------------

.. versionadded:: v25.02.00


.. code-block:: cmake

  rapids_cpm_convert_patch_json( (FROM_JSON_TO_FILE|FROM_FILE_TO_JSON)
                                 <json_var>
                                 FILE_VAR <path>
                                 <PACKAGE_NAME package_name INDEX index> # required for FROM_JSON_TO_FILE
                                )

#]=======================================================================]
function(rapids_cpm_convert_patch_json)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.conver_path_json")

  set(options)
  set(one_value FROM_JSON_TO_FILE FROM_FILE_TO_JSON FILE_VAR PACKAGE_NAME INDEX)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT _RAPIDS_FILE_VAR)
    message(FATAL_ERROR "rapids_cpm_convert_patch_json required field of `FILE_VAR` is missing")
  endif()

  if(_RAPIDS_FROM_JSON_TO_FILE)
    set(json "${${_RAPIDS_FROM_JSON_TO_FILE}}")

    string(JSON type GET "${json}" "type")
    string(JSON json_content GET "${json}" "content")

    # Figure out the file path
    set(file
        "${CMAKE_BINARY_DIR}/rapids-cmake/patches/${_RAPIDS_PACKAGE_NAME}/embedded_patch_${_RAPIDS_INDEX}.${type}"
    )

    # Transform from a list of strings to a single file
    string(JSON content_length LENGTH "${json_content}")
    math(EXPR content_length "${content_length} - 1")
    unset(file_content)
    # cmake-lint: disable=E1120
    foreach(index RANGE ${content_length})
      string(JSON line GET "${json_content}" ${index})
      string(APPEND file_content "${line}\n")
    endforeach()
    file(WRITE "${file}" "${file_content}")

    set(${_RAPIDS_FILE_VAR} "${file}" PARENT_SCOPE)
  elseif(_RAPIDS_FROM_FILE_TO_JSON)
    # extract contents from `file`
    file(STRINGS "${${_RAPIDS_FILE_VAR}}" file_content)
    # Work around https://github.com/rapidsai/rapids-cmake/issues/769
    string(REPLACE "]" "~&93~" file_content "${file_content}")
    string(REPLACE "[" "~&92~" file_content "${file_content}")
    list(LENGTH file_content content_length)

    # Get the file extension
    cmake_path(GET ${_RAPIDS_FILE_VAR} EXTENSION LAST_ONLY patch_ext)
    string(SUBSTRING "${patch_ext}" 1 -1 patch_ext)

    # add each line as a json array element
    set(inline_patch [=[ [ ] ]=])
    foreach(line IN LISTS file_content)
      string(REPLACE "\"" "\\\"" line "${line}")
      string(REPLACE "~&93~" "]" line "${line}")
      string(REPLACE "~&92~" "[" line "${line}")
      string(JSON inline_patch SET "${inline_patch}" ${content_length} "\"${line}\"")
    endforeach()

    set(json_content
        [=[{
      "type" : "",
      "content" : []
    }]=])
    string(JSON json_content SET "${json_content}" "type" "\"${patch_ext}\"")
    string(JSON json_content SET "${json_content}" "content" "${inline_patch}")
    set(${_RAPIDS_FROM_FILE_TO_JSON} "${json_content}" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "rapids_cpm_convert_patch_json unsupported mode: ${mode}")
  endif()

endfunction()
