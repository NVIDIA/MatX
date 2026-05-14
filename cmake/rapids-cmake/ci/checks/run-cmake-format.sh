#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# This script is a wrapper for cmakelang that may be used with pre-commit. The
# wrapping is necessary because RAPIDS libraries split configuration for
# cmakelang linters between a local config file and a second config file that's
# shared across all of RAPIDS via rapids-cmake. In order to keep it up to date
# this file is only maintained in one place (the rapids-cmake repo) and
# pulled down during builds. We need a way to invoke CMake linting commands
# without causing pre-commit failures (which could block local commits or CI),
# while also being sufficiently flexible to allow users to maintain the config
# file independently of a build directory.
#
# This script provides the minimal functionality to enable those use cases. It
# searches in a number of predefined locations for the rapids-cmake config file
# and exits gracefully if the file is not found. If a user wishes to specify a
# config file at a nonstandard location, they may do so by setting the
# environment variable RAPIDS_CMAKE_FORMAT_FILE.
#
# This script can be invoked directly anywhere within the project repository.
# Alternatively, it may be invoked as a pre-commit hook via
# `pre-commit run (cmake-format)|(cmake-lint)`.
#
# Usage:
# bash run-cmake-format.sh {cmake-format,cmake-lint} infile [infile ...]

RAPIDS_CMAKE_ROOT="$(realpath "$(dirname "$0")"/../..)"
DEFAULT_RAPIDS_CMAKE_FORMAT_FILE="${RAPIDS_CMAKE_ROOT}/cmake-format-rapids-cmake.json"

if [ -z ${RAPIDS_CMAKE_FORMAT_FILE:+PLACEHOLDER} ]; then
    RAPIDS_CMAKE_FORMAT_FILE="${DEFAULT_RAPIDS_CMAKE_FORMAT_FILE}"
fi

if [ -z ${RAPIDS_CMAKE_FORMAT_FILE:+PLACEHOLDER} ]; then
  echo "The rapids-cmake cmake-format configuration file was not found in the default location: "
  echo ""
  echo "${DEFAULT_RAPIDS_CMAKE_FORMAT_FILE}"
  echo ""
  echo "Try setting the environment variable RAPIDS_CMAKE_FORMAT_FILE to the path to the config file."
  exit 0
else
  echo "Using format file ${RAPIDS_CMAKE_FORMAT_FILE}"
fi

if [[ $1 == "cmake-format" ]]; then
  # We cannot pass multiple input files because of a bug in cmake-format.
  # See: https://github.com/cheshirekow/cmake_format/issues/284
  for cmake_file in "${@:2}"; do
    cmake-format --in-place --config-files "${RAPIDS_CMAKE_FORMAT_FILE}" "${RAPIDS_CMAKE_ROOT}"/ci/checks/cmake_config_format.json -- "${cmake_file}"
  done
elif [[ $1 == "cmake-lint" ]]; then
  # Since the pre-commit hook is verbose, we have to be careful to only
  # present cmake-lint's output (which is quite verbose) if we actually
  # observe a failure.
  OUTPUT=$(cmake-lint --config-files "${RAPIDS_CMAKE_FORMAT_FILE}" "${RAPIDS_CMAKE_ROOT}"/ci/checks/cmake_config_format.json "${RAPIDS_CMAKE_ROOT}"/ci/checks/cmake_config_lint.json -- "${@:2}")
  status=$?

  if ! [ ${status} -eq 0 ]; then
    echo "${OUTPUT}"
  fi
  exit ${status}
fi
