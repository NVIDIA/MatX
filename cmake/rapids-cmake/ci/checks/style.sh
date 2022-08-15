#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.
#############################
# rapids-cmake Style Tester #
#############################

# Ignore errors and set path
set +e
PATH=/opt/conda/bin:$PATH
LC_ALL=C.UTF-8
LANG=C.UTF-8

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Run cmake-format / cmake-lint and get results/return code
CMAKE_FILES=(`find  rapids-cmake/ | grep -E "^.*\.cmake?$|^.*/CMakeLists.txt$"`)
CMAKE_FILES+=("CMakeLists.txt")

CMAKE_FORMATS=()
CMAKE_FORMAT_RETVAL=0

CMAKE_LINTS=()
CMAKE_LINT_RETVAL=0

for cmake_file in "${CMAKE_FILES[@]}"; do
  cmake-format --in-place --first-comment-is-literal --config-files ./cmake-format-rapids-cmake.json ./ci/checks/cmake_config_format.json -- ${cmake_file}
  TMP_CMAKE_FORMAT=`git diff --color --exit-code -- ${cmake_file}`
  TMP_CMAKE_FORMAT_RETVAL=$?
  if [ "$TMP_CMAKE_FORMAT_RETVAL" != "0" ]; then
    CMAKE_FORMAT_RETVAL=1
    CMAKE_FORMATS+=("$TMP_CMAKE_FORMAT")
  fi

  TMP_CMAKE_LINT=`cmake-lint --config-files ./cmake-format-rapids-cmake.json ./ci/checks/cmake_config_format.json ./ci/checks/cmake_config_lint.json -- ${cmake_file}`
  TMP_CMAKE_LINT_RETVAL=$?
  if [ "$TMP_CMAKE_LINT_RETVAL" != "0" ]; then
    CMAKE_LINT_RETVAL=1
    CMAKE_LINTS+=("$TMP_CMAKE_LINT")
  fi
done

# Output results if failure otherwise show pass
if [ "$CMAKE_FORMAT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: cmake format check; begin output\n\n"
  for CMAKE_FORMAT in "${CMAKE_FORMATS[@]}"; do
    echo -e "$CMAKE_FORMAT"
    echo -e "\n"
  done
  echo -e "\n\n>>>> FAILED: cmake format check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: cmake format check\n\n"
fi

if [ "$CMAKE_LINT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: cmake lint check; begin output\n\n"
  for CMAKE_LINT in "${CMAKE_LINTS[@]}"; do
    echo -e "$CMAKE_LINT"
    echo -e "\n"
  done
  echo -e "\n\n>>>> FAILED: cmake lint check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: cmake lint check\n\n"
fi

RETVALS=($CMAKE_FORMAT_RETVAL $CMAKE_LINT_RETVAL)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
