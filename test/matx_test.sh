#!/bin/bash

echo "MatX Test Wrapper - Running CTest in parallel..."

# Change to the directory where this script is located (test directory)
cd "$(dirname "$0")"

# Build the ctest command with parallel jobs and output on failure
CTEST_CMD="ctest -j4 --output-on-failure"

# Forward any additional arguments to ctest
if [ $# -gt 0 ]; then
    CTEST_CMD="$CTEST_CMD $*"
fi

echo "Executing: $CTEST_CMD"

# Execute ctest and preserve its exit code
exec $CTEST_CMD 