#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

# Disable `sccache` S3 backend since compile times are negligible
unset SCCACHE_BUCKET

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Begin cpp tests"
cmake -S testing -B build

cd build

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

ctest -j20 --schedule-random --output-on-failure

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
