#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

export RAPIDS_VERSION_NUMBER="${RAPIDS_VERSION_MAJOR_MINOR}"
RAPIDS_DOCS_DIR="$(mktemp -d)"
export RAPIDS_DOCS_DIR

rapids-logger "Build Sphinx docs"
pushd docs
sphinx-build -b dirhtml . _html -W
sphinx-build -b text . _text -W
mkdir -p "${RAPIDS_DOCS_DIR}/rapids-cmake/"{html,txt}
mv _html/* "${RAPIDS_DOCS_DIR}/rapids-cmake/html"
mv _text/* "${RAPIDS_DOCS_DIR}/rapids-cmake/txt"
popd

rapids-upload-docs
