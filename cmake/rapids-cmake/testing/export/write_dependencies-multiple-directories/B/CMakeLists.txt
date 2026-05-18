# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
rapids_export_cpm(BUILD SECOND_FAKE_CPM_PACKAGE test_export_set CPM_ARGS VERSION 2.0
                  GLOBAL_TARGETS ABC::ABC ABC::CBA)
