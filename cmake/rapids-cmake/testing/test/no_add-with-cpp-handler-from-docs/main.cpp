/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <vector>

#include "rapids_cmake_ctest_allocation.cpp"
#include "rapids_cmake_ctest_allocation.hpp"

int main()
{
  // Verify we only have a single GPU visible to us
  auto allocs = rapids_cmake::full_allocation();

  if (allocs.size() != 1) { return 1; }

  auto alloc = allocs[0];
  if (alloc.slots != 25) { return 1; }

  return 0;
}
