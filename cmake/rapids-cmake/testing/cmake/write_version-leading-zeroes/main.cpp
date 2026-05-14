/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <type_traits>
#include <version.h>

constexpr int dmajor = DEMO_VERSION_MAJOR;
constexpr int dminor = DEMO_VERSION_MINOR;
constexpr int dpatch = DEMO_VERSION_PATCH;

int main()
{
  static_assert(dmajor == 9);
  static_assert(dminor == 8);
  static_assert(dpatch == 2);

  return 0;
}
