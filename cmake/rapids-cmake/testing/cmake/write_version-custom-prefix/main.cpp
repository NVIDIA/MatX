/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <demo_version.hpp>
#include <nested_version.hpp>
#include <type_traits>

constexpr int dmajor = DEMO_VERSION_MAJOR;
constexpr int dminor = DEMO_VERSION_MINOR;
constexpr int dpatch = DEMO_VERSION_PATCH;

constexpr int nmajor = NESTED_VERSION_MAJOR;
constexpr int nminor = NESTED_VERSION_MINOR;
constexpr int npatch = NESTED_VERSION_PATCH;

int main()
{
  static_assert(dmajor == 3);
  static_assert(dminor == 2);
  static_assert(dpatch == 0);

  static_assert(nmajor == 3);
  static_assert(nminor == 2);
  static_assert(npatch == 0);
  return 0;
}
