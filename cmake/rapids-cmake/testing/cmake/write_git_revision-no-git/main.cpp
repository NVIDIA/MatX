/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <git_version.hpp>
#include <iostream>
#include <type_traits>

constexpr const char* dbranch = DEMO_GIT_BRANCH;
constexpr const char* dsha1   = DEMO_GIT_SHA1;
constexpr const char* dvers   = DEMO_GIT_VERSION;

int main()
{
  static_assert(dbranch == "unknown");
  static_assert(dsha1 == "unknown");
  static_assert(dvers == "unknown");

#ifdef DEMO_GIT_IS_DIRTY
#error "DEMO_GIT_IS_DIRTY define shouldn't exist as git shouldn't have been found"
#endif

  return 0;
}
