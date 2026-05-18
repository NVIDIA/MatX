/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <demo_git_version.hpp>
#include <nested_git_version.h>
#include <type_traits>

constexpr const char* dbranch = DEMO_GIT_BRANCH;
constexpr const char* dsha1   = DEMO_GIT_SHA1;
constexpr const char* dvers   = DEMO_GIT_VERSION;
#if defined(DEMO_GIT_IS_DIRTY)
constexpr const bool disdirty = true;
#else
constexpr const bool disdirty = false;
#endif

constexpr const char* nbranch = NESTED_GIT_BRANCH;
constexpr const char* nsha1   = NESTED_GIT_SHA1;
constexpr const char* nvers   = DEMO_GIT_VERSION;
#if defined(NESTED_GIT_IS_DIRTY)
constexpr const bool nisdirty = true;
#else
constexpr const bool nisdirty = false;
#endif

int main()
{
  static_assert(dbranch == nbranch);
  static_assert(dsha1 == nsha1);
  static_assert(disdirty == nisdirty);
  static_assert(dvers == nvers);

  return 0;
}
