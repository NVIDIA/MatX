/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <git_version.hpp>
#include <iostream>
#include <type_traits>

constexpr const char* dbranch = "branch=" DEMO_GIT_BRANCH;
constexpr const char* dsha1   = "sha1=" DEMO_GIT_SHA1;
constexpr const char* dvers   = "version=" DEMO_GIT_VERSION;

int main()
{
  std::cout << dbranch << std::endl;
  std::cout << dsha1 << std::endl;
  std::cout << dvers << std::endl;
  return 0;
}
