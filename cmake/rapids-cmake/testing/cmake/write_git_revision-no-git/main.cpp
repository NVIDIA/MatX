/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
