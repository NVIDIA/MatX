/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <type_traits>
#include <demo_git_version.hpp>
#include <nested_git_version.h>

constexpr const char* dbranch = DEMO_GIT_BRANCH;
constexpr const char* dsha1 = DEMO_GIT_SHA1;
constexpr const char* dvers = DEMO_GIT_VERSION;
#if defined(DEMO_GIT_IS_DIRTY)
  constexpr const bool disdirty = true;
#else
  constexpr const bool disdirty = false;
#endif

constexpr const char* nbranch = NESTED_GIT_BRANCH;
constexpr const char* nsha1 = NESTED_GIT_SHA1;
constexpr const char* nvers = DEMO_GIT_VERSION;
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
