/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
