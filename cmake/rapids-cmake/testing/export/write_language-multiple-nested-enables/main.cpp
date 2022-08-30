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

#include <iostream>

int static_launch_kernelA(int x, int y);
int static_launch_kernelB(int x, int y);
int static_launch_kernelD(int x, int y);
int cpu_function(int x, int y);

int main(int argc, char **) {

  auto resultA = static_launch_kernelA(3, argc);
  auto resultB = static_launch_kernelB(3, argc);
  auto resultC = cpu_function(3, argc);
  auto resultD = static_launch_kernelD(3, argc);

  if (resultA != 6) {
    return 1;
  }

  if ((resultA != resultB) || (resultA != resultC) || (resultA != resultD)) {
    return 1;
  }

  return 0;
}
