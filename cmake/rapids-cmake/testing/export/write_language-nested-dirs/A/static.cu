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

static __global__ void example_cuda_kernel(int& r, int x, int y) { r = x * y + (x * 4 - (y / 2)); }

int static_launch_kernelA(int x, int y)
{
  int r;
  example_cuda_kernel<<<1, 1>>>(r, x, y);
  return r;
}
