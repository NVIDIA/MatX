/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

int static_launch_kernelA(int x, int y);
int static_launch_kernelB(int x, int y);
int static_launch_kernelD(int x, int y);
int cpu_function(int x, int y);

int main(int argc, char**)
{
  auto resultA = static_launch_kernelA(3, argc);
  auto resultB = static_launch_kernelB(3, argc);
  auto resultC = cpu_function(3, argc);
  auto resultD = static_launch_kernelD(3, argc);

  if (resultA != 6) { return 1; }

  if ((resultA != resultB) || (resultA != resultC) || (resultA != resultD)) { return 1; }

  return 0;
}
