/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

int static_launch_kernelA(int x, int y);
int static_launch_kernelB(int x, int y);

int main(int argc, char**)
{
  auto resultA = static_launch_kernelA(3, argc);
  auto resultB = static_launch_kernelB(3, argc);
  if (resultA != 6 && resultB != 6) { return 1; }
  return 0;
}
