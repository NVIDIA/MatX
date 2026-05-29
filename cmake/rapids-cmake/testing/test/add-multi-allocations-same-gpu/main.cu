/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <vector>

int main()
{
  // Very we only have a single GPU visible to us
  int nDevices = 0;
  cudaGetDeviceCount(&nDevices);

  if (nDevices == 0) { return 1; }
  std::cout << "Seeing at least a single GPU" << std::endl;
  return 0;
}
