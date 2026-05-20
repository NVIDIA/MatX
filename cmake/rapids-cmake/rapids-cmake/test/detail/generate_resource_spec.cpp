/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct version {
  version() : json_major(1), json_minor(0) {}
  int json_major;
  int json_minor;
};

struct gpu {
  gpu(int i) : id(i), memory(0), slots(0) {};
  gpu(int i, size_t mem) : id(i), memory(mem), slots(100) {}
  int id;
  size_t memory;
  int slots;
};

// A hard-coded JSON printer that generates a ctest resource-specification file:
// https://cmake.org/cmake/help/latest/manual/ctest.1.html#resource-specification-file
void to_json(std::ostream& buffer, version const& v)
{
  buffer << "\"version\": {\"major\": " << v.json_major << ", \"minor\": " << v.json_minor << "}";
}
void to_json(std::ostream& buffer, gpu const& g)
{
  buffer << "\t\t{\"id\": \"" << g.id << "\", \"slots\": " << g.slots << "}";
}

int main(int argc, char** argv)
{
  std::vector<gpu> gpus;
  int nDevices = 0;

#ifdef HAVE_CUDA
  cudaGetDeviceCount(&nDevices);
  if (nDevices == 0) {
    gpus.push_back(gpu(0));
  } else {
    for (int i = 0; i < nDevices; ++i) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      gpus.push_back(gpu(i, prop.totalGlobalMem));
    }
  }
#else
  gpus.push_back(gpu(0));
#endif

  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <filename>\n";
    return 1;
  }

  // GENERATED_RESOURCE_SPEC_FILE requires an absolute path, and CMake does
  // not have a "give me the current working directory" command, so we have to
  // get it from here.
  std::string arg = argv[1];
  if (arg == "--cwd") {
    std::cout << std::filesystem::current_path().string();
    std::cout.flush();
    return 0;
  }

  std::ofstream fout(argv[1]);

  version v;
  fout << "{\n";
  to_json(fout, v);
  fout << ",\n";
  fout << "\"local\": [{\n";
  fout << "\t\"gpus\": [\n";
  for (int i = 0; i < gpus.size(); ++i) {
    to_json(fout, gpus[i]);
    if (i != (gpus.size() - 1)) { fout << ","; }
    fout << "\n";
  }
  fout << "\t]\n";
  fout << "}]\n";
  fout << "}" << std::endl;
  return 0;
}
