/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#define DefineToString(a) define_to_str(a)
#define define_to_str(a)  #a

std::string lock_file_name(int tid)
{
  const static std::string dir = DefineToString(BINARY_DIR);
  return std::string(dir + "/lock." + std::to_string(tid));
}

struct ctest_lock {
  std::string file_name;
  int fd;

  ctest_lock(int test_id)
  {
    this->file_name = lock_file_name(test_id);
    this->fd        = open(file_name.c_str(), O_RDWR | O_CREAT, S_IRGRP | S_IRWXU);

    int activeDevice = 0;
    cudaGetDevice(&activeDevice);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, activeDevice);

    auto deviceIdAsStr = std::to_string(prop.pciBusID);
    write(fd, deviceIdAsStr.c_str(), deviceIdAsStr.size());
    lockf(fd, F_TLOCK, 0);
  }
};

template <typename LockValidator>
bool validate_locks(LockValidator lock_checker, int min_lock_id, int max_lock_id)
{
  using namespace std::chrono_literals;

  // barrier
  // wait for all other tests to lock the respective sentinel file
  std::this_thread::sleep_for(5000ms);

  int valid_count = 0;
  for (int i = min_lock_id; i <= max_lock_id; ++i) {
    auto path       = lock_file_name(i);
    auto fd         = open(path.c_str(), O_RDONLY);
    auto lock_state = lockf(fd, F_TEST, 0);

    bool valid_lock_state = lock_checker(lock_state, i);
    if (valid_lock_state) { ++valid_count; }
  }
  // barrier again so nothing unlocks while other are checking
  // for a lock
  std::this_thread::sleep_for(2000ms);

  return (valid_count == ((max_lock_id + 1) - min_lock_id));
}

void unlock(ctest_lock& lock) { close(lock.fd); }
