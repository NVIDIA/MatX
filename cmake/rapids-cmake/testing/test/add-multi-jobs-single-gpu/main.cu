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

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "../detail/file_locker.hpp"

int main(int argc, char** argv)
{
  const constexpr int min_lock_id = 0;
  const constexpr int max_lock_id = 5;

  // Lock our sentinel file
  auto my_id = std::stoi(argv[1]);
  auto lock  = ctest_lock(my_id);

  // verify all sentinel files are locked
  auto checker = [my_id](int lock_state, int i) {
    bool valid_lock_state = false;
    if (i == my_id) {
      // we have this file locked
      valid_lock_state = (lock_state == 0);
    } else {
      // some other process has this file locked
      valid_lock_state = (lock_state == -1);
    }
    std::cout << i << " lock_state: " << lock_state << " valid " << valid_lock_state << std::endl;
    return valid_lock_state;
  };
  bool all_locked = validate_locks(checker, min_lock_id, max_lock_id);
  // unlock and return
  unlock(lock);
  return (all_locked) ? 0 : 1;
}
