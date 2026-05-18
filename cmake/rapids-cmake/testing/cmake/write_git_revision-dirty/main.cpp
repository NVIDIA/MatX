/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <git_version.hpp>
#include <iostream>
#include <type_traits>

int main()
{
#if defined(IS_DIRTY) && !defined(DEMO_GIT_IS_DIRTY)
#error "failed to encode dirty state correctly"
#endif

#if defined(DEMO_GIT_IS_DIRTY) && !defined(IS_DIRTY)
#error "failed to encode dirty state correctly"
#endif

  return 0;
}
