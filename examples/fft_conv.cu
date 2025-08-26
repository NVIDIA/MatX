#include "matx.h"
#include <cassert>
#include <cstdio>
#include <cuda/std/ccomplex>

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  {
    using TestType = float;
    int size = 32;
    auto exec = cudaExecutor{};

    tensor_t<TestType, 1> av{{size}};
    tensor_t<TestType, 1> avo{{size}};

    (av = random<TestType>(av.Shape(), UNIFORM)).run(exec);
    print(av);

    (avo = 5.0f * sort(av * 8.f, SORT_DIR_ASC)).run(exec);

    print(avo);
  }

  MATX_EXIT_HANDLER();
}