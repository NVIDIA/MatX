#include "matx.h"

#include <cassert>
#include <cstdio>
#include <math.h>

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
    auto a = matx::make_tensor<float>({10});
    a.SetVals({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    matx::print(a);

    return 0;
}
