#include <matx.h>

using namespace matx;

int main() {
  auto C = make_tensor<float>({2, 3});

  C.SetVals({{7, 8, 9}, {10, 11, 12}});

  // TODO: Divide tensor C by 2 and store in C

  print(C);
}