#include <matx.h>

using namespace matx;

int main() {

  auto A = make_tensor<float>({2, 3});
  auto B = make_tensor<float>({2, 3});
  auto C = make_tensor<float>({2, 3});
  auto V = make_tensor<float>({3});

  C.SetVals({ {1, 2, 3},
        {4, 5, 6}});
  
  V.SetVals({7, 8, 9});

  // TODO: Add vector V to matrix C using rank expansion. Store result in C


  print(C);

  return 0;
}
