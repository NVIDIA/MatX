#include <matx.h>

using namespace matx;

int main() {
  auto A = make_tensor<float>({2, 3});
  auto B = make_tensor<float>({2, 3});
  auto C = make_tensor<float>({2, 3});
  auto V = make_tensor<float>({3});

  A.SetVals({ {1, 2, 3},
        {4, 5, 6}});
  
  V.SetVals({7, 8, 9});

  // TODO: Add A to itself plus 1, divide the result by 2, and add vector V. 

  
  C.Print();
  
  return 0;
}
