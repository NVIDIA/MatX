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

  // TODO: Add the value 1 to all elements of A and store the result in B


  print(A);
  printf("\n");
  print(B);
  
  return 0;
}
