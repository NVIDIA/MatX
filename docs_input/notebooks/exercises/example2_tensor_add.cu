#include <matx.h>

using namespace matx;

int main() {
  auto A = make_tensor<float>({2, 3});
  auto B = make_tensor<float>({2, 3});
  auto C = make_tensor<float>({2, 3});
  auto V = make_tensor<float>({3});

  A.SetVals({ {1, 2, 3},
        {4, 5, 6}});
  
  B.SetVals({ {7, 8, 9},
        {10, 11, 12}});

  // TODO: Add tensors A and B and store the result in C

  
  print(A);
  printf("\n");
  print(B);
  printf("\n");
  print(C);
}