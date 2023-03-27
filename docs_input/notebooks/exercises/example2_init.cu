#include <matx.h>

using namespace matx;

int main() {
  auto A = make_tensor<float>({2, 3});
  auto B = make_tensor<float>({2, 3});
  auto C = make_tensor<float>({2, 3});
  auto V = make_tensor<float>({3});

  // TODO: Initialize the A tensor to contain values increasing from 1 to 6, and
  // V from 7 to 9.
  A = {};
  V = {};

  print(A);
  printf("\n");
  print(V);
  
  return 0;
}
