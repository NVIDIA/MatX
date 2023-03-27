#include <matx.h>

using namespace matx;

int main() {
  auto A = make_tensor<int>({2, 3});
  auto C = make_tensor<float>({2, 3});

  C.SetVals({ {1, 2, 3},
        {4, 5, 6}});
  

  // TODO: Conditionally assign elements of A the value of 1 if the same element in C is > 3, or 0 otherwise
  

  print(A);

  return 0;
}
