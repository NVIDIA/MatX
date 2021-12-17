#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({2, 3});
  tensor_t<float, 2> A(shape);
  tensor_t<float, 2> B(shape);
  tensor_t<float, 2> C(shape);
  tensor_t<float, 1> V({3});

  // TODO: Initialize the A tensor to contain values increasing from 1 to 6, and
  // V from 7 to 9.
  A = {};
  V = {};

  A.Print();
  printf("\n");
  V.Print();
  
  return 0;
}
