#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({2, 3});
  tensor_t<int, 2> A(shape);
  tensor_t<float, 2> C(shape);

  CSetVals({ {1, 2, 3},
        {4, 5, 6}});
  

  // TODO: Conditionally assign elements of A the value of 1 if the same element in C is > 3, or 0 otherwise
  

  A.Print();

  return 0;
}
