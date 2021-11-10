#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({2, 3});
  tensor_t<float, 2> A(shape);
  tensor_t<float, 2> B(shape);
  tensor_t<float, 2> C(shape);
  tensor_t<float, 1> V({3});

  A.SetVals({ {1, 2, 3},
        {4, 5, 6}});
  
  B.SetVals({ {7, 8, 9},
        {10, 11, 12}});

  // TODO: Add tensors A and B and store the result in C

  
  A.Print();
  printf("\n");
  B.Print();
  printf("\n");
  C.Print();
}