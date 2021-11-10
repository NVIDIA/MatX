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
  
  V.SetVals({7, 8, 9});

  // TODO: Add the value 1 to all elements of A and store the result in B


  A.Print();
  printf("\n");
  B.Print();
  
  return 0;
}
