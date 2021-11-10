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

  // TODO: Add A to itself plus 1, divide the result by 2, and add vector V. 

  
  C.Print();
  
  return 0;
}
