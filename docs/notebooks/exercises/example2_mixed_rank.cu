#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({2, 3});
  tensor_t<float, 2> A(shape);
  tensor_t<float, 2> B(shape);
  tensor_t<float, 2> C(shape);
  tensor_t<float, 1> V({3});

  C.SetVals({ {1, 2, 3},
        {4, 5, 6}});
  
  V.SetVals({7, 8, 9});

  // TODO: Add vector V to matrix C using rank expansion. Store result in C


  C.Print();

  return 0;
}
