#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({2, 3});
  tensor_t<float, 2> C(shape);

  C.SetVals({{7, 8, 9}, {10, 11, 12}});

  // TODO: Divide tensor C by 2 and store in C

  C.Print();
}