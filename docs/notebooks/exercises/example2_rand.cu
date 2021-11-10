#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({4,4});
  tensor_t<float, 2> A(shape);

  (A = 0).run();

  // TODO: Set tensor A to normally-distributed random numbers


  A.Print();
}
