#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<1> shape({10});
  tensor_t<float, 1> B(shape);

  // TODO: Set tensor B such that it forms a Hamming window


  B.Print();

  return 0;
}
