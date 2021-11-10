#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({8, 8});
  tensor_t<float, 2> B(shape);

  // TODO: Set tensor B such that it forms an identity matrix

  B.Print();

  return 0;
}
