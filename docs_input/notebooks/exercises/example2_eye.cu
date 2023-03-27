#include <matx.h>

using namespace matx;

int main() {
  auto B = make_tensor<float>({8, 8});  

  // TODO: Set tensor B such that it forms an identity matrix

  print(B);

  return 0;
}
