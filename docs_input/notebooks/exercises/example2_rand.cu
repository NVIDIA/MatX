#include <matx.h>

using namespace matx;

int main() {

  auto A = make_tensor<float>({4, 4});

  (A = 0).run();

  // TODO: Set tensor A to normally-distributed random numbers


  print(A);
}
