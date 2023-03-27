#include <matx.h>

using namespace matx;

int main() {

  auto B = make_tensor<float>({10});  

  // TODO: Set tensor B such that it forms a Hamming window


  print(B);

  return 0;
}
