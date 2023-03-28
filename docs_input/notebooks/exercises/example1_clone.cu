#include <matx.h>

using namespace matx;

int main() {

  auto t1 = make_tensor<int>({4});

  // Initialize the tensor linearly
  t1.SetVals({1, 2, 3, 4});

  // TODO: Clone tensor t1 into a 2D tensor by making a new outer dimension 5.
  auto t2c = ...;

  print(t2c);

  // TODO: After compiling and running the code above, modify the first element
  // in t1 to be 10 on the next line. Uncomment the print line as well. t1(0) =
  // ...

  // print(t2c);

  return 0;
}
