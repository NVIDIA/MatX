#include <matx.h>

using namespace matx;

int main() {

  auto t1 = make_tensor<int>({4});

  // Initialize the tensor linearly
  t1.SetVals({1, 2, 3, 4});

  t1.PrefetchDevice(0);

  // TODO: Clone tensor t1 into a 2D tensor by making a new outer dimension 5.
  auto t2c = ...;

  t2c.Print();

  // TODO: After compiling and running the code above, modify the first element
  // in t1 to be 10 on the next line. Uncomment the Print line as well. t1(0) =
  // ...

  // t2c.Print();

  return 0;
}
