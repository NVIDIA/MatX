#include <matx.h>

using namespace matx;

int main() {

  auto t2 = make_tensor<int>({5, 4});

  // Initialize the tensor linearly
  t2.SetVals({{1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
        {17, 18, 19, 20}});

  // TODO: reduce tensor t2 to a 1D tensor by pulling all columns and the
  // second row
  auto t1 = ...;

  print(t1);

  return 0;
}
