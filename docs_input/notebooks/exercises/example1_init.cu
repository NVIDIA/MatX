#include <matx.h>

using namespace matx;

int main() {

  // TODO: Create a 2D tensor of ints called t2data with dimensions 5, 4, and
  // a view of that data using the default view.
  
  auto t2 = ;

  // Initialize the tensor linearly
  t2.SetVals({  {1, 2, 3, 4},
          {5, 6, 7, 8},
          {9, 10, 11, 12},
          {13, 14, 15, 16},
          {17, 18, 19, 20}});

  print(t2);

  return 0;
}
