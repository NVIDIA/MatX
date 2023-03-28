#include <matx.h>

using namespace matx;

/**
 * MatX training assignment 1. This training goes through basic tensor
 * operations that were learned in the 01_introduction notebook. Uncomment each
 * verification block as you go to ensure your solutions are correct.
 */

int main() {

  /****************************************************************************************************
   * Create a rank-2 tensor data object of ints with 5 rows and 4 columns called
   *"t2"
   *https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#tensor-views
   ****************************************************************************************************/

  /*** End editing ***/

  /****************************************************************************************************
   * Initialize the t2 view to a 4x5 matrix of increasing values starting at 1
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#tensor-views
   ****************************************************************************************************/
  // t2 = ;
  /*** End editing ***/

  t2.PrefetchDevice(0);

  /****************************************************************************************************
   * Get a slice of the second and third rows with all columns
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#slicing-and-dicing
   *****************************************************************************************************/
  auto t2s = t2;
  /*** End editing ***/

  // Verify slice is correct
  // for (int row = 1; row <= 2; row++) {
  //   for (int col = 0; col < t2.Size(1); col++) {
  //     if (t2(row, col) != t2s(row - 1, col)) {
  //       printf("Mismatch in sliced view! actual = %d, expected = %d\n",
  //       t2s(row - 1, col), t2(row, col)); exit(-1);
  //     }
  //   }
  // }

  // print(t2s);
  // printf("Slice verification passed!\n");

  /****************************************************************************************************
   * Take the slice and clone it into a 3D tensor with new outer dimensions as
   *follows: First dim: keep existing row dimension from t2s Second dim: 2 Third
   *dim: keep existing col dimension from t2s
   https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#increasing-dimensionality
   *****************************************************************************************************/
  auto t3c = t2s;
  /*** End editing ***/

  // Verify clone
  // for (int first = 0; first < t3c.Size(0); first++) {
  //   for (int sec = 0; sec < t3c.Size(1); sec++) {
  //     for (int third = 0; third < t3c.Size(2); third++) {
  //       if (t3c(first, sec, third) != t2s(first, third)) {
  //         printf("Mismatch in cloned view! actual = %d, expected = %d\n",
  //         t3c(first, sec, third), t2s(first, third)); exit(-1);
  //       }
  //     }
  //   }
  // }

  // print(t3c);
  // printf("Clone verification passed!\n");

  /****************************************************************************************************
   * Permute the two outer dimensions of the cloned tensor
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#permuting
   *****************************************************************************************************/
  auto t3p = t3c;
  /*** End editing ***/

  // Verify clone
  // for (int first = 0; first < t3p.Size(0); first++) {
  //   for (int sec = 0; sec < t3p.Size(1); sec++) {
  //     for (int third = 0; third < t3p.Size(2); third++) {
  //       if (t3c(first, sec, third) != t2s(first, third)) {
  //         printf("Mismatch in permuted view! actual = %d, expected = %d\n",
  //         t3c(first, sec, third), t2s(sec, third)); exit(-1);
  //       }
  //     }
  //   }
  // }

  // print(t3p);
  // printf("Permute verification passed!\n");

  return 0;
}
