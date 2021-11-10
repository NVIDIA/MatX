#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({2, 3});
  tensor_t<float, 2> A(shape);
  tensor_t<float, 2> B(shape);
  tensor_t<float, 1> V({3});

  /****************************************************************************************************
   * Initialize tensor A with increasing values from 0.5 to 3.0 in steps of 0.5,
   *and tensor V from -1 to -3 in steps of -1.
   ****************************************************************************************************/
  A.SetVals({{0.5, 1, 1.5}, {2.0, 2.5, 3.0}});

  V.SetVals({-1, -2, -3});
  /*** End editing ***/

  // Verify init is correct
  float step = 0.5;
  for (int row = 0; row < A.Size(0); row++) {
    for (int col = 0; col < A.Size(1); col++) {
      if (A(row, col) != step) {
        printf("Mismatch in A init view! actual = %f, expected = %f\n",
               A(row, col), step);
        exit(-1);
      }
      step += 0.5;
    }
  }

  for (int col = 0; col < V.Size(0); col++) {
    if (V(col) != (-1 + col * -1)) {
      printf("Mismatch in A init view! actual = %f, expected = %f\n", V(col),
             (float)(-1 + col * -1));
      exit(-1);
    }
  }

  A.Print();
  V.Print();
  printf("Init verification passed!\n");

  /****************************************************************************************************
   * Add 5.0 to all elements of A and store the results back in A
   ****************************************************************************************************/
  (A = A + 5.0).run();
  /*** End editing ***/

  cudaStreamSynchronize(0);

  step = 0.5;
  for (int row = 0; row < A.Size(0); row++) {
    for (int col = 0; col < A.Size(1); col++) {
      if (A(row, col) != (5.0 + step)) {
        printf("Mismatch in A sum view! actual = %f, expected = %f\n",
               A(row, col), 5.0 + step);
        exit(-1);
      }
      step += 0.5;
    }
  }

  A.Print();
  printf("Sum verification passed!\n");

  /****************************************************************************************************
   * Clone V to match the dimensions of A, and subtract V from A. The results
   * should be stored in A
   *
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#increasing-dimensionality
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/api/tensorview.html#_CPPv4I0_iEN4matx12tensor_tE
   *
   ****************************************************************************************************/
  auto tvs = V.Clone<2>({A.Size(0), matxKeepDim});
  (A = A - tvs).run();
  /*** End editing ***/

  cudaStreamSynchronize(0);

  step = 0.5;
  for (int row = 0; row < A.Size(0); row++) {
    for (int col = 0; col < A.Size(1); col++) {
      if (A(row, col) != (5.0 + step - tvs(row, col))) {
        printf("Mismatch in A sub view! actual = %f, expected = %f\n",
               A(row, col), 5.0 + step - tvs(row, col));
        exit(-1);
      }
      step += 0.5;
    }
  }

  A.Print();
  tvs.Print();
  printf("Clone verification passed!\n");

  /****************************************************************************************************
   * Raise the matrix A to the power of 2 and multiply the output by two. Next,
   * subtract the vector V from each row. Store the result in tensor B.
   *
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/api/tensorops.html#_CPPv4N4matx3powE2Op2Op
   ****************************************************************************************************/
  (B = (pow(A, 2) * 2) - V).run();
  /*** End editing ***/

  cudaStreamSynchronize(0);

  for (int row = 0; row < B.Size(0); row++) {
    for (int col = 0; col < B.Size(1); col++) {
      if (B(row, col) != powf(A(row, col), 2) * 2 - V(col)) {
        printf("Mismatch in B init view! actual = %f, expected = %f\n",
               B(row, col), powf(A(row, col), 2) * 2 - V(col));
        exit(-1);
      }
    }
  }

  B.Print();
  printf("Mixed verification passed!\n");

  return 0;
}
