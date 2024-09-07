////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include <matx.h>

using namespace matx;

int main() {

  tensor_t<int, 2> t2({5, 4});

  /****************************************************************************************************
   * Initialize the t2 view to a 4x5 matrix of increasing values starting at 1
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#tensor-views
   ****************************************************************************************************/
  t2.SetVals({{1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
        {17, 18, 19, 20}});
  /*** End editing ***/

  int count = 1;
  for (int row = 0; row < t2.Size(0); row++) {
    for (int col = 0; col < t2.Size(1); col++) {
      if (t2(row, col) != count++) {
        printf("Mismatch in init view! actual = %d, expected = %d\n",
               t2(row, col), count - 1);
        exit(-1);
      }
    }
  }

  print(t2);
  printf("Init verification passed!\n");

  /****************************************************************************************************
   * Get a slice of the second and third rows with all columns
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#slicing-and-dicing
   *****************************************************************************************************/
  auto t2s = slice(t2, {1, 0}, {3, matxEnd}); // Put code here
  /*** End editing ***/

  // Verify slice is correct
  for (int row = 1; row <= 2; row++) {
    for (int col = 0; col < t2.Size(1); col++) {
      if (t2(row, col) != t2s(row - 1, col)) {
        printf("Mismatch in sliced view! actual = %d, expected = %d\n",
               t2s(row - 1, col), t2(row, col));
        exit(-1);
      }
    }
  }

  print(t2s);
  printf("Slice verification passed!\n");

  /****************************************************************************************************
   * Take the slice and clone it into a 3D tensor with new outer dimensions as
   *follows: First dim: keep existing row dimension from t2s Second dim: 2 Third
   *dim: keep existing col dimension from t2s
   https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#increasing-dimensionality
   *****************************************************************************************************/
  auto t3c = t2s.Clone<3>({matxKeepDim, 2, matxKeepDim});
  /*** End editing ***/

  // Verify clone
  for (int first = 0; first < t3c.Size(0); first++) {
    for (int sec = 0; sec < t3c.Size(1); sec++) {
      for (int third = 0; third < t3c.Size(2); third++) {
        if (t3c(first, sec, third) != t2s(first, third)) {
          printf("Mismatch in cloned view! actual = %d, expected = %d\n",
                 t3c(first, sec, third), t2s(first, third));
          exit(-1);
        }
      }
    }
  }

  print(t3c);
  printf("Clone verification passed!\n");

  /****************************************************************************************************
   * Permute the two outer dimensions of the cloned tensor
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#permuting
   *****************************************************************************************************/
  auto t3p = t3c.Permute({1, 0, 2});
  /*** End editing ***/

  // Verify clone
  for (int first = 0; first < t3p.Size(0); first++) {
    for (int sec = 0; sec < t3p.Size(1); sec++) {
      for (int third = 0; third < t3p.Size(2); third++) {
        if (t3c(first, sec, third) != t2s(first, third)) {
          printf("Mismatch in permuted view! actual = %d, expected = %d\n",
                 t3c(first, sec, third), t2s(sec, third));
          exit(-1);
        }
      }
    }
  }

  print(t3p);
  printf("Permute verification passed!\n");

  return 0;
}
