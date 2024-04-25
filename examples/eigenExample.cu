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


// BUILD NOTES: TO build, include the path to the Eigen header files and uncomment all Eigen commands in this file. 
// #include <Eigen/Dense>

#include <iostream>

using namespace matx;


int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{ 
  int dimX = 3;
  int dimY = 3;
  
  
  // Eigen::MatrixXd a(dimX, dimY);
  // Eigen::MatrixXd b(dimX, dimY);
  // Eigen::MatrixXd addResult(dimX, dimY);
  // Eigen::MatrixXd divResult(dimX, dimY);
  // Eigen::MatrixXd multResult(dimX, dimY);
  // Eigen::MatrixXd elementWise(dimX, dimY);
  
  auto aT = make_tensor<double>({dimX,dimY});
  auto bT = make_tensor<double>({dimX,dimY});
  auto addResultT = make_tensor<double>({dimX,dimY});
  auto divResultT = make_tensor<double>({dimX,dimY});
  auto multResultT = make_tensor<double>({dimX,dimY});
  auto elementWiseT = make_tensor<double>({dimX,dimY});

  // Initialize A and B with random values
  // a.setRandom();
  // b.setRandom();
  
  // Create a 1x5 matrix
  // Eigen::RowVectorXd rowVec(dimX);
  // rowVec << 1, 2, 3;
  
  auto rowT = make_tensor<double>({dimX});
  rowT(0) = 1;
  rowT(1) = 2;
  rowT(2) = 3;
  
  // copy A and B to MatXTensor
  // int dataSize = sizeof(double) * dimX * dimY;
  // cudaMemcpy(aT.Data(), a.data(), dataSize, cudaMemcpyHostToDevice);
  // cudaMemcpy(bT.Data(), b.data(), dataSize, cudaMemcpyHostToDevice);
  
  //transpose to correct storage order
  (aT = transpose(aT)).run();
  (bT = transpose(bT)).run();
  
  // slower alternative of copying per-element
  // for(int curX=0; curX<dimX; curX++)
  // {
  //   for(int curY=0; curY<dimY; curY++)
  //   {
  //     aT(curX,curY) = a(curX,curY); 
  //     bT(curX,curY) = b(curX,curY); 
  //   } 
  // }
  
  
  //
  // Basic Indexing
  //
  std::cout << "=================== Indexing ===================" << std::endl;
  // std::cout << "eigen a(1,2) = " <<  a(1,2) << std::endl;
  std::cout << "MatX  a(1,2) = " << aT(1,2) << std::endl;
  
  
  //
  // Add A and B
  //
  std::cout << "=================== Addition ===================" << std::endl;
  // addResult = a + b;
  // std::cout << "A + B = \n" << addResult << std::endl;
  (addResultT = aT + bT).run();
  matx::print(addResultT);  
   
  
  //
  // Element-Wise Multiply A and B
  // 
  std::cout << "=================== Element-Wise Multiply ===================" << std::endl;
  // elementWise = a.cwiseProduct(b);
  // std::cout << "A .* B = \n" << elementWise << std::endl;
  
  (elementWiseT=aT*bT).run();
  matx::print(elementWiseT);  
  
  //
  // Divide A and B 
  //
  std::cout << "=================== Element-Wise Division ===================" << std::endl;
  // divResult = a.cwiseQuotient(b);
  // std::cout << "A / B = \n" << divResult << std::endl;
  
  (divResultT= aT / bT).run();
  matx::print(divResultT);
  
  //
  // Slice (Continuous)
  //
  std::cout << "=================== Continuous Slice ===================" << std::endl;
  // Eigen::Matrix2d aSlice = a.block(0, 0, 2, 2);
  // std::cout << "A Sliced: \n" << aSlice << std::endl;
  
  auto aSliceT = matx::slice<2>(aT,{0,0},{2,2});
  matx::print(aSliceT);
  
  //
  // Slice (Strided)
  //
  std::cout << "=================== Strided Slice ===================" << std::endl;
  // Eigen::MatrixXf matrix(10, 10); // Create a 10x10 matrix
  // matrix.setRandom(); // Fill it with random numbers

  auto matT = make_tensor<float>({10,10});
  // cudaMemcpy(matT.Data(), matrix.data(), sizeof(float)*10*10, cudaMemcpyHostToDevice);
  (matT = transpose(matT)).run();  
  cudaDeviceSynchronize();
  
  // std::cout << "Original matrix:\n" << matrix << "\n\n";

  // Define the starting point, number of elements to select, and strides for both rows and columns
  // int startRow = 0, startCol = 0; // Starting index for rows and columns
  // int rowStride = 3, colStride = 2; // Stride along rows and columns
  // int numRows = 5; // Calculate the number of rows, considering every second element
  // int numCols = 3; // Grab every third item until the 8th item (0, 3, 6)

  // Create a Map with Stride to access the elements
      // Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> 
      // strided(matrix.data() + 0 * matrix.outerStride() + 0, 
              // 5, 3, 
              // Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(rowStride * matrix.outerStride(), colStride));

  // Print the strided matrix
  // std::cout << "Strided matrix:\n" << strided << "\n";
  
  auto slicedMat = matx::slice(matT, {0,0}, {matxEnd,9}, {2,3});
  matx::print(slicedMat);
  
  //
  // Clone
  //
  std::cout << "=================== Clone ===================" << std::endl;
  // Use the replicate function to create a 5x5 matrix by replicating the 1x5 matrix
  // Eigen::MatrixXd mat = rowVec.replicate(3, 1);
  // std::cout << "1D Cloned to 2D \n" << mat << std::endl;
  
  auto cloned3T = clone<2>(rowT, {3, matxKeepDim});
  matx::print(cloned3T);    
  
  //
  // Slice Row
  //
  std::cout << "=================== Slice Row ===================" << std::endl;
  // Eigen::RowVector3d row = a.row(1);
  // std::cout << "Sliced Row \n" << row << std::endl;

  auto rowSlice = slice<1>(aT, {1, 0}, {matxDropDim, matxEnd});
  matx::print(rowSlice);
  
  //
  // Permute Rows
  //
  std::cout << "=================== Permute Rows ===================" << std::endl;
  // std::cout << "Original Matrix:\n" << a << std::endl;
  // Define a permutation a
  // Eigen::PermutationMatrix<3> perm;
  // perm.indices() << 2, 1, 0;  // This permutation swaps the first and third rows
  // Apply the permutation to the rows
  // Eigen::Matrix3d permutedMatrix = perm * a;
  // std::cout << "Permuted Matrix (Rows):\n" << permutedMatrix << std::endl;

  // Define a permutation a
  auto permVec = make_tensor<int>({dimX});
  permVec(0) = 2;
  permVec(1) = 1;
  permVec(2) = 0;
  auto permMat = make_tensor<double>({dimX,dimY});
  // Apply the permutation to the rows    
  (permMat = remap<0>(aT, permVec)).run();
  matx::print(permMat);
  
  
  //
  // Permutation Dimensions
  //
  std::cout << "=================== Permute Dimension ===================" << std::endl;
  // Unsupported by eigen
  auto permA = permute(aT, {1,0});
  matx::print(permA);
  
  //
  // Get Real Value
  //
  std::cout << "=================== Get Real Values ===================" << std::endl;
  // Define a 2x2 matrix of complex numbers
  // Eigen::Matrix<std::complex<double>, 2, 2> complexMatrix;
  // complexMatrix(0, 0) = std::complex<double>(1.0, 2.0);
  // complexMatrix(0, 1) = std::complex<double>(2.0, 3.0);
  // complexMatrix(1, 0) = std::complex<double>(3.0, 4.0);
  // complexMatrix(1, 1) = std::complex<double>(4.0, 5.0);

  // Output the original complex matrix
  // std::cout << "Original Complex Matrix:\n" << complexMatrix << std::endl;

  // Extract and output the real part of the complex matrix
  // Eigen::Matrix<double, 2, 2> realMatrix = complexMatrix.real();
  // std::cout << "Real Part of Matrix:\n" << realMatrix << std::endl;

  ///\todo TYLER_TODO setup code to have same matrix for MatX

  //
  // Multiply A and B 
  //
  std::cout << "=================== Matrix Multiply ===================" << std::endl;
  // multResult = a * b;
  // std::cout << "A * B = \n" << multResult << std::endl;
  
  (multResultT=matmul(aT,bT)).run();
  matx::print(multResultT);
  
  
  //
  // inverse Matrix
  //
  std::cout << "=================== Invert Matrix ===================" << std::endl;
  // Eigen::MatrixXd inverseMatrix(dimX, dimY);
  // inverseMatrix = a.inverse();
  // std::cout << "Inverse of the Real Part:\n" << inverseMatrix << std::endl;
  
  auto invMat = matx::inv(aT);
  matx::print(invMat);
  
  //
  // 1D FFT
  //
  // Unsupported by eigen
  
  //
  // 1D IFFT
  //  
  // Unsupported by eigen
  
  //
  // 2D FFT
  //
  // Unsupported by eigen
  
  //
  // 2D IFFT
  //  
  // Unsupported by eigen
  
  //
  // Covariance
  //
  // Unsupported by eigen
  
  return 0; 
}
