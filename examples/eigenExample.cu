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
// to build with eigen, search for all instances of "Uncomment_Eigen" and uncomment those lines
// #include <Eigen/Dense> // Uncomment_Eigen

#include <iostream>



int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{ 
  int dimX = 3;
  int dimY = 3;
  
  
  ///////////////////////////////////////////////////////////////////////////////
  //////////////               Eigen Test Data Setup               //////////////
  ///////////////////////////////////////////////////////////////////////////////
  
  // Eigen::MatrixXd a(dimX, dimY);            // Uncomment_Eigen
  // Eigen::MatrixXd b(dimX, dimY);            // Uncomment_Eigen
  // Eigen::MatrixXf matrix10x10(10, 10);      // Uncomment_Eigen
  // Eigen::RowVectorXd rowVec(dimX);          // Uncomment_Eigen
  // Eigen::Matrix<std::complex<double>, 2, 2> complexMatrix;  // Uncomment_Eigen

  ///////////////////////////////////////////////////////////////////////////////
  //////////////                MatX Test Data Setup               //////////////
  ///////////////////////////////////////////////////////////////////////////////
  auto aTensor  = matx::make_tensor<double>({dimX,dimY});
  auto bTensor  = matx::make_tensor<double>({dimX,dimY});
  auto tensor1D = matx::make_tensor<double>({dimX});
  auto matTensor10x10 = matx::make_tensor<float>({10,10});
  auto complexTensor = matx::make_tensor<cuda::std::complex<double>>({2,2});


  ///////////////////////////////////////////////////////////////////////////////
  //////////////                   Initialize Data                 //////////////
  ///////////////////////////////////////////////////////////////////////////////
  
  // provide data in tensors if eigen is commented out. not needed if eigen is setting data
  (aTensor = matx::random<double>({dimX, dimY}, matx::UNIFORM)).run(); 
  (bTensor = matx::random<double>({dimX, dimY}, matx::UNIFORM)).run(); 
  (matTensor10x10 = matx::random<double>({10, 10}, matx::UNIFORM)).run(); 
  (complexTensor = matx::random<cuda::std::complex<double>>({2, 2}, matx::UNIFORM)).run(); 
    
  // Initialize with random values
  // a.setRandom(); // Uncomment_Eigen
  // b.setRandom(); // Uncomment_Eigen
  // matrix10x10.setRandom(); // Uncomment_Eigen 
  
  // rowVec << 1, 2, 3; // Uncomment_Eigen
  
  // complexMatrix(0, 0) = std::complex<double>(1.0, 2.0); // Uncomment_Eigen
  // complexMatrix(0, 1) = std::complex<double>(2.0, 3.0); // Uncomment_Eigen
  // complexMatrix(1, 0) = std::complex<double>(3.0, 4.0); // Uncomment_Eigen
  // complexMatrix(1, 1) = std::complex<double>(4.0, 5.0); // Uncomment_Eigen

  


  ///////////////////////////////////////////////////////////////////////////////
  //////////////              Copy Eigen inputs to MatX            //////////////
  ///////////////////////////////////////////////////////////////////////////////
  

  
  // cudaMemcpy(aTensor.Data(), a.data(), sizeof(double) * dimX * dimY, cudaMemcpyHostToDevice); // Uncomment_Eigen
  // cudaMemcpy(bTensor.Data(), b.data(), sizeof(double) * dimX * dimY, cudaMemcpyHostToDevice); // Uncomment_Eigen
  // cudaMemcpy(matTensor10x10.Data(), matrix10x10.data(), sizeof(float)*10*10, cudaMemcpyHostToDevice);   // Uncomment_Eigen
  // cudaMemcpy(complexTensor.Data(), complexMatrix.data(), sizeof(std::complex<double>)*2*2, cudaMemcpyHostToDevice);   // Uncomment_Eigen
  
  
  // (aTensor = matx::transpose(aTensor)).run();      // Uncomment_Eigen
  // (bTensor = matx::transpose(bTensor)).run();      // Uncomment_Eigen
  // (matTensor10x10 = matx::transpose(matTensor10x10)).run();  // Uncomment_Eigen
  // (complexTensor = matx::transpose(complexTensor)).run();  // Uncomment_Eigen
  
  tensor1D(0) = 1;
  tensor1D(1) = 2;
  tensor1D(2) = 3;
  cudaDeviceSynchronize();  
  
  // slower alternative of copying per-element
  // for(int curX=0; curX<dimX; curX++)
  // {
  //   for(int curY=0; curY<dimY; curY++)
  //   {
  //     aTensor(curX,curY) = a(curX,curY); 
  //     bTensor(curX,curY) = b(curX,curY); 
  //   } 
  // }
  
  
  ///////////////////////////////////////////////////////////////////////////////
  //////////////                 Operation Examples                //////////////
  ///////////////////////////////////////////////////////////////////////////////
  
  //
  // Basic Indexing
  //
  std::cout << "=================== Indexing ===================" << std::endl;
  // std::cout << "eigen a(1,2) = " <<  a(1,2) << std::endl;  // Uncomment_Eigen
  
  std::cout << "MatX  a(1,2) = " << aTensor(1,2) << std::endl;
  
  
  //
  // Add A and B
  //
  std::cout << "=================== Addition ===================" << std::endl;
  // Eigen::MatrixXd addResult = a + b; // Uncomment_Eigen
  // std::cout << "A + B = \n" << addResult << std::endl; // Uncomment_Eigen
  
  auto addTensor = aTensor + bTensor;
  matx::print(addTensor);  
   
  
  //
  // Element-Wise Multiply A and B
  // 
  std::cout << "=================== Element-Wise Multiply ===================" << std::endl;
  // Eigen::MatrixXd elementWise = a.cwiseProduct(b); // Uncomment_Eigen
  // std::cout << "A .* B = \n" << elementWise << std::endl; // Uncomment_Eigen
  
  auto elementWiseTensor = aTensor*bTensor;
  matx::print(elementWiseTensor);  
  
  
  //
  // Divide A and B 
  //
  std::cout << "=================== Element-Wise Division ===================" << std::endl;
  // Eigen::MatrixXd divResult = a.cwiseQuotient(b); // Uncomment_Eigen
  // std::cout << "A / B = \n" << divResult << std::endl; // Uncomment_Eigen
  
  auto divResultTensor = aTensor / bTensor;
  matx::print(divResultTensor);
  
  
  //
  // Slice (Continuous)
  //
  std::cout << "=================== Continuous Slice ===================" << std::endl;
  // Eigen::Matrix2d aSlice = a.block(0, 0, 2, 2);       // Uncomment_Eigen 
  // std::cout << "A Sliced: \n" << aSlice << std::endl; // Uncomment_Eigen 
    
  auto aSliceTensor = matx::slice<2>(aTensor,{0,0},{2,2});
  matx::print(aSliceTensor);
  
  
  //
  // Slice (Strided)
  //
  std::cout << "=================== Strided Slice ===================" << std::endl;  
  // std::cout << "Original matrix10x10:\n" << matrix10x10 << "\n\n"; // Uncomment_Eigen
  // Define the starting point, number of elements to select, and strides for both rows and columns
  // int startRow = 0, startCol = 0; // Starting index for rows and columns             
  // int rowStride = 3, colStride = 2; // Stride along rows and columns                 
  // int numRows = 5; // Calculate the number of rows, considering every second element 
  // int numCols = 3; // Grab every third item until the 8th item (0, 3, 6)             
  
  // Create a Map with Stride to access the elements
  // Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>                        // Uncomment_Eigen   
  // strided(matrix10x10.data() + 0 * matrix10x10.outerStride() + 0,                                                // Uncomment_Eigen     
          // 5, 3,                                                                                        // Uncomment_Eigen  
          // Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(3 * matrix10x10.outerStride(), 2));                 // Uncomment_Eigen 

  // Print the strided matrix10x10
  // std::cout << "Strided matrix10x10:\n" << strided << "\n";  // Uncomment_Eigen
  
  auto slicedMat = matx::slice(matTensor10x10, {0,0}, {matx::matxEnd,9}, {2,3});
  matx::print(slicedMat);
  
  
  //
  // Clone
  //
  std::cout << "=================== Clone ===================" << std::endl;
  // Use the replicate function to create a 5x5 matrix by replicating the 1x5 matrix
  // Eigen::MatrixXd mat = rowVec.replicate(3, 1);           // Uncomment_Eigen
  // std::cout << "1D Cloned to 2D \n" << mat << std::endl;  // Uncomment_Eigen
  
  auto cloned3Tensor = matx::clone<2>(tensor1D, {3, matx::matxKeepDim});
  matx::print(cloned3Tensor);    
  
  
  //
  // Slice Row
  //
  std::cout << "=================== Slice Row ===================" << std::endl;
  // Eigen::RowVector3d row = a.row(1);                // Uncomment_Eigen
  // std::cout << "Sliced Row \n" << row << std::endl; // Uncomment_Eigen

  auto rowSlice = matx::slice<1>(aTensor, {1, 0}, {matx::matxDropDim, matx::matxEnd});
  matx::print(rowSlice);
  
  
  //
  // Permute Rows
  //
  std::cout << "=================== Permute Rows ===================" << std::endl;
  // std::cout << "Original Matrix:\n" << a << std::endl;   // Uncomment_Eigen
  // Define a permutation a
  // Eigen::PermutationMatrix<3> perm;     // Uncomment_Eigen
  // perm.indices() << 2, 1, 0;  // This permutation swaps the first and third rows // Uncomment_Eigen
  // Apply the permutation to the rows  
  // Eigen::Matrix3d permutedMatrix = perm * a; // Uncomment_Eigen
  // std::cout << "Permuted Matrix (Rows):\n" << permutedMatrix << std::endl; // Uncomment_Eigen

  // Define a permutation a
  auto permVec = matx::make_tensor<int>({dimX});
  permVec(0) = 2;
  permVec(1) = 1;
  permVec(2) = 0;
  // Apply the permutation to the rows    
  auto permTensor = matx::remap<0>(aTensor, permVec);
  matx::print(permTensor);
  
  
  //
  // Permutation Dimensions
  //
  std::cout << "=================== Permute Dimension ===================" << std::endl;
  // Unsupported by eigen
  auto permA = permute(aTensor, {1,0});
  matx::print(permA);
  
  //
  // Get Real Value
  //
  std::cout << "=================== Get Real Values ===================" << std::endl;
  // std::cout << "Original Complex Matrix:\n" << complexMatrix << std::endl; // Uncomment_Eigen 

  // Extract and output the real part of the complex matrix
  // Eigen::Matrix<double, 2, 2> realMatrix = complexMatrix.real();    // Uncomment_Eigen
  // std::cout << "Real Part of Matrix:\n" << realMatrix << std::endl; // Uncomment_Eigen

  auto realTensor = matx::real(complexTensor);
  matx::print(realTensor);
  

  //
  // Multiply A and B 
  //
  std::cout << "=================== Matrix Multiply ===================" << std::endl;
  // Eigen::MatrixXd multResult = a * b;                    // Uncomment_Eigen
  // std::cout << "A * B = \n" << multResult << std::endl;  // Uncomment_Eigen
  
  auto multResultTensor=matmul(aTensor,bTensor);
  matx::print(multResultTensor);
  
  
  //
  // inverse Matrix
  //
  std::cout << "=================== Invert Matrix ===================" << std::endl;
  // Eigen::MatrixXd inverseMatrix(dimX, dimY);                                // current bug where .run() in inverse is ambiguous, so cannot be used with MatX 
  // inverseMatrix = a.inverse();                                              // current bug where .run() in inverse is ambiguous, so cannot be used with MatX
  // std::cout << "Inverse of the Real Part:\n" << inverseMatrix << std::endl; // current bug where .run() in inverse is ambiguous, so cannot be used with MatX
  
  auto invTensor = matx::inv(aTensor);
  matx::print(invTensor);
  
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
