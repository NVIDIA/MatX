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


// BUILD NOTES: TO build, include the path to the eigen in cmake with the variable eigen_DIR="Path/To/Eigen"
#ifdef USE_EIGEN
  #include <Eigen/Dense>
#endif

#include <iostream>



int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  int dimX = 3;
  int dimY = 3;

  matx::cudaExecutor exec{};

  ///////////////////////////////////////////////////////////////////////////////
  //////////////               Eigen Test Data Setup               //////////////
  ///////////////////////////////////////////////////////////////////////////////
#ifdef USE_EIGEN
  
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdRowMajor; // define a custom type that is aligned to MatX row-Major.
  
  Eigen::MatrixXd  a(dimX, dimY);
  MatrixXdRowMajor b(dimX, dimY);
  Eigen::RowVectorXd rowVec(dimX);
  Eigen::Matrix<std::complex<double>, 2, 2> complexMatrix;
  Eigen::MatrixXf matrix10x10(10, 10);
#endif
  ///////////////////////////////////////////////////////////////////////////////
  //////////////                MatX Test Data Setup               //////////////
  ///////////////////////////////////////////////////////////////////////////////
  auto aTensor  = matx::make_tensor<double>({dimX,dimY});
  auto bTensor  = matx::make_tensor<double>({dimX,dimY});
  auto tensor1D = matx::make_tensor<double>({dimX});
  auto complexTensor = matx::make_tensor<cuda::std::complex<double>>({2,2});
  auto matTensor10x10 = matx::make_tensor<float>({10,10});


  ///////////////////////////////////////////////////////////////////////////////
  //////////////                   Initialize Data                 //////////////
  ///////////////////////////////////////////////////////////////////////////////
#ifdef USE_EIGEN
  std::cout <<"!!!!!!!!! Using Eigen in Test !!!!!!!!!" << std ::endl;
  // Initialize with random values
  a.setRandom();
  b.setRandom();
  matrix10x10.setRandom();

  rowVec << 1, 2, 3;

  complexMatrix(0, 0) = std::complex<double>(1.0, 2.0);
  complexMatrix(0, 1) = std::complex<double>(2.0, 3.0);
  complexMatrix(1, 0) = std::complex<double>(3.0, 4.0);
  complexMatrix(1, 1) = std::complex<double>(4.0, 5.0);
  
#else
  std::cout <<"!!!!!!!!! Eigen NOT USED in Test !!!!!!!!!" << std ::endl;
  // provide data in tensors if eigen is not used
  (aTensor = matx::random<double>({dimX, dimY}, matx::UNIFORM)).run(exec);
  (bTensor = matx::random<double>({dimX, dimY}, matx::UNIFORM)).run(exec);
  (complexTensor = matx::random<cuda::std::complex<double>>({2, 2}, matx::UNIFORM)).run(exec);
  (matTensor10x10 = matx::random<double>({10, 10}, matx::UNIFORM)).run(exec);

#endif




  ///////////////////////////////////////////////////////////////////////////////
  //////////////              Copy Eigen inputs to MatX            //////////////
  ///////////////////////////////////////////////////////////////////////////////
#ifdef USE_EIGEN
  cudaMemcpy(aTensor.Data(), a.data(), sizeof(double) * dimX * dimY, cudaMemcpyHostToDevice);
  cudaMemcpy(bTensor.Data(), b.data(), sizeof(double) * dimX * dimY, cudaMemcpyHostToDevice);
  cudaMemcpy(complexTensor.Data(), complexMatrix.data(), sizeof(std::complex<double>)*2*2, cudaMemcpyHostToDevice);
  cudaMemcpy(matTensor10x10.Data(), matrix10x10.data(), sizeof(float)*10*10, cudaMemcpyHostToDevice);

  (aTensor = matx::transpose(aTensor)).run(exec);
  // (bTensor = matx::transpose(bTensor)).run(exec); // do not need to transpose because b has the same layout
  (complexTensor = matx::transpose(complexTensor)).run(exec);
  (matTensor10x10 = matx::transpose(matTensor10x10)).run(exec);
#endif

  tensor1D(0) = 1;
  tensor1D(1) = 2;
  tensor1D(2) = 3;
  exec.sync();

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
  // Data Mapping Example
  // 
#ifdef USE_EIGEN  
  std::cout << "=================== Data Map Example ===================" << std::endl;
  double *raw_data;
  // memory could be any type of allocation, but choosing to use managed memory so it's valid on the host and device (this does cost performance)
  cudaMallocManaged((void**)&raw_data, dimX*dimY * sizeof(double)); 
  
  for(int i=0; i < dimX * dimY; i++)
  {
    raw_data[i] = 0.1 + i * 0.1;
  }
  
  // map user memory into Eigen Matrix
  Eigen::Map<MatrixXdRowMajor> mappedMatrix(raw_data, dimX, dimY);
  std::cout << "Eigen Mapped Data :\n" << mappedMatrix << std::endl;

  // map user memory into Eigen Matrix
  auto mappedTensor = matx::make_tensor(raw_data, {dimX, dimY}, false); // create MatX tensor with non-owning user allocated memory
  matx::print(mappedTensor);
  
  // modify the data from each of the references  
  raw_data[4] = 117;
  mappedMatrix(0,1) = 42;
  mappedTensor(2,1) = 87;
  
  // print modified data
  std::cout << "Eigen Mapped Data After Modified :\n" << mappedMatrix << std::endl;
  matx::print(mappedTensor);
#endif 

  //
  // Basic Indexing
  //
  std::cout << "=================== Indexing ===================" << std::endl;
#ifdef USE_EIGEN
  std::cout << "eigen a(1,2) = " <<  a(1,2) << std::endl; 
#endif  

  std::cout << "MatX  a(1,2) = " << aTensor(1,2) << std::endl;


  //
  // Add A and B
  //
  std::cout << "=================== Addition ===================" << std::endl;
#ifdef USE_EIGEN
  Eigen::MatrixXd addResult = a + b; 
  std::cout << "A + B = \n" << addResult << std::endl; 
#endif  

  auto addTensor = aTensor + bTensor;
  matx::print(addTensor);


  //
  // Element-Wise Multiply A and B
  //
  std::cout << "=================== Element-Wise Multiply ===================" << std::endl;
#ifdef USE_EIGEN
  Eigen::MatrixXd elementWise = a.cwiseProduct(b); 
  std::cout << "A .* B = \n" << elementWise << std::endl; 
#endif  

  auto elementWiseTensor = aTensor*bTensor;
  matx::print(elementWiseTensor);


  //
  // Divide A and B
  //
  std::cout << "=================== Element-Wise Division ===================" << std::endl;
#ifdef USE_EIGEN
  Eigen::MatrixXd divResult = a.cwiseQuotient(b); 
  std::cout << "A / B = \n" << divResult << std::endl; 
#endif  

  auto divResultTensor = aTensor / bTensor;
  matx::print(divResultTensor);


  //
  // Slice (Continuous)
  //
  std::cout << "=================== Continuous Slice ===================" << std::endl;
#ifdef USE_EIGEN
  Eigen::Matrix2d aSlice = a.block(0, 0, 2, 2);       
  std::cout << "A Sliced: \n" << aSlice << std::endl; 
#endif  

  auto aSliceTensor = matx::slice<2>(aTensor,{0,0},{2,2});
  matx::print(aSliceTensor);


  //
  // Slice (Strided)
  //
  std::cout << "=================== Strided Slice ===================" << std::endl;
#ifdef USE_EIGEN
  std::cout << "Original matrix10x10:\n" << matrix10x10 << "\n\n"; 
  // Define the starting point, number of elements to select, and strides for both rows and columns
  // int startRow = 0, startCol = 0; // Starting index for rows and columns
  // int rowStride = 3, colStride = 2; // Stride along rows and columns
  // int numRows = 5; // Calculate the number of rows, considering every second element
  // int numCols = 3; // Grab every third item until the 8th item (0, 3, 6)

  // Create a Map with Stride to access the elements
  Eigen::Map<Eigen::MatrixXf, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>                        
  strided(matrix10x10.data() + 0 * matrix10x10.outerStride() + 0,                                                
          5, 3,                                                                                        
          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(3 * matrix10x10.outerStride(), 2));                 

  // Print the strided matrix10x10
  std::cout << "Strided matrix10x10:\n" << strided << "\n";  
#endif  

  auto slicedMat = matx::slice(matTensor10x10, {0,0}, {matx::matxEnd,9}, {2,3});
  matx::print(slicedMat);


  //
  // Clone
  //
  std::cout << "=================== Clone ===================" << std::endl;
#ifdef USE_EIGEN
  // Use the replicate function to create a 5x5 matrix by replicating the 1x5 matrix
  Eigen::MatrixXd mat = rowVec.replicate(3, 1);         
  std::cout << "1D Cloned to 2D \n" << mat << std::endl;
#endif  

  auto cloned3Tensor = matx::clone<2>(tensor1D, {3, matx::matxKeepDim});
  matx::print(cloned3Tensor);


  //
  // Slice Row
  //
  std::cout << "=================== Slice Row ===================" << std::endl;
#ifdef USE_EIGEN
  Eigen::RowVector3d row = a.row(1);               
  std::cout << "Sliced Row \n" << row << std::endl;
#endif  

  auto rowSlice = matx::slice<1>(aTensor, {1, 0}, {matx::matxDropDim, matx::matxEnd});
  matx::print(rowSlice);


  //
  // Permute Rows
  //
  std::cout << "=================== Permute Rows ===================" << std::endl;
#ifdef USE_EIGEN
  std::cout << "Original Matrix:\n" << a << std::endl;  
  // Define a permutation a
  Eigen::PermutationMatrix<3> perm;    
  perm.indices() << 2, 1, 0;  // This permutation swaps the first and third rows
  // Apply the permutation to the rows
  Eigen::Matrix3d permutedMatrix = perm * a;
  std::cout << "Permuted Matrix (Rows):\n" << permutedMatrix << std::endl;
#endif  

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
#ifdef USE_EIGEN
  std::cout << "Original Complex Matrix:\n" << complexMatrix << std::endl; 

  // Extract and output the real part of the complex matrix
  Eigen::Matrix<double, 2, 2> realMatrix = complexMatrix.real();    
  std::cout << "Real Part of Matrix:\n" << realMatrix << std::endl; 
#endif  

  auto realTensor = matx::real(complexTensor);
  matx::print(realTensor);


  //
  // Multiply A and B
  //
  std::cout << "=================== Matrix Multiply ===================" << std::endl;
#ifdef USE_EIGEN
  Eigen::MatrixXd multResult = a * b;                    
  std::cout << "A * B = \n" << multResult << std::endl;  
#endif  

  auto multResultTensor=matmul(aTensor,bTensor);
  matx::print(multResultTensor);


  //
  // inverse Matrix
  //
  std::cout << "=================== Invert Matrix ===================" << std::endl;
#ifdef USE_EIGEN
  // Eigen::MatrixXd inverseMatrix = a.inverse();                                              // current bug where .run(exec) in inverse is ambiguous, so cannot be used with MatX
  // std::cout << "Inverse of the Real Part:\n" << inverseMatrix << std::endl; // current bug where .run(exec) in inverse is ambiguous, so cannot be used with MatX
#endif  

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
