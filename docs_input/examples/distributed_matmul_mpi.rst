MPI Multi-GPU Matrix Multiplication
###################################

``distributed_matmul_mpi`` is a true multi-process, multi-GPU cuBLASMp example.
Each MPI rank selects one node-local CUDA GPU and owns a block-cyclic portion of
each global matrix. The MatX ``matmul`` expression runs collectively across the
entire process grid through cuBLASMp.

The example uses:

* MPI to launch ranks and exchange the NCCL unique ID.
* NCCL as the communicator used by cuBLASMp.
* ``block_cyclic_distribution_t`` to describe matrix ownership.
* ``distributedCUDAExecutor`` to own each rank's cuBLASMp grid and handle.

Prerequisites
=============

cuBLASMp, NCCL, and an MPI implementation with C++ development headers are
required. Each node must expose at least one CUDA GPU per MPI rank on that node.
The example target is available only when ``MATX_EN_CUBLASMP`` is enabled.

Build the example with::

  cmake -S . -B build \
        -DMATX_BUILD_EXAMPLES=ON \
        -DMATX_EN_CUBLASMP=ON \
        -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target distributed_matmul_mpi

Use ``CUBLASMP_HOME`` and ``NCCL_HOME``, or the corresponding CMake package
directory variables, when those libraries are outside the default search path.

Execution
=========

Run two ranks on a node with at least two visible GPUs::

  mpirun -n 2 ./build/examples/distributed_matmul_mpi

The optional arguments are the global ``M``, ``K``, and ``N`` dimensions and
the square block size::

  mpirun -n 4 ./build/examples/distributed_matmul_mpi 4096 2048 3072 128

The matrix dimensions need not be divisible by the block size or process-grid
dimensions. When all node-local GPUs are visible, MPI local-rank discovery
assigns ranks to distinct CUDA device ordinals. Launchers may instead expose one
distinct physical GPU as device zero to each rank. On multi-node launches, each
node repeats that local mapping.

Every rank initializes and verifies only its locally owned fragments. An MPI
all-reduce reports the maximum error across the complete distributed result;
the output matrix is never gathered onto one process.
