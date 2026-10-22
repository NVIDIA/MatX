Multi-Process Cholesky Decomposition
####################################

``distributed_cholesky_mpi`` demonstrates a collective Cholesky decomposition
across multiple processes and GPUs. Each MPI rank selects one node-local GPU,
owns one block-cyclic portion of the global matrix, and enters the MatX
``chol`` expression collectively. MatX dispatches the operation to cuSOLVERMp
using an NCCL communicator exchanged through MPI.

Unlike the single-node multi-GPU FFT example, this program can span nodes.
MPI controls process placement and the total number of GPUs:

* ``mpirun -n 2`` uses two ranks and two GPUs.
* ``mpirun -n 8`` uses eight ranks and eight GPUs.
* A multi-node host mapping can place those ranks across several machines.

Prerequisites
=============

cuSOLVERMp, NCCL, and an MPI implementation with C++ development headers are
required. The example target is available only when ``MATX_EN_CUSOLVERMP`` is
enabled.

Build the example with::

  cmake -S . -B build \
        -DMATX_BUILD_EXAMPLES=ON \
        -DMATX_EN_CUSOLVERMP=ON \
        -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target distributed_cholesky_mpi

Use ``CUSOLVERMP_HOME`` and ``NCCL_HOME``, or the corresponding CMake package
directory variables, when those libraries are outside the default search path.

Execution
=========

Run four processes on four GPUs in one node::

  mpirun -n 4 ./build/examples/distributed_cholesky_mpi

Run eight processes split across two four-GPU nodes using the host syntax
provided by the local MPI implementation, for example::

  mpirun -n 8 --host node0:4,node1:4 \
    ./build/examples/distributed_cholesky_mpi

The optional arguments select the square matrix size and square block size::

  mpirun -n 4 ./build/examples/distributed_cholesky_mpi 4096 256

When all node-local GPUs are visible, MPI local-rank discovery assigns ranks to
distinct device ordinals. Launchers may instead expose one distinct physical
GPU as device zero to each rank.

The input is a deterministic positive-definite diagonal matrix. Every rank
verifies only its local Cholesky-factor fragment, and an MPI all-reduce reports
the maximum error without gathering the matrix onto one process.
