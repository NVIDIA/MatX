.. _distributed-tensors:

Experimental Distributed Tensors
################################

Status and goals
================

``experimental::distributed_tensor_t`` is a prototype for representing one
logical tensor whose storage is split across CUDA devices and, eventually,
processes. The prototype executes single-process, multi-GPU pointwise work,
batch-local ``matmul``, ``chol``, and ``fft``, communicator-backed
cuBLASMp and cuSOLVERMp execution selected by block-cyclic inputs, a cuFFT
Xt/Mg transform selected when an FFT dimension spans local GPUs, and gathers a
distributed tensor into a regular tensor. It is not yet a general distributed
MatX operator system.

The logical tensor has exactly one element type and rank. Every local fragment
is a ``tensor_t<T, RANK>``. Fragment extents, strides, process ranks, device
IDs, and allocation sizes may differ, but fragment value types may not.

Wrapping existing storage
=========================

``make_distributed_tensor`` can allocate each local fragment, or it can wrap
existing device pointers. The short forms accept one pointer when the process
owns exactly one fragment, or a vector ordered by increasing distribution
fragment index:

.. code-block:: cpp

  auto local = make_tensor<float>({local_count}, MATX_DEVICE_MEMORY);
  auto distributed =
      make_distributed_tensor(layout, context, local.Data());

  std::vector<float *> pointers{gpu0_pointer, gpu1_pointer};
  auto distributed_many =
      make_distributed_tensor(layout, context, pointers);

For nontrivial mappings, use ``distributed_local_pointer_t`` to bind pointers
to explicit distribution indices. Its optional ``owner`` is a
``std::shared_ptr<void>`` lifetime token for backend-owned allocations:

.. code-block:: cpp

  std::vector<distributed_local_pointer_t<float>> bindings{
      {3, gpu1_pointer, backend_owner},
      {1, gpu0_pointer, backend_owner}};
  auto distributed =
      make_distributed_tensor(layout, context, bindings);

Pointers are otherwise non-owning. The caller must keep their allocations alive
and must supply storage large enough for the corresponding local shape. Only
fragments owned by the current process and one of its local devices may be
bound.

Pointwise execution
===================

``apply`` accepts either ordinary inputs or distributed inputs. Distributed and
ordinary tensor inputs cannot be mixed in one call. Distributed inputs may have
different element types, but they and the output must have the same global
shape, context, and exact fragment layout; the callable's result type must
exactly match the output type. The executor builds an ordinary ``matx::apply``
expression for each group of corresponding local views and launches it on that
fragment's device and stream. It performs no peer copies, collectives, or
implicit redistribution.

For example:

.. code-block:: cpp

  using namespace matx::experimental;

  distributed_context context{{0, 1}};
  distributedCUDAExecutor exec{context};
  auto layout = block_distribution_t<1>::Slab({count}, {{0, 0}, {0, 1}});

  auto a = make_distributed_tensor<float>(layout, context);
  auto b = make_distributed_tensor<float>(layout, context);
  auto output = make_distributed_tensor<float>(layout, context);

  auto add = [] __host__ __device__ (float x, float y) { return x + y; };
  (output = apply(add, a, b)).run(exec);
  exec.sync();

Input fragments are initialized through ``LocalView(i)``. Assignment from a
regular tensor to a distributed tensor is intentionally unavailable because it
would hide a scatter and its distribution policy.

Batch-local transforms
======================

The first transform paths reuse the ordinary ``matmul``, ``chol``, and ``fft``
names. When operands are distributed tensors, the executor constructs the
existing local MatX operation for each aligned fragment and runs it on that
fragment's device stream:

.. code-block:: cpp

  auto layout = block_distribution_t<3>::Slab(
      {batch_count, rows, columns}, {{0, 0}, {0, 1}}, 0);
  auto a = make_distributed_tensor<float>(layout, context);
  auto b = make_distributed_tensor<float>(layout, context);
  auto c = make_distributed_tensor<float>(layout, context);

  (c = matmul(a, b)).run(exec);
  exec.sync();

This is data parallelism, not a distributed transform within a matrix or FFT.
Leading batch dimensions may be partitioned, but the trailing two matrix
dimensions for ``matmul`` and ``chol``, or trailing FFT dimension for ``fft``,
must be fully local. Inputs and output must have aligned batch fragments and
endpoints. The operation performs no communication, and each local call uses
the same regular MatX accelerated path it would use for a non-distributed
tensor.

Collective MP linear algebra
============================

``block_cyclic_distribution_t`` describes the two-dimensional block-cyclic
layout used by cuBLASMp and cuSOLVERMp. The endpoint vector is ordered by the
selected process-grid layout and contains one endpoint per NCCL rank. The first
collective configuration supports one local CUDA device per process:

.. code-block:: cpp

  #include <matx/distributed.h>

  distributed_context context{{local_device}, mpi_rank, mpi_size};
  block_cyclic_distribution_t layout{
      {n, n}, {block_rows, block_columns}, {process_rows, process_columns},
      endpoints};

  // The application bootstraps and owns this NCCL communicator.
  distributedCUDAExecutor exec{
      context, nccl_communicator, process_rows, process_columns};
  auto a = make_distributed_tensor<float>(layout, context);
  auto b = make_distributed_tensor<float>(layout, context);
  auto c = make_distributed_tensor<float>(layout, context);

  (c = matmul(a, b)).run(exec);
  (c = chol(a, SolverFillMode::LOWER)).run(exec);

Both operations are collective: every rank in the process grid must enter them
in the same order. The executor borrows the NCCL communicator, which must
outlive the executor. MatX local views remain ordinary row-major tensors. The
adapters pack them into the column-major local buffers required by the MP
libraries and unpack the result, so the initial path favors correctness and
interoperability over eliminating local layout conversions. The operations
synchronize their local stream before returning because the libraries may
retain host workspace during execution.

Enable these paths with ``MATX_EN_CUBLASMP`` and ``MATX_EN_CUSOLVERMP``. They
support rank-2 ``float``, ``double``, ``complex<float>``, and
``complex<double>`` tensors. Batched MP operations, transpose modes, mixed
precision, redistribution, and more solver factorizations remain future work.

cuFFT multi-GPU
===============

The regular ``fft`` and ``ifft`` functions use the CUDA Toolkit's cuFFT Xt
multi-GPU API when a single rank-1 complex transform is split across two or
more GPUs in one process:

.. code-block:: cpp

  auto layout =
      block_distribution_t<1>::Slab({fft_size}, {{0, 0}, {0, 1}});
  auto input = make_distributed_tensor<complex<float>>(layout, context);
  auto output = make_distributed_tensor<complex<float>>(layout, context);
  (output = fft(input)).run(exec);

The input distribution determines whether the trailing transform dimension is
fully local. Fully local transforms keep using the batch-local path; a
partitioned rank-1 transform selects Xt/Mg. The adapter stages through pinned
host memory because the public cuFFT Xt copy API converts between a contiguous
host array and its opaque multi-GPU descriptor. This also restores natural
output order. It supports ``complex<float>`` and ``complex<double>`` and honors
MatX ``FFTNorm`` modes. cuFFT itself determines which transform sizes and GPU
counts its multi-GPU planner accepts.

cuFFTMp is deliberately not treated as an interchangeable cuFFT Mg backend.
It targets multi-process 2D/3D slab and pencil decompositions and requires
NVSHMEM-compatible allocation, bootstrapping, and descriptor ownership.
Ordinary ``make_distributed_tensor`` allocations do not satisfy that contract.
Configure with ``MATX_EN_CUFFTMP`` to require and link compatible cuFFTMp and
NVSHMEM installations. Transform execution still requires a cuFFTMp-owned
tensor factory and reshape semantics rather than silently copying through a
nominally distributed tensor.

Materialization
===============

A distributed tensor can be assigned to a same-type, same-rank regular tensor:

.. code-block:: cpp

  auto gathered = make_tensor<float>({count});
  (gathered = output).run(exec);
  exec.sync();

The current implementation copies fragments directly into canonical global
row-major order and supports contiguous or strided regular destinations. It
uses unified memory or direct CUDA peer access when source and destination are
on different devices. Host staging is not used; materialization into device
memory fails when direct access is unavailable.

The specified multi-process meaning of this assignment is a collective
all-gather: every participating process receives a complete regular tensor.
The current executor rejects contexts with more than one process until a
collective backend is attached. A future root-only gather will use an explicit
API rather than changing assignment semantics.

Limitations and next steps
==========================

* Pointwise execution and materialization remain single-process. Multi-process
  execution is currently limited to block-cyclic ``matmul`` and ``chol``
  collectives.
* Pointwise operations require identical layouts. Batch-local transforms
  require aligned batch fragments and fully local operation dimensions.
  Redistribution and scattering will be explicit operations.
* Other distributed BLAS, solver, cuFFTMp, reduction, DLPack, printing, and
  global element-access paths are not provided yet.
* Materialization enqueues copies on the per-endpoint CUDA streams;
  ``distributedCUDAExecutor::sync`` is the completion boundary.
