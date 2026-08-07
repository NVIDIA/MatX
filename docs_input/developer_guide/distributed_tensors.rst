.. _distributed-tensors:

Experimental Distributed Tensors
################################

Status and goals
================

``experimental::distributed_tensor_t`` is a prototype for representing one
logical tensor whose storage is split across CUDA devices and, eventually,
processes. The prototype currently executes single-process, multi-GPU
pointwise work, batch-local ``matmul``, ``chol``, and ``fft``, and gathers a
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

* Only one process is executable today; no optional communication dependency is
  introduced by the core type.
* Pointwise operations require identical layouts. Batch-local transforms
  require aligned batch fragments and fully local operation dimensions.
  Redistribution and scattering will be explicit operations.
* Other distributed BLAS, solver, reduction, DLPack, printing, and global
  element-access paths are not provided yet.
* Materialization enqueues copies on the per-endpoint CUDA streams;
  ``distributedCUDAExecutor::sync`` is the completion boundary.
