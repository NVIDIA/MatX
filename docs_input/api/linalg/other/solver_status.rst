.. _solver_status_func:

Solver Status
#############

Report per-matrix solver errors without a host synchronization inside the solver call.

Without a status, CUDA solvers copy each matrix's library ``info`` value to the host and synchronize the stream
inside every call. Most check the values only in debug builds; ``inv`` throws for a singular matrix in every build
type. A solver status instead has the library write
the ``info`` values into a tensor, by default in host-pinned memory that the device writes directly, queued on the
executor's stream like the solver's other outputs. The caller checks the result later, in any build type.

The status does not track when the solver's writes finish. As with any other output of an asynchronous call, you
order the read after the solver call: ``ok(exec)`` and ``check(exec)`` synchronize the executor's stream first, and
``ok()`` and ``check()`` read without waiting, after your own stream or event synchronization. Reading too early
returns an earlier call's values or the tensor's initial contents. A status tensor MatX allocates starts filled with a
value no library writes (``INT_MIN``), so reading it too early after its first call reports a failure, "no solver
result written yet". A tensor you provide holds whatever it held, and zeros there report success.

A CUDA solver call with a status does not synchronize on a device that supports stream-ordered memory pools
(``cudaDevAttrMemoryPoolsSupported``). Without them, MatX's temporary buffers fall back to ``cudaMalloc`` and
``cudaFree``, which synchronize the device. The first call for a shape also creates and caches the solver plan, which
may synchronize.

The table below lists the solvers that accept a status and the number of stages each writes. Pass the status as
the argument right after the input, as in ``chol(A, status)``, or after ``B`` for ``solve(A, B, status)``. The
meaning of a positive ``info`` value depends on the routine; a negative value always means an illegal argument.

.. list-table::
   :header-rows: 1
   :widths: 14 9 40 37

   * - Solver
     - Stages
     - Routines (CUDA / host)
     - Positive ``info``
   * - :ref:`qr_solver <qr_func>`
     - 1
     - cuSolver ``Xgeqrf`` / LAPACK ``geqrf``
     - Not reported
   * - :ref:`qr_econ <qr_func>`
     - 2
     - cuSolver ``Xgeqrf``, then ``orgqr`` / not supported
     - Not reported
   * - :ref:`chol <chol_func>`
     - 1
     - cuSolver ``Xpotrf`` / LAPACK ``potrf``
     - i: the leading minor of order i is not positive definite
   * - :ref:`lu <lu_func>`
     - 1
     - cuSolver ``Xgetrf`` / LAPACK ``getrf``
     - i: U(i,i) is exactly zero, so U is singular
   * - :ref:`eig <eig_func>`
     - 1
     - cuSolver ``Xsyevd`` or its batched form / LAPACK ``syevd``
     - The algorithm did not converge
   * - :ref:`svd <svd_func>`
     - 1
     - cuSolver ``Xgesvd`` or ``gesvdjBatched`` / LAPACK ``gesvd`` or ``gesdd``
     - The algorithm did not converge
   * - :ref:`solve <solve_func>` (dense A)
     - 2
     - cuSolver ``Xgetrf``, then ``Xgetrs``, or cuBLAS ``getrfBatched``, then ``getrsBatched`` / LAPACK ``gesv``
     - Stage 0, i: U(i,i) is exactly zero, so A is singular
   * - :ref:`inv <inv_func>`
     - 2
     - cuBLAS ``getrfBatched``, then ``getriBatched``, ``matinvBatched``, or cuSolver ``Xgetrf``, then ``Xgetrs`` /
       not supported
     - Stage 0, i: U(i,i) is exactly zero, so A is singular

Each solver writes the same number of stages on every path. Where no library call reports a per-matrix result for
stage 1 (``matinvBatched`` for ``inv``, cuBLAS ``getrsBatched`` and LAPACK ``gesv`` for ``solve``), the solver writes
zeros there itself. cuBLAS ``getrsBatched`` reports one host-side value for the whole batch instead, and a nonzero
value throws from the call. A solver with a status keeps running after a matrix fails, as in release builds without
one: for example, ``solve`` still solves a system whose ``getrf`` reported a singular matrix, and ``inv`` no longer
throws for one.

Creating a status
~~~~~~~~~~~~~~~~~

The status tensor has shape ``[stages, batch...]``, where the batch dimensions match the solver input and each stage
holds one library call's ``info`` value per matrix. Element ``[s, b...]`` belongs to stage ``s`` of the matrix at batch
index ``b...``. An unbatched input uses a rank-1 ``[stages]`` tensor. Batch sizes must be positive, or creating the
status throws ``matxInvalidSize``.

A solver call checks only that its input has as many matrices as the status, not that its batch dimensions have the
same shape. ``infos()`` and ``check()`` therefore identify each matrix by its linear batch index, which is the same for
any batch shape with that many matrices.

A status created with a CUDA executor is used only by CUDA solvers, and one created with a host executor only by host
solvers. A mismatched executor and status kind throws ``matxInvalidExecutor`` before the solver queues work. A CUDA status may be used on any stream of its device.
Calls on different streams must be ordered by the caller so they do not write the status at the same time, as for
any output tensor.

A status tensor in device memory is specific to its device, as any device tensor is: use it only with solvers on
streams of that device. MatX checks the device when the status is created, not at each solver call.

A status also serves one solver. Its first solver call that does not throw fixes the number of stages it holds, and a
later call to a solver that writes a different number, such as ``qr_econ`` (two stages) on a status first used by
``qr_solver`` (one), throws before the solver queues work. Every call on a status then writes the same stages, so reads cover the right
values whichever call ran last, including a captured graph replayed after direct calls. Use one status per solver.

* ``make_solver_status(A, exec, params = {})`` allocates a tensor suited to the executor, filled with ``INT_MIN``:

  * With a CUDA executor, host-pinned memory (``MATX_HOST_MEMORY``). The library writes it directly through its
    device-mapped pointer, and the host reads it directly, with no copy. This requires a device that can map host
    memory, which current GPUs on 64-bit platforms can. Otherwise creating the status throws, and you pass a
    device-memory tensor to the other overload instead. The tensor is freed with ``cudaFreeHost``, which waits for all
    work on the device, so destroying the status does too. Create a status once and reuse it across calls rather than
    creating one per call.
  * With a host executor, ordinary host memory.

  ``params`` is a ``SolverStatusParams`` aggregate whose one field, ``stages`` (default 2), is the number of stages
  MatX allocates. Two stages cover every solver that accepts a status:

  .. code-block:: cpp

     auto st = make_solver_status(A, exec);                    // two stages
     auto st1 = make_solver_status(A, exec, {.stages = 1});    // enough for one-stage solvers

* ``make_solver_status(A, info, exec)`` uses a contiguous ``int`` tensor that you provide, whose first dimension is the
  stage count. MatX does not initialize user-provided tensors. With a CUDA executor, the tensor must be in host-pinned memory, which the device
  writes directly through its mapped pointer, or in device memory on the device of the executor's stream. Other memory
  throws when the status is created. With a host executor, the tensor must be host-accessible (pageable, host-pinned,
  or managed memory), which host solvers write directly. Device memory throws when the status is created on host executors.

Where the values can be read follows from the status tensor's memory:

* Host-pinned or other host-accessible memory can be read on the host, with ``ok()``, ``check()``, or ``infos()``,
  and on the device through ``infos()``. Device reads cross the host interconnect.
* Device memory can only be read on the device, through ``infos()``, for example to reduce the values in a later
  expression. ``ok()`` and ``check()`` throw. Use it to keep the values on the device, or to run a solver without the
  forced synchronization when the results are not needed on the host.

Checking a status
~~~~~~~~~~~~~~~~~

* ``ok()`` returns whether every ``info`` value written by the last call is zero, reading without waiting.
* ``check()`` reads without waiting and throws ``matxSolverError`` if any value is nonzero. The message gives the
  number of failures and names the first few, with each one's stage, batch index, and ``info`` value. A negative
  value means an illegal parameter, and ``INT_MIN`` in a status MatX allocated means no result was written yet. The
  batch index is the matrix's linear index over the batch dimensions, in row-major order, which is also its column in
  ``infos()``. Use ``infos()`` for every value. When MatX is built with
  ``MATX_DISABLE_EXCEPTIONS``, a failure logs the message and aborts the process instead, so use ``ok()`` in those
  builds.
* ``ok(exec)`` and ``check(exec)`` synchronize ``exec``'s stream, then read as ``ok()`` and ``check()`` do. That waits
  for all work on the stream, including work queued after the solver call. To wait only for the solver call, record
  an event after it and synchronize on that event instead, then call ``ok()`` or ``check()``. ``exec`` must be of the
  kind the status was created with. A host executor's solver calls finish before returning, so for a host status these
  do not wait.
* ``infos()`` returns the status tensor, viewed as ``[stages, batches]``, in its own memory space. Reading it on the
  host must be ordered after the solver call. On the device, work queued on the solver's stream after the call is
  ordered after the writes. Only the stages written by the last call hold its values.

Library errors reported through the return status of a cuSolver call, rather than through ``info``, throw from the
solver call itself in all build types, with or without a status.

Reads report what the status tensor holds for the stages of the status's solver. The status keeps no other record of
whether a call ran or finished:

* Before any solver call there are no such stages, so ``ok()`` returns true and ``check()`` passes.
* Do not read a status for a solver call that threw. The status may hold partial values, an earlier call's values,
  or its initial contents, which for a tensor of zeros you provide report success. The exception reports the failure, and the next solver call on the
  status that does not throw replaces the values.
* A solver call captured into a CUDA graph writes the status when the graph launches. Order reads after the launch,
  for example by synchronizing the stream the graph was launched on.

Ownership and lifetimes
~~~~~~~~~~~~~~~~~~~~~~~

A ``SolverStatus`` is a handle to shared state, so copies refer to the same status. The status keeps a reference to
the tensor's storage, so the tensor stays valid while the status exists. As with any output tensor, the status must
outlive the solver work that writes it, including every captured graph that is launched later.

Reusing a status for another solver call overwrites the previous results.

A status is not thread-safe. Do not use it, or any copy of it, from two threads at once: a solver call and a read
of the status are not synchronized with each other. It can be handed between threads when the threads synchronize
the handoff themselves.

A default-constructed ``SolverStatus`` is empty. Solvers treat it as no status, and its members throw.

For advanced use, ``state()`` returns the shared ``detail::SolverStatusState``, which holds the status tensor and the
number of stages the status's solver writes. Its fields and methods are implementation details and may change between releases.

Limitations
~~~~~~~~~~~

* Only host executors and ``cudaExecutor`` are supported. ``make_solver_status`` throws ``matxInvalidExecutor`` with
  any other executor, such as ``CUDAJITExecutor`` or ``distributedCUDAExecutor``, and so do ``ok(exec)`` and
  ``check(exec)``, since such an executor never matches a status's kind.
* Solvers and solver projections with a status do not report JIT support, so a JIT expression that uses one throws. The
  fused JIT kernel has no way to write a status.
* QR reports only illegal arguments through ``info``. It does not report rank deficiency or ill-conditioning; inspect
  the diagonal of R for that.
* ``det``, ``pinv``, sparse ``solve``, ``cgsolve``, ``svdpi``, and ``svdbpi`` do not accept a status. ``inv`` with
  ``MAT_INVERSE_ALGO_POSV`` runs only on ``CUDAJITExecutor``, so it does not either.

Examples
~~~~~~~~

A MatX-owned status in host-pinned memory for a CUDA executor:

.. literalinclude:: ../../../../test/00_solver/SolverStatus.cu
   :language: cpp
   :start-after: example-begin solver-status-test-1
   :end-before: example-end solver-status-test-1
   :dedent:

A user-provided host-pinned tensor with ``qr_econ``, read after synchronizing the executor:

.. literalinclude:: ../../../../test/00_solver/SolverStatus.cu
   :language: cpp
   :start-after: example-begin solver-status-test-2
   :end-before: example-end solver-status-test-2
   :dedent:

A user-provided status tensor in device memory, read only on the device:

.. literalinclude:: ../../../../test/00_solver/SolverStatus.cu
   :language: cpp
   :start-after: example-begin solver-status-test-3
   :end-before: example-end solver-status-test-3
   :dedent:

A user event that orders the read after the solver call without waiting for later work on the stream:

.. literalinclude:: ../../../../test/00_solver/SolverStatus.cu
   :language: cpp
   :start-after: example-begin solver-status-test-4
   :end-before: example-end solver-status-test-4
   :dedent:

A one-stage status with ``chol``:

.. literalinclude:: ../../../../test/00_solver/SolverStatus.cu
   :language: cpp
   :start-after: example-begin solver-status-test-5
   :end-before: example-end solver-status-test-5
   :dedent:

A two-stage status with ``solve``:

.. literalinclude:: ../../../../test/00_solver/SolverStatus.cu
   :language: cpp
   :start-after: example-begin solver-status-test-6
   :end-before: example-end solver-status-test-6
   :dedent:

API
~~~

.. doxygenclass:: matx::SolverStatus
   :members:

.. doxygenstruct:: matx::SolverStatusParams
   :members:

.. doxygenfunction:: make_solver_status(const OpA &a, const Executor &exec, const SolverStatusParams &params = {})
.. doxygenfunction:: make_solver_status(const OpA &a, const InfoTensor &info, const Executor &exec)
