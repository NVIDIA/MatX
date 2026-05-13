.. _qr_func:

qr
##

Perform a QR decomposition. 

.. versionadded:: 0.6.0

.. note::
   The ``mtie`` assignment forms of ``qr``, ``qr_econ``, and ``qr_solver`` use the normal non-JIT solver path.
   CUDA JIT fusion is available through lazy projection members such as ``qr(A).Q``, ``qr(A).R``,
   ``qr_econ(A).Q``, ``qr_econ(A).R``, ``qr_solver(A).Out``, and ``qr_solver(A).Tau`` when
   ``-DMATX_EN_MATHDX=ON`` is enabled and the runtime shape/type is supported by cuSolverDx. Projection JIT
   currently supports ranks 2 through 4 and ``float``, ``double``, ``complex<float>``, and
   ``complex<double>`` inputs. The full ``qr`` projection path is limited to square matrices to preserve its full
   Q/R output contract. ``qr_econ(A).Q`` JIT fusion is currently limited to non-wide matrices where ``m >= n``;
   ``qr_econ(A).R`` and ``qr_solver`` projections support rectangular matrices when cuSolverDx supports the shape.

.. doxygenfunction:: qr

.. note::
   This function is currently not supported with host-based executors (CPU), and performs a full QR 
   decomposition of a tensor `A` with shape `... x m x n`, where `Q` is shaped `... x m x m` and `R`
   is shaped `... x m x n`.

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/QR2.cu
   :language: cpp
   :start-after: example-begin qr-test-1
   :end-before: example-end qr-test-1
   :dedent:

.. doxygenfunction:: qr_econ

.. note::
   This function is currently not supported with host-based executors (CPU). It returns an economic 
   QR decomposition, where `Q/R` are shaped `m x k` and `k x n` respectively, where `k = min(m, n)`. 
   This is useful when `m >> n` to save memory and computation time.

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/QREcon.cu
   :language: cpp
   :start-after: example-begin qr-econ-test-1
   :end-before: example-end qr-econ-test-1
   :dedent:

.. doxygenfunction:: qr_solver

.. note::
   This function does not return `Q` explicitly as it only runs :literal:`geqrf` from LAPACK/cuSolver.
   For full or economic `Q/R`, use :literal:`qr` or :literal:`qr_econ` on a CUDA executor.

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/QR.cu
   :language: cpp
   :start-after: example-begin qr_solver-test-1
   :end-before: example-end qr_solver-test-1
   :dedent:
