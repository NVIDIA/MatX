.. _qr_func:

qr
##

Perform a QR decomposition. 

.. doxygenfunction:: qr

.. note::
   This function is currently is not supported with host-based executors (CPU)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/QR2.cu
   :language: cpp
   :start-after: example-begin qr-test-1
   :end-before: example-end qr-test-1
   :dedent:


.. doxygenfunction:: qr_solver

.. note::
   This function does not return `Q` explicitly as it only runs :literal:`geqrf` from LAPACK/cuSolver.
   For full `Q/R`, use :literal:`qr_solver` on a CUDA executor.

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/QR.cu
   :language: cpp
   :start-after: example-begin qr_solver-test-1
   :end-before: example-end qr_solver-test-1
   :dedent: