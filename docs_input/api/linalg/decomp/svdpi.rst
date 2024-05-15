.. _svdpi_func:

svdpi
#####

Perform a singular value decomposition (SVD) using the power iteration method. This method is usually
better than `svd` where the matrices are small and batches are large

.. doxygenfunction:: svdpi

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/SVD.cu
   :language: cpp
   :start-after: example-begin svdpi-test-1
   :end-before: example-end svdpi-test-1
   :dedent:
