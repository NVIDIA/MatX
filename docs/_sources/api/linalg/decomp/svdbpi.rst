.. _svdbpi_func:

svdbpi
######

Perform a singular value decomposition (SVD) using the block power iteration method. This method is usually
better than `svd` where the matrices are small and batches are large

.. doxygenfunction:: svdbpi

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/SVD.cu
   :language: cpp
   :start-after: example-begin svdbpi-test-1
   :end-before: example-end svdbpi-test-1
   :dedent:
