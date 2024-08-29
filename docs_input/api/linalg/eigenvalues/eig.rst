.. _eig_func:

eig
###

Perform an eigenvalue decomposition for Hermitian or real symmetric matrices.

.. doxygenfunction:: eig

Enums
~~~~~

The following enums are used for configuring the behavior of Eig operations.

.. doxygenenum:: EigenMode


Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/Eigen.cu
   :language: cpp
   :start-after: example-begin eig-test-1
   :end-before: example-end eig-test-1
   :dedent:
