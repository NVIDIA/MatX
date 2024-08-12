.. _allclose_func:

allclose
========

Reduce the closeness of two operators to a single scalar (0D) output. The output
from allclose is an ``int`` value since boolean reductions are not available in hardware


.. doxygenfunction:: allclose(OutType dest, const InType1 &in1, const InType2 &in2, double rtol, double atol, const HostExecutor<MODE> &exec)
.. doxygenfunction:: allclose(OutType dest, const InType1 &in1, const InType2 &in2, double rtol, double atol, cudaExecutor exec = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin allclose-test-1
   :end-before: example-end allclose-test-1
   :dedent:
