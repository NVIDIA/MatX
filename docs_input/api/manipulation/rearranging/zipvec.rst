.. _zipvec_func:

zipvec
======

Zips together multiple operators to yield a vectorized operator, like float2.
zipvec can be used, for example, to create a 2D/3D set of coordinates stored as float2/float3
using operators that represent the independent coordinate components.

.. doxygenfunction:: zipvec(const Ts&... ts)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/zipvec_test.cu
   :language: cpp
   :start-after: example-begin zipvec-test-1
   :end-before: example-end zipvec-test-1
   :dedent:

