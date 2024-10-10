.. _ambgfun_func:

ambgfun
#######

Ambiguity function

.. doxygenfunction:: ambgfun(const XTensor &x, const YTensor &y, double fs, AMBGFunCutType_t cut, float cut_val = 0.0)
.. doxygenfunction:: ambgfun(const XTensor &x, double fs, AMBGFunCutType_t cut, float cut_val = 0.0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/01_radar/ambgfun.cu
   :language: cpp
   :start-after: example-begin ambgfun-test-1
   :end-before: example-end ambgfun-test-1
   :dedent:

.. literalinclude:: ../../../../test/01_radar/ambgfun.cu
   :language: cpp
   :start-after: example-begin ambgfun-test-2
   :end-before: example-end ambgfun-test-2
   :dedent:
