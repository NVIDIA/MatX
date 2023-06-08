.. _dct_func:

dct
###

Perform a Direct Cosine Transform (DCT)

.. note::
   These functions are currently not supported with host-based executors (CPU)


.. doxygenfunction:: dct(OutputTensor &out, const InputTensor &in, const cudaStream_t stream = 0)

Examples
~~~~~~~~
.. literalinclude:: ../../../../test/01_radar/dct.cu
  :language: cpp
  :start-after: example-begin dct-1
  :end-before: example-end dct-1
  :dedent:  
