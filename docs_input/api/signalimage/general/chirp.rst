.. _chirp_func:

chirp
=====

Creates a real chirp signal (swept-frequency cosine)

.. doxygenfunction:: chirp(index_t num, TimeType last, FreqType f0, TimeType t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)
.. doxygenfunction:: chirp(SpaceOp t, FreqType f0, typename SpaceOp::value_type t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin chirp-gen-test-1
   :end-before: example-end chirp-gen-test-1
   :dedent:


cchirp
======

Creates a complex chirp signal (swept-frequency cosine)

.. doxygenfunction:: cchirp(index_t num, TimeType last, FreqType f0, TimeType t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)
.. doxygenfunction:: cchirp(SpaceOp t, FreqType f0, typename SpaceOp::value_type t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin cchirp-gen-test-1
   :end-before: example-end cchirp-gen-test-1
   :dedent:  


