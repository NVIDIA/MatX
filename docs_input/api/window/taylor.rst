.. _taylor_func:

taylor
======

Generate a Taylor window.

The Taylor window holds the ``nbar`` sidelobes nearest the mainlobe at a nearly constant level ``sll`` dB below the
mainlobe and tapers the sidelobes beyond them. Parameters are passed in a ``TaylorWindowParams`` struct, so any member
that is not named in a designated initializer keeps its default.

Each sample is evaluated from a cosine series with ``nbar`` terms. The series coefficients are computed once on the host
and stored in a fixed array of ``MaxNbar`` entries (24 by default), so ``nbar`` must not exceed ``MaxNbar``. A taper
without peaks at its edges needs ``nbar >= 2*A^2 + 0.5``, where ``A = acosh(10^(sll/20)) / pi``. With the default
``MaxNbar`` this holds for sidelobe levels up to about 87 dB. For larger values of ``nbar``, raise
``MaxNbar``, e.g. ``taylor<0, float, 32>({N}, {.nbar = 31, .sll = 100})``.

``sll`` must be positive and below about 6165 dB, so that ``10^(sll/20)`` is finite. The parameters are
not otherwise restricted: an ``sll`` below about 15 dB, or an ``nbar`` far above ``2*A^2 + 0.5``, gives a window that
peaks at its edges or has negative samples.

A 2D window for a rectangular aperture, such as SAR range and azimuth weighting, is the product of two 1D windows.

The ``window_functions`` example (``examples/window_functions.cu``) compares the frequency response of the Taylor window
with the other windows MatX provides.

.. versionadded:: head

.. doxygenstruct:: matx::TaylorWindowParams
   :members:

.. doxygenfunction:: matx::taylor(ShapeType &&s, TaylorWindowParams params)
.. doxygenfunction:: matx::taylor(const index_t (&s)[RANK], TaylorWindowParams params)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin taylor-gen-test-1
   :end-before: example-end taylor-gen-test-1
   :dedent:

.. literalinclude:: ../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin taylor-gen-test-2
   :end-before: example-end taylor-gen-test-2
   :dedent:

.. literalinclude:: ../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin taylor-gen-test-4
   :end-before: example-end taylor-gen-test-4
   :dedent:

.. literalinclude:: ../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin taylor-gen-test-3
   :end-before: example-end taylor-gen-test-3
   :dedent:
