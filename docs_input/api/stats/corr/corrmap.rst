.. _corrmap_func:

corrmap
#######

Per-element windowed correlation map between two operators of identical
shape. Unlike :ref:`corr_func`, which produces a single function-of-lag
output, ``corrmap`` produces an output of the **same shape as the input**,
where each output element is a normalized correlation computed over a small
window of samples around the corresponding input element.

Supports a 1-D window over the last input dimension (signal-processing form)
and a 2-D window over the last two input dimensions (image-processing form).
All other dimensions are treated as independent batches. The window is
cropped at the input boundary: out-of-bounds offsets are skipped and the
normalization uses only the in-bounds samples (means and energies are
computed over the cropped window, not over a zero-padded one).

Typical applications:

- SAR / InSAR coherence and interferogram phase (complex inputs, MAGNITUDE normalization)
- Stereo matching disparity cost (real inputs, ZNCC normalization)
- Template matching, optical flow, change detection (ZNCC normalization)

.. versionadded:: 1.0.0

.. doxygenfunction:: corrmap(const OpA &A, const OpB &B, index_t window)
.. doxygenfunction:: corrmap(const OpA &A, const OpB &B, const cuda::std::array<index_t, 2> &window)

Normalization modes
~~~~~~~~~~~~~~~~~~~

The compile-time template parameter ``Mode`` selects how each window's
samples are combined. The full mathematical definition of each mode is
given in its ``CorrMapNormalize`` enum value.

.. doxygenenum:: matx::CorrMapNormalize

Input type requirements
~~~~~~~~~~~~~~~~~~~~~~~

Inputs must be floating-point or complex floating-point. Supported inner
scalar types are ``float``, ``double``, ``matx::matxFp16`` /
``matx::matxBf16`` (and the underlying ``__half`` / ``__nv_bfloat16``),
and their complex counterparts.

Integer and complex-integer inputs are rejected at compile time by a
``static_assert``. This applies to every mode, including
:cpp:enumerator:`matx::CorrMapNormalize::NONE`: although the NONE-mode
product :math:`A \cdot \bar B` is mathematically well-defined for
integer operands, the normalized modes (MAGNITUDE / ZNCC) require a
final division that would silently truncate to zero for integer
arithmetic, so all integer inputs are rejected uniformly to avoid a
surprising mode-dependent failure mode. Cast integer inputs explicitly
before calling, e.g. ``corrmap(as_float(A), as_float(B), w)``.

Output element type
~~~~~~~~~~~~~~~~~~~

- Complex if either input is complex.
- Real otherwise.
- Precision is the greater of the inputs: e.g. ``float + double`` produces
  ``double``, ``complex<float> + complex<double>`` produces
  ``complex<double>``, and ``complex<float> + double`` also produces
  ``complex<double>``.

Window indexing
~~~~~~~~~~~~~~~

The window uses the floor-center convention: for a
window of length :math:`w` centered at index :math:`n`, the offsets span
:math:`[-\lfloor w/2 \rfloor,\ w - 1 - \lfloor w/2 \rfloor]`. Odd window
sizes are exactly centered; even window sizes introduce a half-element
registration offset.

Examples
~~~~~~~~

2-D MAGNITUDE on complex inputs (SAR/InSAR coherence):

.. literalinclude:: ../../../../test/00_operators/corrmap_test.cu
   :language: cpp
   :start-after: example-begin corrmap-2d-magnitude
   :end-before: example-end corrmap-2d-magnitude
   :dedent:

2-D ZNCC on real inputs (classic normalized cross-correlation):

.. literalinclude:: ../../../../test/00_operators/corrmap_test.cu
   :language: cpp
   :start-after: example-begin corrmap-2d-zncc
   :end-before: example-end corrmap-2d-zncc
   :dedent:

1-D MAGNITUDE on a batched real signal:

.. literalinclude:: ../../../../test/00_operators/corrmap_test.cu
   :language: cpp
   :start-after: example-begin corrmap-1d-magnitude
   :end-before: example-end corrmap-1d-magnitude
   :dedent:
