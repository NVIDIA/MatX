.. _sar_bulk_mocomp_func:

sar_bulk_mocomp
################

Apply bulk motion compensation to complex SAR phase-history data in the FX
domain. sar_bulk_mocomp is currently in the matx::experimental namespace
because its API is subject to change.

The operator is lazy, allocation-free, and composable with other MatX
expressions. The final two FX dimensions are interpreted as pulse and frequency
sample, and any leading dimensions are batches:

.. code-block:: text

   fx:           [batch..., pulses, samples]
   range_offset: [batch..., pulses]
   result:       [batch..., pulses, samples]

For sample index :math:`n`, the frequency is

.. math::

   f_n = f_{\mathrm{ref}} +
         \left(n - \left\lfloor N/2 \right\rfloor\right)\Delta f,

where :math:`f_{\mathrm{ref}}` is
SarBulkMocompParams::phase_reference_frequency and :math:`\Delta f` is
SarBulkMocompParams::sample_frequency_spacing. For differential reference
range :math:`\Delta R_p`, the applied correction is

.. math::

   \exp\left(-j\,\mathrm{sgn}\,\frac{4\pi}{c}\,f_n\,\Delta R_p\right).

The one-range overload assumes an initial reference range of zero. The
two-range overload changes reference from initial_reference_range to
target_reference_range and uses
:math:`\Delta R_p = R_{\mathrm{target},p} - R_{\mathrm{initial},p}`.

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/sar_bulk_mocomp_test.cu
   :language: cpp
   :start-after: example-begin sar-bulk-mocomp-1
   :end-before: example-end sar-bulk-mocomp-1
   :dedent:

Phase arithmetic follows the range-offset value type. Float ranges use
single-precision frequency, phase, and trigonometric arithmetic; double ranges
use double precision for phase construction and argument reduction. With CUDA,
when double ranges are combined with ``cuda::std::complex<float>`` FX data, the
bounded, reduced angle is converted to float for a fast single-precision
``sincospif`` evaluation. Double-complex FX data retains the full double-precision
trigonometric path. The parameter frequencies are stored as double and narrowed
for the single-precision range path. Other range types, including fltflt and
half precision, are not currently supported. FX values must be
cuda::std::complex<float> or cuda::std::complex<double>.

Host, CUDA, and CUDAJIT executors are supported. Distributed expressions and
dynamic-rank inputs are not currently supported.

.. versionadded:: head

.. doxygenfunction:: sar_bulk_mocomp(const FxOp &fx, const TargetRangeOp &target_reference_range, const SarBulkMocompParams &params)
.. doxygenfunction:: sar_bulk_mocomp(const FxOp &fx, const InitialRangeOp &initial_reference_range, const TargetRangeOp &target_reference_range, const SarBulkMocompParams &params)
.. doxygenstruct:: matx::experimental::SarBulkMocompParams
   :members:
