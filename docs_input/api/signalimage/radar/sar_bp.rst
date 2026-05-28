.. _sar_bp_func:

sar_bp
#######

Synthetic aperture radar backprojection. The sar_bp operator is currently in the experimental
namespace as its API is subject to change.

.. versionadded:: head

.. doxygenfunction:: sar_bp(const ImageType &initial_image, const RangeProfilesType &range_profiles, const PlatPosType &platform_positions, const VoxLocType &voxel_locations, const RangeToMcpType &range_to_mcp, const SarBpParams &params)
.. doxygenenum:: matx::SarBpComputeType
.. doxygenenum:: matx::SarBpFeature
.. doxygenstruct:: matx::PropSarBpTaylorFastAddThirdOrder
.. doxygenstruct:: matx::SarBpParams
   :members:

TaylorFast Range Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SarBpComputeType::TaylorFast`` approximates the range from each pulse to each
pixel by expanding the range about a reference pixel for the current thread
block. The goal is to replace most per-pixel range work with arithmetic in a
local coordinate system, while computing the expensive reference quantities once
per pulse and per thread block. By default, ``TaylorFast`` keeps terms through
second order in the local pixel offset. Users can add the third-order term with
the ``PropSarBpTaylorFastAddThirdOrder`` property. The third-order term is most
useful for short stand-off ranges where the second-order approximation may not
provide sufficient accuracy.

The derivation of the Taylor approximation follows.
For pulse :math:`p`, let :math:`a_p` be the platform position and let :math:`m_p` be the
range to the motion compensation point. For image pixel position :math:`x`, the
range quantity used by backprojection is

.. math::

   \Delta R_p(x) = \|x - a_p\| - m_p.

For a thread block, choose a reference pixel :math:`x_0` near the center of the
block. Define

.. math::

   r_0 = x_0 - a_p, \qquad R_0 = \|r_0\|, \qquad u = \frac{r_0}{R_0},

where :math:`u` is the unit vector pointing from the platform position to the reference pixel. Hereafter, we generally drop
the :math:`p` subscript for brevity. For any pixel in the same thread block, write its local offset as

.. math::

   d = x - x_0.

Then the exact range to the pixel is

.. math::

   R(d) = \|r_0 + d\| = \|(x_0 - a_p) + (x - x_0)\| = \|x - a_p\|.

Decompose :math:`d` into the component along :math:`u` and the component perpendicular
to :math:`u`:

.. math::

   s = u \cdot d.

Here, :math:`s` is the signed scalar projection of :math:`d` along :math:`u` and :math:`u` is the
along-range direction. The perpendicular, or cross-range, vector
is then :math:`d - s u`. The squared norm of this perpendicular component is:

.. math::

   \begin{aligned}
   \|d - s u\|^2
     &= d \cdot d - 2s(u \cdot d) + s^2(u \cdot u) \\
     &= \|d\|^2 - 2s(u \cdot d) + s^2 \\
     &= \|d\|^2 - 2s^2 + s^2 \\
     &= \|d\|^2 - s^2
   \end{aligned}

We denote this squared norm term as :math:`q`:

.. math::

   q = \|d\|^2 - s^2.

Mathematically, :math:`q` is the squared perpendicular distance from the local pixel
to the pulse-to-reference-pixel line. With these definitions, recall that
:math:`R(d) = \|r_0 + d\|`. Thus,

.. math::

   \begin{aligned}
   R(d)^2 &= \|r_0 + d\|^2 \\
          &= \|R_0 u + d\|^2 = (R_0 u + d) \cdot (R_0 u + d) \\
          &= R_0^2 (u \cdot u) + 2 R_0 (u \cdot d) + d \cdot d \\
          &= R_0^2 + 2 R_0 s + \|d\|^2 \\
          &= R_0^2 + 2 R_0 s + s^2 + q \\
          &= (R_0 + s)^2 + q
   \end{aligned}

where we used :math:`q = \|d\|^2 - s^2` and thus :math:`\|d\|^2 = s^2 + q`.

Finally,

.. math::

   R(d) = \sqrt{(R_0 + s)^2 + q}.

:math:`R(d)` is the exact range to the pixel :math:`x` and the range difference relative to the block reference is

.. math::

   \Delta R(d) = R(d) - R_0.

We started this section with the differential range from pixel :math:`x` to the motion compensation range
:math:`m_p`. We can now write this as:

.. math::

   \begin{aligned}
   \Delta R_p(x) &= \|x - a_p\| - m_p \\
                 &= R_p(x) - m_p \\
                 &= (R_0 + \Delta R(d)) - m_p \\
                 &= \Delta R(d) + (R_0 - m_p)
   \end{aligned}

The bin coordinate for pixel :math:`x` is typically:

.. math::

   b(x) = \frac{\Delta R_p(x)}{\Delta r} + b_{\mathrm{offset}}

Here :math:`\Delta r` is the range-bin spacing and :math:`b_{\mathrm{offset}}` is the centered
range-bin offset used by the SAR backprojection operator. We can now reformulate this as:

.. math::

   \begin{aligned}
   b(x) &= \frac{\Delta R_p(x)}{\Delta r} + b_{\mathrm{offset}} \\
        &= \frac{\Delta R(d) + (R_0 - m_p)}{\Delta r} + b_{\mathrm{offset}} \\
        &= \frac{\Delta R(d)}{\Delta r} + \frac{R_0 - m_p}{\Delta r} + b_{\mathrm{offset}}
   \end{aligned}

We can precompute the right-hand terms as:

.. math::

   b_0 = \frac{R_0 - m_p}{\Delta r} + b_{\mathrm{offset}}

and thus:

.. math::

   b(x) = b_0 + \frac{\Delta R(d)}{\Delta r}

The following sections derive an approximation of :math:`\Delta R(d)` using a Taylor series expansion.

Derivation of the Taylor Expansion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting from

.. math::

   R(d) = \sqrt{R_0^2 + 2 R_0 s + \|d\|^2},

factor out :math:`R_0`:

.. math::

   R(d)
     = R_0
       \sqrt{1 + \frac{2s}{R_0} + \frac{\|d\|^2}{R_0^2}}.

Let

.. math::

   y = \frac{2s}{R_0} + \frac{\|d\|^2}{R_0^2}.

The scalar Taylor series is applied to the one-variable function
:math:`f(y) = \sqrt{1 + y}` about :math:`y = 0`:

.. math::

   f(y) = \sqrt{1 + y}
     = 1 + \frac{y}{2} - \frac{y^2}{8} + \frac{y^3}{16}
       + O(y^4),

so

.. math::

   R(d)
     = R_0
       \left(
         1 + \frac{y}{2} - \frac{y^2}{8} + \frac{y^3}{16}
       \right)
       + O(R_0 y^4).

The order used by ``TaylorFast`` is not the number of scalar Taylor-series
terms retained in :math:`f(y)`. It is the order in the local pixel offset
:math:`d`. Recall that :math:`s = u \cdot d`, so :math:`s` is linear in :math:`d`.

The following table shows how the scalar Taylor-series terms contribute to the
local-offset orders used by the ``TaylorFast`` approximation.

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   * - Scalar term
     - Contribution
     - Local-offset order
   * - :math:`R_0`
     - :math:`R_0`
     - Zeroth order. This is the reference range and cancels in :math:`\Delta R(d) = R(d) - R_0`.
   * - :math:`R_0 y / 2`
     - :math:`s + \dfrac{s^2 + q}{2R_0}`
     - First-order term :math:`s`; second-order term :math:`(s^2 + q)/(2R_0)`.
   * - :math:`-R_0 y^2 / 8`
     - :math:`-\dfrac{s^2}{2R_0} - \dfrac{s(s^2 + q)}{2R_0^2} + O(d^4)`
     - Second-order term :math:`-s^2/(2R_0)`; third-order term :math:`-s(s^2 + q)/(2R_0^2)`.
   * - :math:`R_0 y^3 / 16`
     - :math:`\dfrac{s^3}{2R_0^2} + O(d^4)`
     - Third-order term :math:`s^3/(2R_0^2)`.

The first-order local-offset term is therefore

.. math::

   s.

The second-order local-offset terms are

.. math::

   R_0
   \left(
     \frac{s^2 + q}{2 R_0^2}
     - \frac{4s^2}{8 R_0^2}
   \right)
   =
   \frac{q}{2 R_0}.

The third-order local-offset terms are

.. math::

   R_0
   \left(
     -\frac{4s(s^2 + q)}{8 R_0^3}
     + \frac{8s^3}{16 R_0^3}
   \right)
   =
   -\frac{s q}{2 R_0^2}.

Combining terms gives

.. math::

   R(d)
     =
     R_0
     + s
     + \frac{q}{2 R_0}
     - \frac{s q}{2 R_0^2}
     + O\left(\frac{\|d\|^4}{R_0^3}\right).

Equivalently, the local range delta is

.. math::

   \Delta R(d)
     =
     s
     + \frac{q}{2 R_0}
     - \frac{s q}{2 R_0^2}
     + O\left(\frac{\|d\|^4}{R_0^3}\right).


Second-Order Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The second-order approximation keeps the linear along-range term and the
quadratic cross-range correction:

.. math::

   \Delta R(d)
     =
     s + \frac{q}{2 R_0}.

We then compute the bin coordinate as

.. math::

   b(d) \approx b_0 + \frac{1}{\Delta r}
            \left(s + \frac{q}{2R_0}\right).

The leading omitted term is

.. math::

   \epsilon(d)
     \approx
     -\frac{s q}{2 R_0^2}.

This is the default ``TaylorFast`` implementation. It is most accurate when the pixel block
is small relative to the platform stand-off range and when the along-range
offset :math:`s` remains small.

Third-Order Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``PropSarBpTaylorFastAddThirdOrder`` property instantiates a ``TaylorFast``
kernel variant that also keeps the cubic term:

.. math::

   \Delta R(d)
     =
     s
     + \frac{q}{2 R_0}
     - \frac{s q}{2 R_0^2}.

The corresponding bin coordinate is

.. math::

   b(d) \approx b_0 + \frac{1}{\Delta r}
            \left(
              s
              + \frac{q}{2R_0}
              - \frac{s q}{2R_0^2}
            \right).

With the third-order term included, the leading omitted fourth-order terms are

.. math::

   \epsilon(d)
     \approx
     \frac{s^2 q}{2 R_0^3}
     - \frac{q^2}{8 R_0^3}.

Compared to the second-order form, this requires additional per-pixel arithmetic
and will thus have correspondingly lower throughput than the second-order form.
Keeping the third-order term avoids the short-range accuracy loss that occurs when
the image block is large relative to :math:`R_0` or when the platform is close enough
that the :math:`s q / R_0^2` term is no longer negligible.

Accuracy Considerations
^^^^^^^^^^^^^^^^^^^^^^^

The approximation is local to a thread block. Its accuracy depends on the ratio
between the maximum local pixel offset and :math:`R_0`. For satellite-scale data,
:math:`R_0` is typically very large compared to the dimensions of a CUDA thread block, so
the second-order approximation is likely sufficient. It is an assumption that adjacent
pixels are spatially near one another and thus a compact pixel tile has a relatively small
spatial extent.

For shorter stand-off ranges, larger image tiles, or geometries where the look direction varies rapidly across
a thread block, the third-order term becomes more important. The third-order
property uses a separate kernel instantiation, which avoids a run-time order
dispatch inside the backprojection kernel.

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/SarBp.cu
   :language: cpp
   :start-after: example-begin sar-bp-1
   :end-before: example-end sar-bp-1
   :dedent:

The ``PropSarBpTaylorFastAddThirdOrder`` property is a compile-time MatX operator property. The compute type is
selected separately through ``SarBpParams``; the property only adds the
third-order term to a ``TaylorFast`` launch:

.. literalinclude:: ../../../../test/00_transform/SarBp.cu
   :language: cpp
   :start-after: example-begin sar-bp-2
   :end-before: example-end sar-bp-2
   :dedent:

The property only affects ``SarBpComputeType::TaylorFast``. Other compute types
continue to use their ordinary kernel instantiations.
