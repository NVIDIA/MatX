.. _black_scholes_example:

Black-Scholes
#############

This example evaluates the Black-Scholes formula for a European call option
using four equivalent MatX implementations. Several different forms of the same 
expression are compared to illustrate different ways of accomplishing the same computation 
in MatX.

The source code is in ``examples/black_scholes.cu``.

Model
-----

For each input element, the example computes the value of a European call
option without dividends:

.. math::

   C = S N(d_1) - K e^{-rT} N(d_2)

where

.. math::

   d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}
                {\sigma\sqrt{T}},
   \qquad
   d_2 = d_1 - \sigma\sqrt{T}.

``N`` is the standard normal cumulative distribution function, and the five
input tensors represent:

* ``K``: strike price
* ``S``: current price of the underlying asset
* ``V``: volatility :math:`\sigma`
* ``r``: continuously compounded risk-free interest rate
* ``T``: time to expiration

The program creates one-dimensional, single-precision tensors containing 100
million values for each input. The tensors are filled independently with
uniform random values. These inputs provide a large element-wise workload for
the implementation comparison; they are not intended to represent a realistic
distribution of market parameters.

Implementations
---------------

The same calculation is expressed in four ways:

1. **MatX expression:** builds the formula from normal MatX arithmetic and
   mathematical operators.
2. **Custom operator:** implements a ``BaseOp`` whose element accessor loads
   each input once and evaluates the complete formula.
3. **``apply()`` lambda:** passes the five values at an element to a device
   lambda.
4. **``apply_idx()`` lambda:** passes the output index and the input operators
   to a device lambda, which loads the five values explicitly.

MatX uses expression templates to fuse the formula into a lazily evaluated
operation. In a sufficiently complex expression, however, repeated appearances
of an input tensor can make it harder for the CUDA compiler to recognize that a
loaded value can be reused. The custom operator and lambda variants make that
reuse more apparent. The example demonstrates the performance effect of those
different representations while retaining MatX expression fusion.

What is measured
----------------

Each implementation is run 100 times on the same CUDA stream. CUDA events are
recorded immediately before and after those runs, and the elapsed time is
divided by 100. The program reports milliseconds per iteration for:

.. code-block:: text

   Time without custom operator = ...ms per iteration
   Time with custom operator = ...ms per iteration
   Time with apply() lambda = ...ms per iteration
   Time with apply_idx() lambda = ...ms per iteration

Each iteration produces 100 million call prices. The reported value therefore
measures the average device execution time of one complete element-wise
evaluation, including its kernel-launch cost. It is useful for comparing how
the four expression forms affect generated GPU work, particularly input-load
reuse.

The timing does not include tensor allocation, random input generation, or
result validation. There is no separate warm-up pass, so one-time initialization
or first-launch effects may be included in the first timed loop. The four
variants also run in a fixed order. Consequently, the output is best treated as
an illustrative comparison on a particular system rather than a controlled
benchmark or an end-to-end option-pricing measurement.

Verification
------------

After all timings are complete, the example reads the four output tensors and
compares every corresponding result. It reports success when all pairwise
differences are at most :math:`10^{-6}`, otherwise it prints the first
mismatching index and values.

This verification checks that the four implementations agree with one another.
It does not compare them with an independent Black-Scholes reference
implementation, and its host-side traversal is not part of the reported
timings.

Building and running
--------------------

Configure MatX with examples enabled, build the ``black_scholes`` target, and
run the resulting executable:

.. code-block:: shell

   cmake -S . -B build -DMATX_BUILD_EXAMPLES=ON
   cmake --build build --target black_scholes
   ./build/examples/black_scholes

The example allocates nine 100-million-element ``float`` tensors, requiring
approximately 3.6 GB for tensor storage alone. Additional CUDA and runtime
memory is also required.
