.. _fusion:

Operator Fusion
###############

When writing a simple arithmetic expression like the following:

.. code-block:: cpp

    (A = B * (cos(C) / D)).run();

Using the typical order of operations rules, we evaluate the expression in parentheses first (``(cos(C) / D)``),
followed by the multiply, then the assignment. Written using standard C++ operator overloading, we would have a 
cosine, division, multiplication, and assignment overload. Each operator performs their respective task, then returns
the value computed. That returned value is stored somewhere (either out to memory or possible in a register), then
the next operator uses that output as input into its own computation. Finally, the assignment writes the value, 
usually out to memory.

While there's nothing wrong with the approach above, it can lead to significant performance penalties, especially
on hardware where the penalty for going to main memory is orders of magnitude higher than other types of memory. In
the worst case, the expression above would have 4 loads and stores (cosine, division, multiplication, assignment).

To avoid this overhead, MatX uses a technique called *lazy evaluation* to reduce the total number of loads and stores.
It does this by overloading each operator so that instead of performing the operation, such as multiplication, instead
it returns an object that *represents* multiplication when it's needed. The entire expression is generates a single
type in C++ representing the equation above, and when we ask for element ``(0,0)`` of ``A`` above, the value is computed
on-the-fly without storing any values. This also implies that you can store an entire expression into a variable and
nothing will be exectuted: 

.. code-block:: cpp

    auto op = (B * (cos(C) / D));

In the example above ``op`` can be further combined with other expressions, which can increase code readability without
loss of performance.

MatX's operator fusion can extend beyond the simple expressions above. Since transforms are also usable inside operator
expressions, this opens the possibility to selectively fuse more complex expressions:

.. code-block:: cpp

    (A = B * fft(C)).run();

The type system can see that we have a multiply where the right-hand side is an FFT transform and the left side is another
operator. This allows MatX to potentially fuse the output of the FFT with a multiply of B at compile-time. In general, the 
more information it can deduce during compilation and runtime, the better the performance will be.