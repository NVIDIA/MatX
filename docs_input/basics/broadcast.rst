.. _broadcast:

Broadcasting
############

MatX supports broadcasting of values to operators. Broadcasting allows lower-ranked operators to have
their values repeated into a higher-ranked operator. Broadcasting may allow for both memory and cache improvements
by storing fewer values in memory and loading the same value from cache. 

In its simplest form, scalars can be broadcast to an arbitrary-sized tensor:

.. code-block:: cpp

    (t = 4).run();

In the example above the value `4` is broadcast to every value in `t`, regardless of the rank and sizes
of `t`. This same rule applies with 0D operators, which are effectively scalar values:

.. code-block:: cpp

    auto t0 = make_tensor<float>();
    auto t2 = make_tensor<float>({10, 20});
    (t2 = t0).run();

Besides scalars, other rank operators can also be broadcasted:

.. code-block:: cpp

    auto t1 = make_tensor<float>({20});
    auto t2 = make_tensor<float>({10, 20});
    (t2 = t1).run();

In fact, any operator can be broadcasted to another operator as long as the dimensions are *compatible*. For
broadcasting, compatible means either:

* The value being broadcasted from is either a scalar or 0D operator
* The operator being broadcasted from matches all the right-most dimensions of the operator assigned to

The first rule is discussed above. The second rule can be explained with this example:

.. code-block:: cpp

    auto t3 = make_tensor<float>({3, 4, 20});
    auto t5 = make_tensor<float>({10, 6, 3, 4, 20});
    (t5 = t3).run();

In this example `t3`'s dimensions (3, 4, 20) match the right-most (fastest-changing) dimensions in `t5`, so
the values in `t3` can be broadcasted to `t5`. During this assignment the same values in memory in `t3` are 
loaded repeatedly into the left-most dimensions of `t5` (10, 6).
