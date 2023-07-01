.. _printing:

Printing
########

MatX allows printing of any operator or tensor for debugging. If the operator is not backed by memory MatX will
fetch all values based on the size and rank. If the operator is a tensor, MatX can determine whether
the memory is on the host or device and print the tensor appropriately. The `print` function takes an 
operator as the first parameter, and optionally the number of elements to print in each dimension. The
value of 0 is a special placeholder for "all", and using it prints all values in that dimension. 

To print a tensor:

.. code-block:: cpp

    auto t = make_tensor<TypeParam>({3});
    (t = ones(t.Shape())).run();
    print(t);

In this case MatX is fetching the data from the device and printing it to the screen. However, storing
in a tensor is optional. The following code gives the same result:

.. code-block:: cpp

    auto t = make_tensor<TypeParam>({3});
    print(ones(t.Shape()));

In this case MatX is printing data from the operator without going to the device.

To print only two elements:

.. code-block:: cpp

    auto t = make_tensor<TypeParam>({3});
    print(ones(t.Shape()), 2);

For a multi-dimension operator, the following code prints only the first element in the first dimension, every
element in the second dimension, and 10 elements in the last dimension.

.. code-block:: cpp

    auto t = make_tensor<TypeParam>({3, 10, 20});
    print(ones(t.Shape()), 1, 0, 10);
