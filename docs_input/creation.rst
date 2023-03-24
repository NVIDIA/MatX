.. _creating:

Creating Tensors
====================

Basic construction of tensors in MatX is intended to be very simple with minimal parameters. This allows users of other languages
to pick up the syntax quickly without understanding the underlying architecture. While using the simple API provides good performance,
it lacks flexibility and can prevent your code from running at the highest performance possible. This document walks through the
different ways to construct tensors, and when you should use certain methods over others.

A Quick Primer On MatX Types
----------------------------
The basic type of tensor used in most examples and tests is the ``tensor_t`` object. ``tensor_t`` is the highest-level tensor class, and
provides all of the abstractions for viewing and modifying data, holding storage, and any other metadata needed by a tensor. Because of
their relatively large size, ``tensor_t`` objects are not meant to be passed to GPU devices. In fact, doing so will lead to a compiler error
since ``tensor_t`` uses types that are not available on the device at this time. 

Within a ``tensor_t`` there is an abstract object called ``Storage`` (more on that later), and another inherited class called ``tensor_impl_t``.
``tensor_impl_t`` is a lightweight class containing only the minimum amount of member variables needed to access the data from a GPU kernel. Currently the
member variables are a tensor descriptor and a data pointer. Tensor descriptors will be covered later in this document. 

``tensor_impl_t`` also includes member functions for accessing and modifying the tensor. Examples are all ``operator()`` functions 
(both const and non-const), helper functions for the shape (``Size()`` and ``Stride()``), and utilities for printing on the host. ``tensor_impl_t``
is the type that is passed into GPU kernels, and only contains types that are compatible with CUDA. Furthermore, the total size of the ``tensor_impl_t``
object is as small as possible since these objects can be replicated many times within a single complex expression. Reducing the size of 
``tensor_impl_t`` allows for fastest memory accesses, smaller copies before a kernel launch, and makes extending the code easier.

To convert between a ``tensor_t`` and ``tensor_impl_t`` a type trait called ``base_type`` is available and is used like the following:

.. code-block:: cpp

    typename base_type<I1>::type in1_ = in;

where ``in`` is the ``tensor_t`` object and ``in1_`` will be a ``tensor_impl_t``.

MatX Storage
------------
Within the ``tensor_t`` class is an abstract template parameter called ``Storage``. ``Storage`` objects are always created from a ``basic_storage``
class, which provides all accessor functions common to the underlying storage. ``basic_storage`` can wrap raw pointers using the ``raw_pointer_buffer``
class, smart pointers using the ``smart_pointer_buffer`` class, or any RAII object that provides the required interface. If no user-defined storage
is passed in, MatX will default to allocating a raw CUDA managed memory pointer, and back it using a ``shared_ptr`` for garbage collection. 

When not using implicitly-allocated memory, the user is free to define the storage container type, allocator, and ownership semantics. The container
type requires const and non-const iterators, an allocate function (when applicable), a ``data()`` function to get the raw pointer, and a way to get
the size. Currently both ``std::array`` and ``std::vector`` from the STL follow these semantics, as do both the raw and smart pointer MatX containers.

The allocator type is used when the user passes in a shape without a pointer to existing data. By default, the allocator will use ``matx_allocator``, 
which is a PMR-compatible allocator with stream semantics. The allocator is used for both allocation and deallocation when no user-provided pointer
is passed in and ownership semantics are requested. If a pointer is provided, only the deallocator is used when ownership semantics have been requested.

In general, creating a tensor allows you to choose ownership semantics with creation. By using the ``owning`` type, MatX will take ownership of the pointer
and deallocate memory when the last tensor using the memory goes out of scope. By using the ``non_owning`` type, MatX will use the pointer, but not
perform any reference counting or deallocations when out of scope.

Tensor Descriptors
------------------
Tensor descriptors are a template type inside ``tensor_impl_t`` that provide information about the size and strides of the tensor. While descriptors
are a simple concept, the implementation can have a large impact on performance if not tuned properly. Both the sizes and strides of the tensor are
a template class supporting iterators to access the metadata directly, and utility functions for accessing and computing other values from the metadata.
Descriptors are commonly stored as ``std::array`` types given its compile-time features, but any class meeting the accessor properties can be used.

Dynamic Descriptors
###################
Dynamic descriptors use storage in memory to describe the shapes and strides of a tensor. They can have lower performance than static descriptors
since more memory accesses and offset calculations are needed when accessing tensors, but have higher flexibility given the data is only needed at runtime.

Dynamic descriptors should be used when either the sizes are not known at compile-time, or when interoperating with existing code. As mentioned in the 
introduction, the descriptor size is very important for both kernel performance and launch time. For this reason, the data types used to store both the 
shape and size can vary depending on the size of the tensor parameters. While shape and stride storage types must match in length, the underlying types 
used to store them can be different. This is useful in scenarios where the shape can be expressed as a smaller type than the strides. 

Static Descriptors
##################
If the shapes and strides are known at compile time, static descriptors should be used. Static descriptors compute and store the shape and strides in
constexpr variables, and provide constexpr functions to access both values. When used in a GPU kernel, calling either ``Size()`` or ``Stride()`` emits
an immediate rvalue that the compiler can use for address calculations. This removes all loads and complex pointer arithmetic that could affect the
runtime of a kernel


Creating Tensors
----------------
With the tensor terminology out of the way, it's time to discuss how to create tensors. If there's one thing to take from this article, it's that you
should use ``make_tensor`` or ``make_static_tensor`` wherever possible.

.. note::
    Prefer ``make_tensor`` or ``make_static_tensor`` over constructing tensors directly

Using these helper functions has many benefits:

- They remove the need to specify the rank of a tensor in the template parameters
- They abstract away many of the complex template types of creating a tensor directly
- They hide potentially irrelevant types from the user

All ``make_``-style functions return a ``tensor_t`` object with the template parameters deduced or created as part of the input arguments. ``tensor_t``
only has two required template parameters (type and rank). For simple cases where only implicitly-allocated memory is needed, the default constructor
will suffice. Some situations prevent using the ``make_`` functions, such as when a tensor variable is a class member variable. In this case the type of
the member variable must be specified in the member list. In these scenaries it's expected that the user knows what they are doing and can handle 
spelling out the types themselves. For examples of this, see the simple_radar_pipeline files.

All make functions take the data type as the first template parameter.

Make Variants
#############
There are currently 4 different variants of the ``make_`` helper functions:
- ``make_`` for creating a tensor with a dynamic descriptor and returning by value
- ``make_static_`` for creating a tensor with a static descriptor and returning by value
- ``make_X_p`` for creating a tensor with a dynamic descriptor and returning a pointer
- ``make_static_X_p`` for creating a tensor with a static descriptor and returning a pointer

The ``_p`` variants return pointers allocated with `new` and are expected to be deleted by the caller when finished. Returning smart pointers would
have made this easier, but some users have their own smart pointer wrapper and wouldn't want to unpack the standard library versions.

Within each of these types, there are usually versions both with and without user-defined pointers. These forms are used when an existing device pointer
is passed to MatX rather than having the allocation done when the tensor is created.

Each of these 4 variants can be used with all of the construction types when applicable.

Tensor Class Members
####################
When creating a class that has tensors as member variables there's an issue with the ``make_tensor`` syntax above, in that it depends on
being able to use the ``auto`` keyword to deduce the type. Since type deduction is not possible with member variables, the type must be
declared in the variable list. Once declared, a special version of ``make_tensor`` can be used in the constructor or initialization function
of the class to create the tensor in-place. This allows the user to specify only the rank and type in the member list, and the size can be
specified at initialization without repeating the rank and type. 

.. code-block:: cpp
    class MyClass {
        public:
            MyClass() {
                make_tensor(t, {10, 20});
            }
        private:
            tensor_t<float, 2> t;
    };

In the example above ``make_tensor`` takes an existing tensor as input to construct it in-place. Allocation is only performed once during initialization
and not when the tensor is declared. 

Creating From C Array Or a Brace-Enclosed list
##############################################
Tensors can be created using a C-style shape array from an lvalue, or a brace-enclosed list as an rvalue. The following call the same ``make_`` call:

.. code-block:: cpp

    int array[3] = {10, 20, 30};
    auto t = make_tensor<float>(array);

and

.. code-block:: cpp

    auto t = make_tensor<float>({10, 20, 30});

In the former case the array is an lvalue that can be modified in memory before calling, whereas the latter case uses rvalues. When the sizes are known
at compile time the static version of ``make_`` should be used:

.. code-block:: cpp

    auto t = make_static_tensor<float, 10, 20, 30>();

Notice the sizes are now template parameters instead of function parameters. Both ways can be used interchangeable in MatX code, but the static version
can lead to higher performance.

Similarly, all variants can be called with a user-defined pointer:

.. code-block:: cpp

    auto t = make_tensor<float>(ptr, {10, 20, 30}); // ptr is a valid device pointer

All cases shown above use the default stride parameters. If the strides are not linear in memory, they can be passed in as well:

.. code-block:: cpp

    int shape[3] = {10, 20, 30};
    int strides[3] = {1200, 60, 2};
    auto t = make_tensor<float>(shape, strides);

Creating From A Conforming Shape
################################
As mentioned in the descriptor section, any type that conforms to the shape semantics can be used inside of a descriptor, and can also be passed into the 
``make_`` functions:

.. code-block:: cpp

    std::array<int, 3> = {10, 20, 30};
    auto t = make_tensor<float>(array);

Creating From A Descriptor
##########################
Descriptors (both shapes and sizes) can be used to construct tensors. This is useful when taking an existing tensor descriptor and creating a new tensor from it:

.. code-block:: cpp

    auto d = existingTensor.Descriptor();
    auto t = make_tensor<float>(d);

``t`` is now a tensor with the same shapes and strides of ``existingTensor``.

0-D Tensors
###########
0-D tensors are different than higher ranks since they have no meaningful shape or strides, and therefor don't need those parameters. Empty versions of the
``make_`` helpers existing to create these:

.. code-block:: cpp

    auto t0  = make_tensor<float>();
    auto t01 = make_tensor<float>(ptr);

Custom Storage, Descriptors, and Allocators
###########################################
Within most of the ``make_`` functions, there are choices in the template parameters for custom storage, descriptor, and allocator types. 

Storage
-------
Storage types can be created by wrapping a container object in the ``basic_storage`` class. MatX has a container type built-in for both raw pointers and smart 
pointers, but this can be extended to use any conforming container type. The ``basic_storage`` class does not know about any underlying data structures or ownership; 
this is encapsulated inside of the template type ``C``. For example, to create a custom storage object to wrap a raw pointer:

.. code-block:: cpp

    raw_pointer_buffer<T, owning, matx_allocator<T>> rp{ptr, static_cast<size_t>(desc.TotalSize()*sizeof(T))};
    basic_storage<decltype(rp)> s{std::move(rp)};

The code above creates a new ``raw_pointer_buffer`` object with ownership semantics and the ``matx_allocator`` allocator. A constructor taking a pointer and a
size will not allocate any new data, but track the pointer internally using a smart pointer. If instead ``non_owning`` had been passed as a template parameter, the
pointer would not be tracked or freed. With the container created, the next line passes the container into a ``basic_storage`` object for use inside ``tensor_t``.

Descriptors
-----------
Creating a descriptor can be done by using any conforming descriptor type (See descriptor explanation above). Within MatX, ``std::array`` is used by default
when creating dynamic descriptors. Because of the variable size of the stride and shape, MatX provides helper types for creating descriptors of common types:

- ``tensor_desc_cr_disi_dist<RANK>`` for a dynamic descriptor with ``index_t`` strides and shapes. This is the default descriptor and can also be creating using the type
  ``DefaultDescriptor``. ``index_t`` is defined at compile-time, and defaults to 64-bit
- ``tensor_desc_cr_ds_t<ShapeType, StrideType, RANK>`` a ``std::array``-based descriptor with user-provided types
- ``tensor_desc_cr_ds_32_32_t<RANK>`` is a descriptor with 32-bit sizes and strides 
- ``tensor_desc_cr_ds_64_64_t<RANK>`` is a descriptor with 64-bit sizes and strides
- ``tensor_desc_cr_ds_32_64_t<RANK>`` is a descriptor with 32-bit sizes and 64-bit strides
- ``static_tensor_desc_t<size_t I, Size_t Is...>`` is a static-sized descriptor with the shape and stride created at compile time

To create a descriptor:

.. code-block:: cpp

    const index_t arr[3] = {10, 20, 30};
    DefaultDescriptor<RANK> desc{arr};

In this case we create a default descriptor (based on ``index_t`` sizes) using a C-style array.

