.. _devoperators:

Operators
#########

Operators are the most used concept in MatX. The basics of operators are covered in the :ref:`concepts` documentation. *Transforms* are
a superset of an operator in that they define everything discussed in this document, plus more to do with transforms specifically. If you're
creating a transform operator (one that needs cache, temporary memory, etc), read this document first, then view the :ref:`devtransforms`
document. This document specifically covers the basics of operators, which are used for all element-wise operators. This example focuses
on the `lcollapse` operator because of its simplicity, but also because it uses almost all features that an operator could use.

Generator operators
-------------------
Most of this guide can also be used to create a generator operator. Generators are simpler than the operator described below since they
typically don't take any other operators as input, and instead just generate output based on sizes/shapes. This guide can also be used
to create a generator type.


User-facing functions
---------------------

Before describing what's in an operator class, it's important to point out that the operator class is never created directly by the user;
the user-facing functions/API with a readable name create the operator object and return it from the free function. The `lcollapse` user
function is defined as follows:

.. code-block:: cpp

  template <int DIM, typename T1>
  auto __MATX_INLINE__ lcollapse(const T1 &a)
  {
    if constexpr (DIM <= 1) {
      return a;
    }      
    else {
      return detail::LCollapseOp<DIM, T1>(a);
    }
  }

The `lcollapse` function above is what the user would call, and is what is publicly documented in the official MatX docs. The function uses 
an `auto` return type based on `if constexpr` so that multiple return types can be used. These functions almost always take one or more inputs
where the first input is another operator, and any other inputs are extra parameters. The user functions are also host-only and may not be called
on the device. The signature does not specify host/device, and the compiler will treat it as a host function. Ideally if we decide an operator is a no-op, we just
return the input itself and avoid the cost of constructing and returning a new operator. This is why the first branch checks if we're trying
to collapse 1 or fewer ranks, which is a no-op in this particular case:

.. code-block:: cpp

  if constexpr (DIM <= 1) {
    return a;

The second branch is where we actually want to collapse the input, and an `LCollapseOp` is returned. It should be noted that the user-facing
function does not need to have the same template parameters as the operator. In general, the user functions should be as simple as possible, but
if there is a performance reason inside of the operator for using a template type, that should be preferred over a runtime argument.

.. note::
  Prefer runtime arguments on the user-facing function if there is no performance penalty, and convert to template arguments when creating the
  operator object.

Many operators have another overload where the second parameter is a list of axes:

.. code-block:: cpp

  template <typename InType, int D>
  __MATX_INLINE__ auto max(const InType &in, const int (&dims)[D])

If it makes sense, there should always be an overload taking a list of axes since it allows the user to write cleaner code by avoiding an
explicit permutation in their code. It also is trivial for a developer to add this capability. The only change needed is the function returns
a permuted version of the input operator:

.. code-block:: cpp

  template <typename InType, int D>
  __MATX_INLINE__ auto max(const InType &in, const int (&dims)[D])
  {
    static_assert(D <= InType::Rank(), "reduction dimensions must be <= Rank of input");
    auto perm = detail::getPermuteDims<InType::Rank()>(dims);
    auto permop = permute(in, perm);

    return detail::MaxOp<decltype(permop), InType::Rank() - D>(permop);
  }

In the code above the helper function `detail::getPermuteDims` is used to map the inputs to a `permute` format, and the operator is created
with the permuted op.

.. note::
  Use static_assert with a helpful string wherever possible. Compile-time errors are always better than runtime errors, and many rank and type
  checks in MatX can be done at compile-time.

The last thing to mention about the user-facing function is it's ok to define many versions of the same function with different arguments. Since C++
does not have kwargs as Python does, function overloads are the best tool for allowing the user to call your function in many ways.

Operator class
--------------

With the user-facing function covered, the rest of this document focuses on the operator itself. It's important to mention that the operator
objects exist on both the host and device when using a device executor. This means that you **cannot** have any member variables that are not
allowed on the device. Examples of this would be `std::vector`, `std::shared_ptr`, etc. MatX makes extensive use of NVIDIA's CCCL library to
use functions from the `cuda::std` namespace that are available on both the host and device, but in general, many types will not be.

Going back to the `lcollapse` example, the operator class `LCollapseOp` is defined as follows:

.. code-block:: cpp

  template <int DIM, typename T1>
  class LCollapseOp : public BaseOp<LCollapseOp<DIM, T1>>
  {
    private:
      typename detail::base_type_t<T1> op_;
      index_t size_;  // size of collapsed dim

    public:
      using matxop = bool;
      using value_type = typename T1::value_type;
      using shape_type = index_t;
      using matxoplvalue = bool;
      using self_type = LCollapseOp<DIM, T1>;

      __MATX_INLINE__ std::string str() const { return "lcollapse<" + std::to_string(DIM) + ">(" + op_.str() + ")"; }
      __MATX_INLINE__ LCollapseOp(const T1 &op) : op_(op)
      {
        static_assert(DIM <= T1::Rank(),  "Collapse DIM must be less than or equal to Rank() of operator");
        static_assert(DIM > 1, "Must collapse multiple dims");
        static_assert(T1::Rank() >= 2, "Collapse must be called on operators with rank >= 2");

        // compute size of collapsed dimension
        size_ = 1;

        // Collapse left-most dims
        #pragma unroll
        for(int i = 0 ; i < DIM; i++) {
          size_ *= op_.Size(i);
        }
      }

      template <ElementsPerThread EPT, typename Op, typename... Is>
      static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
      {
        if constexpr (EPT == ElementsPerThread::ONE) {
          // indices coming in
          cuda::std::array<index_t, Rank()> in{indices...};  // index coming in
          cuda::std::array<index_t, T1::Rank()> out;         // index going out

          MATX_LOOP_UNROLL
          for(int i = 1; i < Rank(); i++) {
            // copy all but first input index into out array
            out[DIM + i - 1] = in[i];
          }

          // expand first input index into DIM indices
          auto ind = in[0];
          MATX_LOOP_UNROLL
          for(int i = 0; i < DIM; i++) {
            int d = DIM - i - 1;
            out[d] = ind % op.Size(d);
            ind /= op.Size(d);
          }

          return get_value<EPT>(cuda::std::forward<Op>(op), out);
        }
        else {
          return Vector<value_type, static_cast<index_t>(EPT)>{};
        }
      }   

      template <typename Op, typename... Is>
      static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
      {
        return get_impl<detail::ElementsPerThread::ONE>(cuda::std::forward<Op>(op), indices...);
      }

      template <ElementsPerThread EPT, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        return get_impl<EPT>(cuda::std::as_const(op_), indices...);
      }

      template <ElementsPerThread EPT, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
      {
        return get_impl<EPT>(cuda::std::forward<decltype(op_)>(op_), indices...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        return get_impl<ElementsPerThread::ONE>(cuda::std::as_const(op_), indices...);
      } 

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return T1::Rank() - DIM + 1;
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        if(dim == 0)  // if asking for the first dim, return collapsed size
          return size_;
        else // otherwise return the un-collapsed size from operator
          return op_.Size(DIM + dim - 1);
      }

      __MATX_INLINE__ auto operator=(const self_type &rhs) { 
        return set(*this, rhs); 
      }              

      template<typename R> 
      __MATX_INLINE__ auto operator=(const R &rhs) { 
        if constexpr (is_matx_transform_op<R>()) {
          return mtie(*this, rhs);
        }
        else {          
          return set(*this, rhs); 
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<T1>()) {
          op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<T1>()) {
          op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      template <OperatorCapability Cap>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
        if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          return ElementsPerThread::ONE;
        } else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_));
        }
      }      
  };


The operator text above is taken directly from the `lcollapse` operator. The operator class should be wrapped in the `matx::detail` namespace,
whereas the user function is only in `matx::`. This signals to the user that they should not be constructing this directly, nor is it documented
for them to do so. 

Starting from the beginning of the class:

.. code-block:: cpp

  template <int DIM, typename T1>
  class LCollapseOp : public BaseOp<LCollapseOp<DIM, T1>>

A custom operator is always defined as a class with a postfix of `Op`. The operator class always inherits publicly from `BaseOp`
using a CRTP where the class name and template parameters itself are passed as the type to `BaseOp`. Template parameters on
an operator are optional and should be added on a case-by-case basis. 

In the `private` section of the class we define our member variables:

.. code-block:: cpp

  private:
    typename detail::base_type_t<T1> op_;
    index_t size_;  // size of collapsed dim

Almost all operators store the input operator since it's used inside several functions. Rather than specifying the operator as type `T1` directly,
we use the conversion trait `typename detail::base_type_t`. This will ensure that if the input is a `tensor_t` it is converted to its base
type `tensor_impl_t`. For all other types besides `tensor_t` it has no effect and is equivalent to `T1`. It is also common to store the size
or array of sizes of the operator being defined. Storing the size allows the operator to return sizes quickly without recomputing it. More on this later.

The next section starting with the `public` identifier describes type traits used by MatX:

.. code-block:: cpp

  using matxop = bool;
  using value_type = typename T1::value_type;
  using shape_type = index_t;
  using matxoplvalue = bool;
  using self_type = LCollapseOp<DIM, T1>;

Starting from the top:
- `matxop` tells MatX this is an operator.
- `value_type` should be the *output* value type of this operator. Note that this is not necessarily the input type if there's some kind of transformation.
- `shape_type` describes the type used for the shape of the operator, and should be `index_t` by default.
- `matxoplvalue` tells MatX this operator can be used in an lvalue context. More on this later.
- `self_type` is a helper type to avoid writing the name of the operator multiple times.

It is a common source of bugs when developing an operator to define these incorrectly. Compiler errors about type conversions can usually be traced back
to an incorrect definition of `value_type`.

The next function `str` defines a function used for NVTX ranges to label the name of the operator. It recursively calls the input operator(s) `str()` method
to show a nested function call.

.. code-block:: cpp

  __MATX_INLINE__ std::string str() const { return "lcollapse<" + std::to_string(DIM) + ">(" + op_.str() + ")"; }

The constructor of the operator copies all input parameters needed, and initializes and of its member variables. The constructor is where the operator's
sizes are saved by computing it from another input. 

.. code-block:: cpp

  __MATX_INLINE__ LCollapseOp(const T1 &op) : op_(op)
  {
    static_assert(DIM <= T1::Rank(),  "Collapse DIM must be less than or equal to Rank() of operator");
    static_assert(DIM > 1, "Must collapse multiple dims");
    static_assert(T1::Rank() >= 2, "Collapse must be called on operators with rank >= 2");

    // compute size of collapsed dimension
    size_ = 1;

    // Collapse left-most dims
  #pragma unroll
    for(int i = 0 ; i < DIM; i++) {
      size_ *= op_.Size(i);
    }
  }

The next functions are the most important functions in the operator:

.. code-block:: cpp

  template <ElementsPerThread EPT, typename Op, typename... Is>
  static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
  {
    if constexpr (EPT == ElementsPerThread::ONE) {
      // indices coming in
      cuda::std::array<index_t, Rank()> in{indices...};  // index coming in
      cuda::std::array<index_t, T1::Rank()> out;         // index going out

  MATX_LOOP_UNROLL
      for(int i = 1; i < Rank(); i++) {
        // copy all but first input index into out array
        out[DIM + i - 1] = in[i];
      }

      // expand first input index into DIM indices
      auto ind = in[0];
  MATX_LOOP_UNROLL
      for(int i = 0; i < DIM; i++) {
        int d = DIM - i - 1;
        out[d] = ind % op.Size(d);
        ind /= op.Size(d);
      }

      return get_value<EPT>(cuda::std::forward<Op>(op), out);
    }
    else {
      return Vector<value_type, static_cast<index_t>(EPT)>{};
    }
  }

  template <typename Op, typename... Is>
  static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
  {
    return get_impl<detail::ElementsPerThread::ONE>(cuda::std::forward<Op>(op), indices...);
  }  

The `get_impl` function is used by the executor to get values from the operator at specified indices. Usually `get_impl` is only implemented 
in operators where `operator()` would duplicate this code. Instead, we make one accessor function called `get_impl` that various versions of 
`operator()` call. The `ElementsPerThread` template parameter is used to specify the number of elements per thread that are being accessed. This 
value could be compiled with EPT values that the operator does not support, but it will not be called with unsupported values. `EPT` should be 
used as a parameter to `get_value` to ensure the correct vectorization is used. Note there is also a version of `get_impl` that does not take
an `ElementsPerThread` parameter, and is used in the default path where vectorization is not used.

The various versions of `operator()` are defined as follows:

.. code-block:: cpp

  template <ElementsPerThread EPT, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
  {
    return get_impl<EPT>(cuda::std::as_const(op_), indices...);
  }

  template <ElementsPerThread EPT, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
  {
    return get_impl<EPT>(cuda::std::forward<decltype(op_)>(op_), indices...);
  }

  template <typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
  {
    return get_impl<ElementsPerThread::ONE>(cuda::std::as_const(op_), indices...);
  }

  template <typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
  {
    return get_impl<ElementsPerThread::ONE>(cuda::std::forward<decltype(op_)>(op_), indices...);
  }

`operator()` is defined on both the host and device, and is used by the executor to get values from this operator at specified indices. Most operators 
support arbitrary-rank inputs, and a parameter pack is used as input to reflect this. If the operator will never need more than a certain rank input, 
it's also valid to only specify `operator()` with a fixed number of indices. In general, this should be avoided since it prevents batching and broadcasting.

Sometimes there is a `const` and a non-`const` version of `operator()` defined. If the operator will never be written to as an lvalue, 
it's valid to only define the `const` version. Either way, the non-`const` version should call the `const` version with a cast as shown above to combine 
code in `get_impl`. As with `get_impl` above, `operator()` also needs a version without the `ElementsPerThread` parameter when the fallback path is used.

.. note::
  Since `operator()` may be called on both the host and device, it's important to make sure that it's as performant as possible. Using extra stack memory, 
  big loops, etc, are generally bad practice here unless there's a good reason to do so. This function will be called by *every* thread in almost all
  cases.

The body of the `get_impl()` function should contain all the business logic your operator uses to perform the function it's designed for. For example, 
`lcollapse` is intended to take a higher-rank operator, collapse some number of left-most dimenions, and allow accessing that new collapsed operator. To
perform that function `get_impl()` indexes the higher-rank operator using indices for its own rank, which is lower. Some operators return the same rank, 
while others return higher ranks. It's worth looking at an operator that's most similar to yours to see how it might be done.

The next function describes the rank of our operator:

.. code-block:: cpp

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
  {
    return T1::Rank() - DIM + 1;
  }

The rank of the operator is its output rank. The output rank is usually based on the input operator rank, but not always. For example, a `flatten` operator
*always* returns a 1D operator regardless of the input rank. `Rank()` is a `static constexpr` function since ranks in MatX are defined at compile-time
currently.

.. code-block:: cpp

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
  {
    if(dim == 0)  // if asking for the first dim, return collapsed size
      return size_;
    else // otherwise return the un-collapsed size from operator
      return op_.Size(DIM + dim - 1);
  }

`Size()` defines the size of a specific dimension of our operator. Like `Rank()`, `Size()` is usually dependent on the sizes of the input operator. The function
prototype on `Size` should be identical for all operators. Declaring it as `constexpr` allows it to be used in a compile-time context optionally if static tensors
are used.

.. code-block:: cpp

  template<typename R> 
  __MATX_INLINE__ auto operator=(const R &rhs) {          
    return set(*this, rhs); 
  }  

If the operator provides lvalue semantics, meaning it can be assigned on the left-hand side of an expression, it must also define `operator=` when assigning to itself.
The `operator=` function is identical on any operator implementing lvalue semantics.

The next functions defined are `PreRun` and `PostRun`. While these are defined in all operators, they're only used in transform operators and will be covered in that 
document. For regular operators these functions can be copied and pasted from other operators:

.. code-block:: cpp

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept

Last, the `get_capability` function is defined. This function is used by the executor to determine if the operator supports a specific capability.

.. code-block:: cpp

  template <OperatorCapability Cap>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
    if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
      return ElementsPerThread::ONE;
    } else {
      auto self_has_cap = capability_attributes<Cap>::default_value;
      return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_));
    }
  }

An operator only needs a branch above if it wishes to override the support for a capability. For example, if an operator does not support vectorization,
it should define a branch for `OperatorCapability::ELEMENTS_PER_THREAD` and force the value to `ElementsPerThread::ONE` (by default it allows up to 
`ElementsPerThread::THIRTY_TWO`). Any capability the operator wishes to use the default values for simply uses code similar to what's in the `else` branch. 
The `combine_capabilities` function must be called with all input operators to the operator being defined so that they are aggregated properly.

With the operator class written, there are only a few loose ends to finish to make the operator useable. 

- First, the new operator should be added to matx/core/operators/operators.h as a new include. 
- Next, unit tests should be written to exercise as much functionality of the operator as possible. This include all user-facing overloads and multiple ranks. Code 
  coverage reports can be viewed at https://coveralls.io/github/NVIDIA/MatX.
- Lastly, documentation should be written for the new operator. Please see :ref:`devdocs` for more information on writing documentation.