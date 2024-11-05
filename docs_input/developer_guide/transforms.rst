.. _devtransforms:

Transforms
##########

Transform operators are more complex than regular operators in that they do everything a normal operator does, but also 
maintains state during, after the call, or both. Transform operators may rely on other backend kernels or libraries to 
accomplish the task, or it can simply be a composition of other MatX functions. For example, a `matmul()` or `fft()` use 
CUDA, FFTW, or BLAS backends, while functions like `median()` use other MatX functions. Transform operators usually have two 
required pieces: the operator file, and one or more transform files. 

.. note::
  If you have not read the :ref:`devoperators` document, please read that first since this document builds on all the concepts 
  mentioned there.

.. note::
  From a user perspective, there is *no difference* between a regular operator or a transform operator. The distinction is 
  for developers of MatX since the requirements of the two are different.

Much like a regular operator, the operator file (in the operators/ directory) defines the operator type that is passed to and from the 
device, but unlike  regular operators, it does not have any of the "business logic" of what the operator does inside of it. For example, 
the `FftOp` operator does not contain any logic to perform an FFT. This is intentional so that the operator purely defines sizes, shapes, 
types, and other properties of the transform. It then calls a generic executor function that is implemented inside of the transform 
file (in the transforms/ directory). There can be many transform files, each with an overload for a different executor. This allows the 
same interface to be used across many executors, leaving the library implementation isolated from the MatX pieces. 

Like regular operators, transform operators define all the functions of the *operator interface*: `Rank()`, `Size()`, and 
at least one `operator()`. 

Transform Operators
-------------------

External library operators rely on either vendor libraries or MatX custom kernels to execute their functions. Examples are 
cuFFT, cuBLAS, FFTW, convolution, etc. Often these dependencies are optional if they don't come with the CUDA toolkit. Compared 
to regular operators, 

The main difference with regular operators is transform operators often need to define temporary 
memory or write to multiple outputs to complete their work without blocking. For example, if a user writes:

.. code-block:: cpp

  (out = fft(a) * shift(b, 3)).run(stream);

They would expect this call to not block and execute on the stream specified. But what if the underlying library for this call is 
cuFFT or FFTW? Those libraries in general do not provide a way to integrate the `shift` function into the output. The example above 
is relatively simple, but MatX allows any arbitrary sequence of operators in a single expression, which further complicates what 
an operator needs to do. 

To illustrate this concept, the `fft` operator is used as an example. Similar to a regular operator, `fft` defines two user-facing 
functions to create their operators:

.. code-block:: cpp

  template<typename OpA>
  __MATX_INLINE__ auto fft(const OpA &a, const int32_t (&axis)[1], uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD)

  template<typename OpA>
  __MATX_INLINE__ auto fft(const OpA &a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD)

Inside of the operator class, there are several notable differences from the regular operators from the previous tutorial. First, there 
are two new data members:  

.. code-block:: cpp

  mutable detail::tensor_impl_t<ttype, OpA::Rank()> tmp_out_;
  mutable ttype *ptr = nullptr;

`tmp_out_` is a temporary uninitialized tensor. Only the type and rank are defined, but the size is undefined at this point. `ptr` is 
a pointer of the same type as the tensor that will be used later to allocate memory for the tensor.

For transforms a new type trait is also needed: 

.. code-block:: cpp

  using matx_transform_op = bool;

This allows MatX to identify a transform operator for performance reasons.

Next, the operator defines a `Data()` function for cases where the output memory is needed to be accessed. This function is only defined
and used on the host.

.. code-block:: cpp

  __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

Compared to regular operators, the `operator()` function is comparitively simple:

.. code-block:: cpp

  template <typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
  {
    return tmp_out_(indices...);
  }  

The only job of `operator()` in most transform operators is to return the value at that temporary tensor's indices. The reason this works is 
by the time `operator()` is called we are *guaranteed* to have completed the transform and the output in `tmp_out_` is valid.

The biggest difference from a regular operator is the introduction of a new `Exec` member function:

.. code-block:: cpp

  template <typename Out, typename Executor>
  void Exec(Out &&out, Executor &&ex) const {
    fft_impl(cuda::std::get<0>(out), a_, fft_size_, norm_, ex);
  }

`Exec()` is responsible for executing the transform with all the appropriate input parameters into the output tensor. Note that both the output 
type and the executor type are template parameters on the function. This means that to add a new implementation of an fft for a new executor 
only `fft_impl` needs to be updated with the new executor. Everything else is exactly the same, and even more important, all the same unit 
tests that work on existing executors can automatically run (and hopefully pass) on new executors.

You may be wondering "why is `out` a function parameter instead of just using `tmp_out_` directly?". This is a good question, and is related to 
performance. Given the following code:

.. code-block:: cpp

  (a = fft(b)).run(stream);
  (a = fft(b) * 5).run(stream);

What is the difference between those lines (besides the multiply)? In the first case, the `fft` operator *does not* have to allocate temporary 
memory before writing out to `a`. In the second case the `fft` operator allocates and writes the output into a temporary tensor that is then 
used for the multiply. The short version is that we want to avoid allocating and copying if necessary, so our `Exec` function must take the output 
as a parameter. Perhaps more important is that even though MatX may implement those allocations like that now, if in the future the FFT library is 
capable of performing the `* 5` on the output, nothing changes from the user's perspective.

The last functions to discuss are the Pre/PostRun functions:

.. code-block:: cpp

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
  {
    if constexpr (is_matx_op<OpA>()) {
      a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }         
  }      

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
  {
    InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));  

    detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);         

    Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
  {
    if constexpr (is_matx_op<OpA>()) {
      a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }

    matxFree(ptr); 
  }

`PreRun` runs *before* the start of the executor calling `run()`, while `PostRun()` runs *after* `run()` is complete. `InnerPreRun` is separate so 
that the operator can run nested PreRuns without any allocations.

Starting with `InnerPreRun`, all it does is conditionally call the member operator's `PreRun()`. If there were multiple inputs, it would 
call both `PreRun` functions for each of those operators. 

`PreRun` is for running any work that needs to be done before `run()`. For most transforms this means allocating our temporary `tmp_out_` tensor 
with the appropriate dimensions, then executing our transform into that memory.

Conversely, `PostRun` does any cleanup necessary after the execution of the `run()` statement completes. For most transforms this means freeing 
our temporary memory used for `tmp_out_`.

While these calls recursively call other operators' `Pre/PostRun`, it does so following the usual C++ order of operations. Looking at the FFT 
convolution example: 

.. code-block:: cpp

  (sig_freq = ifft(fft(sig_time, filtered_size) * fft(filt_time, filtered_size))).run(exec);

The order of `PreRun` will be `MulOp` for the multiply, `fft`, `fft`, then `ifft`. This guarantees that the `fft`s and the multiply are complete 
before the `ifft` tries to access their output.

Implementation Function
-----------------------

The implementation function defines how the transform is done.

Library Backend Transforms
__________________________

The FFT example above used an implementation function called `fft_impl`. In the case of an FFT, the implementation is quite complex and has multiple 
executor overloads for cuFFT and FFTW. In general, for any transform there will be common functions that are not related to the executor and can be 
placed into a common file. For example, for an FFT saving off the strides and batch number are common functions that apply to both host and device 
code. 

For the executor-specific implementation file there are typically three parts: the `impl` function, the cache, and the transform class. 

The `impl` function has been described briefly above, but its main job is to do basic checks on input, convert any required operators, then run 
the execution. As an example from the 1D FFT:

.. code-block:: cpp

  template <typename OutputTensor, typename InputTensor>
  __MATX_INLINE__ void fft_impl(OutputTensor o, const InputTensor i,
          uint64_t fft_size, FFTNorm norm, const cudaExecutor &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    const auto stream = exec.getStream();

    // converts operators to tensors
    auto out = getCufft1DSupportedTensor(o, stream);
    auto in_t = getCufft1DSupportedTensor(i, stream);

    if(!in_t.isSameView(i)) {
      (in_t = i).run(stream);
    }

    auto in = detail::GetFFTInputView(out, in_t, fft_size, exec);

    // Get parameters required by these tensors
    auto params = detail::matxCUDAFFTPlan_t<decltype(out), decltype(in)>::GetFFTParams(out, in, 1);
    params.stream = stream;

    using cache_val_type = detail::matxCUDAFFTPlan1D_t<decltype(out), decltype(in)>;
    detail::GetCache().LookupAndExec<detail::fft_cuda_cache_t>(
      detail::GetCacheIdFromType<detail::fft_cuda_cache_t>(),
      params,
      [&]() {
        return std::make_shared<cache_val_type>(out, in, stream);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->Forward(out, in, stream, norm);
      }
    );

    if(!out.isSameView(o)) {
      (o = out).run(stream);
    }
  }  

Starting with this:

.. code-block:: cpp

  auto out = getCufft1DSupportedTensor(o, stream);
  auto in_t = getCufft1DSupportedTensor(i, stream);

  if(!in_t.isSameView(i)) {
    (in_t = i).run(stream);
  }  

These functions check to see if the input operator is compatible with the library. In the case of cuFFT, it only supports tensor inputs, so 
any top that's not a tensor needs to async-allocate a tensor and copy to it before executing. This is what `getCufft1DSupportedTensor` does. 
Each type of transform may have their own "support" function tailored to its limitations.

The next section of code is for caching and executing the transform:

.. code-block:: cpp

  using cache_val_type = detail::matxCUDAFFTPlan1D_t<decltype(out), decltype(in)>;
  detail::GetCache().LookupAndExec<detail::fft_cuda_cache_t>(
    detail::GetCacheIdFromType<detail::fft_cuda_cache_t>(),
    params,
    [&]() {
      return std::make_shared<cache_val_type>(out, in, stream);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Forward(out, in, stream, norm);
    }
  );

Caching is used for transforms that maintain some kind of state like a plan or handle for executing the transform. That plan/handle is typically 
expensive to create, so creating it once for a specific signature improves performance. What gets cached is dependent on the transform. FFT's cache 
looks like this:

.. code-block:: cpp

  struct FftCUDAParamsKeyHash {
    std::size_t operator()(const FftCUDAParams_t &k) const noexcept
    {
      return (std::hash<uint64_t>()(k.n[0])) + (std::hash<uint64_t>()(k.n[1])) +
            (std::hash<uint64_t>()(k.fft_rank)) +
            (std::hash<uint64_t>()(k.exec_type)) +
            (std::hash<uint64_t>()(k.batch)) + (std::hash<uint64_t>()(k.istride)) +
            (std::hash<uint64_t>()((uint64_t)k.stream));
    }
  };

  struct FftCUDAParamsKeyEq {
    bool operator()(const FftCUDAParams_t &l, const FftCUDAParams_t &t) const noexcept
    {
      return l.n[0] == t.n[0] && l.n[1] == t.n[1] && l.batch == t.batch &&
            l.fft_rank == t.fft_rank && l.stream == t.stream &&
            l.inembed[0] == t.inembed[0] && l.inembed[1] == t.inembed[1] &&
            l.onembed[0] == t.onembed[0] && l.onembed[1] == t.onembed[1] &&
            l.istride == t.istride && l.ostride == t.ostride &&
            l.idist == t.idist && l.odist == t.odist &&
            l.transform_type == t.transform_type &&
            l.input_type == t.input_type && l.output_type == t.output_type &&
            l.exec_type == t.exec_type && l.irank == t.irank && l.orank == t.orank ;
    }
  };  

The `FftCUDAParamsKeyHash` type creates a quick hash that can be used for the initial comparison inside of the map. Once a cache hit occurs, the 
second function `FftCUDAParamsKeyEq` is used to match all parameters needed for that cache. The cached parameters should be specific enough 
so that workspaces can be reused when possible, but not so specific that too many caches are created unnecessarily.

The rest of the transform, including the class used for the transform is up to the developer on how best to handle the transform, and no
two types are the same. Some useful examples to look at are fft_cuda.h and matmul_cuda.h.

MatX-Based Transforms
_____________________

Some transforms can be implemented purely using MatX functions. This type is preferrable when possible because it allows the developer to build 
on an existing/tested set of functions without rewriting new pieces. These transforms typically are implemented entirely in their `impl` function 
and don't require any extra state. 

The `softmax_impl` function is implemented the following way:

.. code-block:: cpp
  
  template <typename OutType, typename InType>
  void __MATX_INLINE__ softmax_impl(OutType dest, const InType &in, cudaStream_t stream = 0)
  {
    MATX_NVTX_START("softmax_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

    auto tmp_sum = make_tensor<typename InType::value_type>({}, MATX_ASYNC_DEVICE_MEMORY, stream);
    auto tmp_max = make_tensor<typename InType::value_type>({}, MATX_ASYNC_DEVICE_MEMORY, stream);
    max_impl(tmp_max, in, cudaExecutor{stream});
    sum_impl(tmp_sum, exp(in - tmp_max), stream);
    (dest = exp(in - tmp_max) / tmp_sum).run(stream);
  }

The implementation is not only easily readable, but leverages other functions in MatX that have already been optimized. Temporary tensors are created 
inside the function, and using RAII in C++ they will also be freed asynchronously when they go out of scope. No cache is needed for MatX-based 
implementations since they indirectly use the caches for the functions they call.


Multiple Return Transforms
--------------------------

Some operators require multiple ouputs or return types. Examples are `svd()` returning `S`, `V`, and `D` tensors. These operators are implemented 
mostly the same as other transforms with the following exceptions:

- `Rank()` and `Size()` are not called, and therefor can be deleted
- They are not allowed to be called as part of larger expressions since the output would be ambiguous. Instead, they must only define `Exec` with 
  a tuple output equal to match the number of outputs it expects.