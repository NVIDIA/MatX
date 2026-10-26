.. _tensor_methods:

Tensor Class Methods
####################

This page lists the public methods of ``matx::tensor_t`` and its base class,
``matx::detail::tensor_impl_t``, followed by the dynamic, sparse, and distributed
tensor interfaces. Overloads are grouped by method name in the tables; the
generated ``tensor_t`` reference at the end gives its full declarations.

For an introduction to tensors, see :ref:`tensor_api`. Prefer
:ref:`make_tensor` to direct construction. Include ``<matx.h>`` to use the
regular tensor API. Dimensions and element indices are zero-based, and strides
are measured in elements.

.. contents:: On this page
   :local:
   :depth: 2

Dense tensors
=============

``tensor_t<T, RANK, Desc>`` owns or shares a storage container and inherits
shape, stride, pointer, and element-access methods from ``tensor_impl_t``.
Copy construction shares storage; it does not copy elements. Methods that
create views generally share the same allocation, so writes through one view
are visible through the others.

Shape and layout
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Result and limitations
   * - ``Rank()``
     - Static, compile-time rank ``RANK``. A scalar has rank zero.
   * - ``Size(dim)``
     - Extent of dimension ``dim``; requires ``0 <= dim < Rank()``.
   * - ``Shape()``
     - Copy of the descriptor's shape container.
   * - ``Stride(dim)``
     - Element stride of dimension ``dim``. Indexed strides on ``tensor_t``
       require rank at least one.
   * - ``Strides()``
     - Copy of the descriptor's stride container.
   * - ``TotalSize()``
     - Product of the extents (one for a rank-zero scalar).
   * - ``Lsize()``
     - Size of the last dimension. Requires rank at least one.
   * - ``Bytes()``
     - ``TotalSize() * sizeof(T)``. This is the logical element byte count,
       not the allocation size or the memory span of a strided view.
   * - ``IsContiguous()``
     - Whether the descriptor has contiguous row-major layout. For the default
       runtime descriptor, the last stride must be one and each preceding stride
       must equal the product of the subsequent extents. A rank-zero descriptor
       is contiguous. This query does not check pointer validity or memory space.
   * - ``Descriptor()``
     - Copy of the descriptor, including shape and strides.
   * - ``isSameView(other)``
     - Whether another tensor view has the same rank, data pointer, shape, and
       strides. This compares metadata, not element values.

For example, permuting a dense matrix changes its layout without moving data:

.. code-block:: cpp

   auto a = matx::make_tensor<float>({2, 3});
   auto transposed = a.PermuteMatrix();

   bool contiguous = a.IsContiguous();            // true
   bool transposed_contiguous = transposed.IsContiguous(); // false
   auto rows = a.Size(0);                         // 2
   auto row_stride = a.Stride(0);                 // 3 elements
   auto elements = a.TotalSize();                 // 6
   auto bytes = a.Bytes();                        // 6 * sizeof(float)

These metadata queries do not synchronize outstanding GPU work. Shape and
stride changes can make a view noncontiguous even when its original allocation
was contiguous.

Data access and initialization
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Behavior
   * - ``Data()``
     - Pointer to the start of this view, which may be offset from the allocation
       base. A pointer returned on the host may still refer to device memory.
   * - ``IsInitialized()``
     - Whether ``Data()`` is non-null. Does not validate shape, ownership, or
       accessibility.
   * - ``operator()(indices...)``; ``operator()(index_array)``
     - Access an element using strides, with const and non-const overloads.
       Supply exactly ``Rank()`` integral indices, or a
       ``cuda::std::array<index_t, RANK>``. Use ``a()`` for a scalar. The
       capability-template overloads are used by executors for vector access.
       Planar complex elements use a proxy for reads and writes.
   * - ``operator[](indices...)``
     - Equivalent element access for rank greater than zero, when the compiler
       provides C++23 multidimensional subscripting
       (``__cpp_multidimensional_subscript >= 202211L``).
   * - ``GetPointer(indices...)``
     - Pointer at the stride-adjusted element offset for ordinary dense storage.
   * - ``data_ptr<CapType>(indices...)``
     - Capability-aware element pointer for dense storage; returns null for
       sparse storage. Intended for executor integration.
   * - ``SetVals(value)``; ``SetVals(nested_initializer_list)``
     - Immediate host initialization for ranks zero through four. Real tensors
       take a scalar or lists nested to match the rank. CUDA complex tensors
       also support an extra list level containing real/imaginary pairs.
       The pointer must be host-accessible and the supplied values must fit.
   * - ``IsHostAccessiblePointer()``
     - Checks host accessibility through the CUDA driver, raising
       ``matxNotSupported`` when its pointer query fails.

Direct element access requires memory accessible to the calling processor.
Synchronize relevant GPU work before reading or writing its results on the
host. ``operator()`` and ``SetVals`` perform immediate accesses; assignment
expressions below defer execution.

Assignment and ownership
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Behavior
   * - ``operator=(rhs)``
     - Creates a lazy element assignment from a tensor, operator, or scalar.
       Execute the returned expression with ``.run(executor)`` or ``.run(stream)``.
   * - ``operator+=``, ``operator-=``, ``operator*=``, ``operator/=``
     - Lazy arithmetic compound assignment, with tensor and generic RHS overloads.
   * - ``operator%=``, ``operator<<=``, ``operator>>=``
     - Lazy remainder and shift assignment for element types supporting the
       corresponding operations.
   * - ``operator|=``, ``operator&=``, ``operator^=``
     - Lazy bitwise compound assignment for compatible element types.
   * - ``Shallow(rhs)``
     - Rebinds the tensor's pointer, descriptor, and shared storage to ``rhs``.
       Does not assign elements.
   * - ``View()``
     - Returns a copy of the tensor sharing its storage and metadata values.
   * - ``GetStorage()``
     - On a non-const tensor, returns a storage copy sharing ownership. On a const
       tensor, returns a const storage reference that must not outlive the tensor.
   * - ``GetRefCount()``
     - Reference count reported by the storage container.
   * - ``Reset(data, shape)``; ``Reset(data)``; ``Reset(data, local_data)``
     - Replaces storage with a non-owning wrapper. The shape overload recomputes
       the descriptor; the other overloads retain it. The two-pointer form uses
       ``data`` as the storage base and ``local_data`` as the view start. The
       caller must keep the external allocation alive.
   * - ``SetData(data)``; ``SetLocalData(data)``
     - Low-level inherited pointer setters. They change the view pointer only;
       they do not update the storage owner or descriptor. Prefer ``Reset`` when
       rebinding a ``tensor_t`` to external storage.
   * - ``swap(lhs, rhs)``
     - Friend function that swaps pointers, descriptors, and storage owners.
   * - ``str()``; ``set_name(name)``
     - Gets or sets the tensor's diagnostic name.

For example, ``auto b = a`` shares storage, ``b.Shallow(a)`` rebinds an existing
tensor, and ``(b = a).run(exec)`` assigns elements into ``b``'s current storage.
Compound assignment requires the same explicit execution step.

Views
-----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Behavior and restrictions
   * - ``View(shape)``; ``View<M, R>(shape)``
     - Creates a contiguous view, optionally changing element type with the
       generic shape overload. The source must be contiguous and the requested
       bytes must fit in storage. The C-array shape overload deduces the new
       rank. No elements are copied.
   * - ``Permute(dims)``
     - Reorders dimensions using a ``cuda::std::array<int32_t, RANK>`` or C array.
       Each dimension must appear exactly once; requires rank at least one.
   * - ``PermuteMatrix()``
     - Swaps the final two dimensions, preserving leading batch dimensions.
       Requires rank at least two.
   * - ``Slice<N>(firsts, ends)``;
       ``Slice<N>(firsts, ends, strides)``
     - Creates a view of a subset, with optional steps and output rank ``N``.
       Accepts CUDA arrays or C arrays. End indices are exclusive;
       ``matxEnd`` selects a dimension's end, ``matxDropDim`` drops it, and
       ``matxKeepStride`` retains the original stride. Prefer the free
       ``slice()`` operator for new code.
   * - ``Clone<N>(clones)``
     - Adds broadcast dimensions using zero strides, without copying data.
       Accepts a CUDA array or C array; use ``matxKeepDim`` for each existing
       dimension. Writes through broadcast dimensions alias the same elements.
   * - ``OverlapView(windows, strides)``
     - Creates a rank-two overlapping window view of a rank-one tensor. Window
       and step arrays have one entry; the step must be positive and smaller
       than the window. Trailing elements that do not fill a window are omitted.
   * - ``RealView()``; ``ImagView()``
     - Real or imaginary component view of an interleaved complex tensor, using
       doubled element strides. These views use non-owning storage: keep the
       original allocation alive while using them.

Prefetch and interchange
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Behavior
   * - ``PrefetchDevice(stream)``; ``PrefetchHost(stream)``
     - Asynchronously prefetches managed memory to the current CUDA device or
       CPU, using the supplied stream. Requires memory supported by CUDA
       prefetching. The requested range is ``TotalSize() * sizeof(T)`` from
       ``Data()``; use with contiguous views.
   * - ``ToDlPack()``; ``ToDlPackVersioned()``
     - Exports a legacy or versioned DLPack managed tensor, sharing the tensor's
       lifetime through its manager context. Prefer the versioned API. The
       consumer must eventually call the returned object's ``deleter``.
   * - ``ToDlPackImpl<ManagedType>()``;
       ``FreeDLPackCommon_(managed_tensor)``
     - Public implementation helpers for the export methods and their deleter.
       Use the export methods for interoperability.

Base implementation and integration methods
===========================================

``detail::tensor_impl_t`` is the lightweight, non-owning base used in device
code. Its pointer and descriptor do not keep an allocation alive. It provides
the common metadata and access methods above, lazy assignment and compound
assignment operators, and CUDA-array forms of ``Slice``, ``Clone``, ``Permute``,
and ``OverlapView``. The owning overrides in ``tensor_t`` preserve storage
ownership. Host/device availability is determined by each declaration's
``__MATX_HOST__`` and ``__MATX_DEVICE__`` annotations; not every public method
is callable from a CUDA kernel.

The base constructors accept no arguments, a scalar pointer, a shape, a shape
and strides, a pointer and shape (with optional strides), a descriptor, or a
pointer and descriptor. Copy/move construction, move assignment, and destruction
are defaulted. Its ``Shallow(rhs)`` and ``copy(rhs)`` methods copy pointer and
descriptor metadata only, and its friend ``swap`` swaps these fields.

The following publicly declared helpers support MatX's implementation. They
are listed for completeness; ordinary tensor use does not require calling them.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``SliceImpl<N>(firsts, ends, strides)``
     - Computes the descriptor and adjusted data pointer for slicing.
   * - ``CloneImpl<N>(clones)``; ``PermuteImpl(dims)``;
       ``OverlapViewImpl<N>(windows, strides)``
     - Computes descriptors for the corresponding view methods.
   * - ``DimStride<DIM, RANK_VAL, CapType>(index)``;
       ``GetOffsetOptimized<CapType>(indices...)``
     - Computes capability-aware dimension and element offsets.
   * - ``GetVal<CapType, I>(tuple)``; ``GetValC<CapType, I>(tuple)``
     - Recursive offset accumulation helpers (non-const and const).
   * - ``GetPos<L>(level_sizes, levels, position)``;
       ``GetSparseValue(indices...)``
     - Searches sparse storage and returns a stored value or implicit zero.
   * - ``get_capability<Cap>(input)``; ``has_capability(cap)``
     - Executor capability queries. ``has_capability`` currently returns false.
   * - ``PreRun(shape, executor)``; ``PostRun(shape, executor)``
     - Empty hooks for expression execution.
   * - ``ToJITStorage()``; ``get_jit_class_name()``; ``get_jit_op_str()``
     - JIT storage and code-generation helpers, available with ``MATX_EN_JIT``.

The public nested ``PlanarComplexProxy`` supports conversion to ``T``, assignment
from complex values, and ``real()`` and ``imag()`` accessors. It implements
element access for planar complex storage.

Dynamic tensors
===============

``matx::detail::dynamic_tensor_t<T>`` has runtime rank and is created with
``make_tensor<T>()``, then initialized with ``make_tensor(tensor, shape, ...)``.
Execution requires ``CUDAJITExecutor`` and a build with ``MATX_EN_JIT=ON``;
see :ref:`executor_compatibility` for executor restrictions. It does not inherit
the dense tensor methods listed above.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Behavior
   * - ``dynamic_tensor_t(...)``
     - Default construction; allocation-space/stream configuration; or a pointer
       or ``Storage<T>`` with runtime shape and optional strides. Pointer forms
       are non-owning. Shape and stride vector lengths must agree and rank must
       not exceed ``MATX_MAX_DYNAMIC_RANK``.
   * - ``DynRank()``
     - Actual runtime rank.
   * - ``Rank()``
     - Static upper bound ``MATX_MAX_DYNAMIC_RANK`` used by expression machinery.
       Use ``DynRank()`` when inspecting the tensor's dimensions.
   * - ``Size(dim)``; ``Stride(dim)``; ``TotalSize()``
     - Runtime extent, element stride, and product of extents.
   * - ``Data()``; ``IsInitialized()``
     - View pointer and whether it is non-null.
   * - ``AllocationSpace()``; ``AllocationStream()``;
       ``SetAllocationParams(space, stream)``
     - Gets or sets allocation preferences; setting them does not allocate or
       move existing data.
   * - ``operator=(rhs)``
     - Templated lazy assignment. Compound assignment is unsupported; use explicit
       expressions such as ``(a = a + b).run(exec)``.
   * - ``str()``
     - Diagnostic name containing runtime rank and element type.
   * - ``get_capability<Cap>(input)``
     - Executor capability query.
   * - ``ToJITStorage()``; ``get_jit_class_name()``; ``get_jit_op_str()``
     - JIT integration helpers, present with ``MATX_EN_JIT``.

Dynamic tensors currently have no public ``IsContiguous()``, ``Shape()``,
``Strides()``, or element-access ``operator()`` method.

Sparse tensors
==============

``matx::experimental::sparse_tensor_t`` inherits ``tensor_impl_t`` with sparse
data storage. See :ref:`sparse_tensor_api` for factories and supported formats.
Inherited ``Size``, ``Shape``, and ``TotalSize`` describe the logical tensor;
``Data()`` points to stored values. Element access searches sparse storage,
returns a value (including implicit zero), and does not provide a writable
element reference. Dense layout/view helpers inherited from the base should
not be used to reshape sparse storage; its format defines the storage mapping.
In particular, inherited ``IsContiguous()`` describes the logical descriptor,
not the arrangement of sparse values and index buffers.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Behavior
   * - ``sparse_tensor_t(shape, values, coordinates, positions)``;
       ``~sparse_tensor_t()``
     - Constructs from value and per-level coordinate/position storage;
       destruction releases these storage containers. Prefer the sparse factories.
   * - ``SetVal(storage)``; ``SetCrd(level, storage)``;
       ``SetPos(level, storage)``
     - Replaces the corresponding storage container. Call ``SetSparseDataImpl()``
       after changing storage to refresh the inherited raw pointers.
   * - ``SetSparseDataImpl()``
     - Refreshes the base class's value, coordinate, and position pointers.
   * - ``Nse()``
     - Number of stored values, including any explicitly stored zeros.
   * - ``crdSize(level)``; ``posSize(level)``
     - Number of coordinate or position entries for a storage level.
   * - ``CRDData(level)``; ``POSData(level)``
     - Inherited raw coordinate and position pointers; available only with sparse
       data storage. Storage levels need not correspond one-to-one to dimensions.
   * - ``SetSparseData(values, coordinates, positions)``
     - Inherited low-level setter for all raw sparse pointers; does not transfer
       storage ownership.
   * - ``operator=(conversion)``
     - Lazy assignment from a supported sparse conversion transform.

Distributed tensors
===================

``matx::experimental::distributed_tensor_t<T, RANK, Distribution>`` describes a
global tensor using local dense fragments. Include
``<matx/core/distributed_tensor.h>`` and see :ref:`distributed-tensors` for
construction, executor setup, and current single-process limitations.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Behavior
   * - ``distributed_tensor_t(distribution, context, local_fragments)``
     - Constructs and validates a distributed tensor. Copy and move constructors
       are also available; fragments share their dense storage when copied.
   * - ``Rank()``; ``Size(dim)``; ``Shape()``; ``TotalSize()``
     - Global rank, extents, shape, and element count.
   * - ``ContextId()``; ``ProcessRank()``
     - Associated context identifier and this process's rank.
   * - ``DistributionDescriptor()``
     - Const reference to the distribution descriptor.
   * - ``LocalFragmentCount()``
     - Number of fragments local to this process.
   * - ``LocalView(local_fragment)``
     - Const or mutable reference to the local dense ``tensor_t``. Use this view
       for methods such as ``IsContiguous()`` and ``Data()``.
   * - ``LocalFragment(local_fragment)``
     - Const reference to a local fragment, including its view and endpoint.
   * - ``LocalFragmentForDistributionIndex(index)``
     - Const or mutable reference to the local fragment with the specified
       distribution index. Raises ``matxInvalidParameter`` if it is not local.
   * - ``operator=(expression)``; ``operator=(other)``
     - Creates a lazy distributed assignment for a distributed executor.
   * - ``MaterializeTo(output, executor)``
     - Gathers into an ordinary tensor with matching global shape and element
       type. Currently requires a single-process distributed executor.
   * - ``ExecuteTo(output, executor)``
     - Copies into a compatible distributed tensor with matching element type,
       distribution, shape, and context.
   * - ``str()``
     - Returns the diagnostic name ``"distributed_tensor"``.

Dense tensor declarations
=========================

The declarations below include constructors, the destructor, and overloads
defined directly on ``tensor_t``. Inherited methods are documented in the
tables above because the generated API excludes the ``detail`` namespace.

.. doxygenclass:: matx::tensor_t
   :members:
   :undoc-members:
