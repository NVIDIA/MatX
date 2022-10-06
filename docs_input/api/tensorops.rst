Tensor Operators
################

This page lists all built-in operators to MatX. Operators are categorized by unary (one input), binary (two inputs), and advanced. 
The advanced operators perform more complex operations on the input instead of a straight 1:1 mapping, such as an fftshift. 
Some advanced operators require that the input and output tensors be different to prevent a race condition in reading and writing
the values. 

Unary Operators
----------------
.. doxygenfunction:: sqrt(Op t)
.. doxygenfunction:: exp(Op t)
.. doxygenfunction:: expj(Op t)
.. doxygenfunction:: log10(Op t) 
.. doxygenfunction:: log2(Op t)
.. doxygenfunction:: log(Op t)
.. doxygenfunction:: loge(Op t)  
.. doxygenfunction:: conj(Op t)
.. doxygenfunction:: norm(Op t)
.. doxygenfunction:: abs(Op t)
.. doxygenfunction:: sin(Op t)
.. doxygenfunction:: cos(Op t)
.. doxygenfunction:: tan(Op t)       
.. doxygenfunction:: asin(Op t)
.. doxygenfunction:: acos(Op t)
.. doxygenfunction:: atan(Op t)
.. doxygenfunction:: sinh(Op t)
.. doxygenfunction:: cosh(Op t)
.. doxygenfunction:: tanh(Op t)   
.. doxygenfunction:: asinh(Op t)
.. doxygenfunction:: acosh(Op t)
.. doxygenfunction:: atanh(Op t)  
.. doxygenfunction:: angle(Op t)
.. doxygenfunction:: atan2(Op t)
.. doxygenfunction:: floor(Op t)
.. doxygenfunction:: ceil(Op t)
.. doxygenfunction:: round(Op t)  
.. doxygenfunction:: operator!(Op t)  
.. doxygenfunction:: operator-(Op t)  

Binary Operators
----------------
.. doxygenfunction:: operator+(Op t, Op t2)
.. doxygenfunction:: operator-(Op t, Op t2)  
.. doxygenfunction:: operator*(Op t, Op t2) 
.. doxygenfunction:: mul(Op t, Op t2)
.. doxygenfunction:: operator/(Op t, Op t2)  
.. doxygenfunction:: operator%(Op t, Op t2)  
.. doxygenfunction:: fmod(Op t, Op t2)
.. doxygenfunction:: pow(Op t, Op t2) 
.. doxygenfunction:: max(Op t, Op t2)
.. doxygenfunction:: min(Op t, Op t2)
.. doxygenfunction:: operator<(Op t, Op t2)
.. doxygenfunction:: operator>(Op t, Op t2) 
.. doxygenfunction:: operator<=(Op t, Op t2) 
.. doxygenfunction:: operator>=(Op t, Op t2)
.. doxygenfunction:: operator==(Op t, Op t2)
.. doxygenfunction:: operator!=(Op t, Op t2)  
.. doxygenfunction:: operator&&(Op t, Op t2)
.. doxygenfunction:: operator||(Op t, Op t2)  

Casting Operators
------------------

.. doxygenfunction:: matx::as_type 
.. doxygenfunction:: matx::as_int8
.. doxygenfunction:: matx::as_uint8
.. doxygenfunction:: matx::as_int16
.. doxygenfunction:: matx::as_uint16
.. doxygenfunction:: matx::as_int32
.. doxygenfunction:: matx::as_uint32
.. doxygenfunction:: matx::as_float
.. doxygenfunction:: matx::as_double

Advanced Operators
------------------

.. doxygenclass:: matx::IF 
.. doxygenclass:: matx::IFELSE
.. doxygenfunction:: reverse(Op t)
.. doxygenfunction:: reverse(Op_type t)
.. doxygenfunction:: shift(OpT op, ShiftOpT s)
.. doxygenfunction:: fftshift1D
.. doxygenfunction:: fftshift2D    
.. doxygenfunction:: repmat(T1 t, index_t reps)    
.. doxygenfunction:: repmat(T1 t, const index_t(&reps)[])
.. doxygenfunction:: repmat(T1 t, const index_t *reps)
.. doxygenfunction:: kron
.. doxygenfunction:: hermitianT
.. doxygenfunction:: r2cop
.. doxygenfunction:: flatten
.. doxygenfunction:: remap(Op t, Ind idx)
.. doxygenfunction:: remap(Op t, Ind idx, Inds... inds)
.. doxygenfunction:: rcollapse
.. doxygenfunction:: lcollapse
.. doxygenfunction:: clone
.. doxygenfunction:: slice( const T_wStrideT op, const typename T_wStrideT::shape_type (&starts)[T_wStrideT::Rank()], const typename T_wStrideT::shape_type (&ends)[T_wStrideT::Rank()], const typename T_wStrideT::stride_type (&strides)[T_wStrideT::Rank()])                 
.. doxygenfunction:: slice( const T_wShapeT op, const typename T_wShapeT::shape_type (&starts)[T_wShapeT::Rank()], const typename T_wShapeT::shape_type (&ends)[T_wShapeT::Rank()])
.. doxygenfunction:: slice( const T_wStridet_2 op, const typename T_wStridet_2::shape_type (&starts)[T_wStridet_2::Rank()], const typename T_wStridet_2::shape_type (&ends)[T_wStridet_2::Rank()], const typename T_wStridet_2::stride_type (&strides)[T_wStridet_2::Rank()])
.. doxygenfunction:: slice( const T_wShapet_2 op, const typename T_wShapet_2::shape_type (&starts)[T_wShapet_2::Rank()], const typename T_wShapet_2::shape_type (&ends)[T_wShapet_2::Rank()])
  
.. doxygenfunction:: permute(detail::tensor_impl_t<T, Rank> &out, const detail::tensor_impl_t<T, Rank> &in, const std::initializer_list<uint32_t> &dims, const cudaStream_t stream)
.. doxygenfunction:: permute(const T op, const int32_t (&dims)[T::Rank()])  