////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once


namespace matx
{
  namespace detail {
    template <typename Generator1D, int Dim, typename ShapeType>
    class matxGenerator1D_t : public BaseOp<matxGenerator1D_t<Generator1D, Dim, ShapeType>>{
      public:
        // dummy type to signal this is a matxop
        using matxop = bool;
        using value_type = typename Generator1D::value_type;
        // Rank of the output shape, which can be any rank. The generator varies only along Dim and repeats
        // across the other dimensions.
        static constexpr int RANK = cuda::std::tuple_size<cuda::std::decay_t<ShapeType>>::value;

        __MATX_INLINE__ std::string str() const { return "gen1d"; }

        template <typename S>
          matxGenerator1D_t(S &&s, Generator1D f) : f_(f), s_(std::forward<S>(s)) {}

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            if constexpr (RANK != 1) { // Vectorization not supported yet
              return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, detail::ElementsPerThread::ONE};
            }
            else {
              return detail::capability_attributes<Cap>::default_value;
            }
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
            return f_.template get_capability<Cap>(in);
          }
          else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            return std::format("{}<{}>", get_jit_class_name(), f_.template get_capability<Cap>(in));
#else
            return f_.template get_capability<Cap>(in);
#endif
          }
          else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
            detail::register_jit_class(in, *this);
#endif
            return f_.template get_capability<Cap>(in);
          }
          else {
            return detail::capability_attributes<Cap>::default_value;
          }
        }

#ifdef MATX_EN_JIT
        // The JIT class wraps the generator's class so the generator is indexed along Dim and the wrapper
        // reports the full rank and shape
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<Generator1D>> f_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(f_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITGenerator1D_dim{}", Dim) + detail::jit_shape_suffix(*this);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          cuda::std::array<index_t, RANK> sizes;
          for (int i = 0; i < RANK; i++) {
            sizes[i] = Size(i);
          }

          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename Gen> struct {} {{\n"
                "  using value_type = typename Gen::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int Dim_ = {};\n"
                "  constexpr static int Rank_ = {};\n"
                "  constexpr static cuda::std::array<index_t, Rank_> sizes_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<Gen>> f_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const\n"
                "  {{\n"
                "    const cuda::std::array<index_t, Rank_> idx{{static_cast<index_t>(indices)...}};\n"
                "    return f_.template operator()<CapType>(idx[Dim_]);\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return sizes_[dim]; }}\n"
                "}};\n",
                func_name, Dim, RANK, detail::array_to_string(sizes))
          );
        }
#endif

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const {
          return f_.template operator()<CapType>(pp_get<Dim>(indices...));
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const
        {
          return *(s_.begin() + dim);
        }
        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }

      private:
        Generator1D f_;
        // Stored by value: an lvalue shape argument deduces ShapeType as a reference, which would dangle
        // once the caller's shape goes out of scope
        cuda::std::decay_t<ShapeType> s_;
    };
  }
} // end namespace matx
