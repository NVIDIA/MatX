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

#include "matx/operators/unary_operators.h"
#include "matx/operators/binary_operators.h"

#include "matx/operators/ambgfun.h"
#include "matx/operators/at.h"
#include "matx/operators/cart2sph.h"
#include "matx/operators/collapse.h"
#include "matx/operators/concat.h"
#include "matx/operators/constval.h"
#include "matx/operators/cast.h"
#include "matx/operators/channelize_poly.h"
#include "matx/operators/chol.h"
#include "matx/operators/clone.h"
#include "matx/operators/conv.h"
#include "matx/operators/cgsolve.h"
#include "matx/operators/comma.h"
#include "matx/operators/corr.h"
#include "matx/operators/cov.h"
#include "matx/operators/cumsum.h"
#include "matx/operators/diag.h"
#include "matx/operators/dct.h"
#include "matx/operators/det.h"
#include "matx/operators/eig.h"
#include "matx/operators/einsum.h"
#include "matx/operators/find.h"
#include "matx/operators/find_idx.h"
#include "matx/operators/fft.h"
#include "matx/operators/fftshift.h"
#include "matx/operators/filter.h"
#include "matx/operators/flatten.h"
#include "matx/operators/frexp.h"
#include "matx/operators/hermitian.h"
#include "matx/operators/hist.h"
#include "matx/operators/if.h"
#include "matx/operators/ifelse.h"
#include "matx/operators/index.h"
#include "matx/operators/interleaved.h"
#include "matx/operators/isclose.h"
#include "matx/operators/inverse.h"
#include "matx/operators/kronecker.h"
#include "matx/operators/legendre.h"
#include "matx/operators/lu.h"
#include "matx/operators/matmul.h"
#include "matx/operators/matvec.h"
#include "matx/operators/norm.h"
#include "matx/operators/outer.h"
#include "matx/operators/overlap.h"
#include "matx/operators/percentile.h"
#include "matx/operators/permute.h"
#include "matx/operators/planar.h"
#include "matx/operators/polyval.h"
#include "matx/operators/pwelch.h"
#include "matx/operators/qr.h"
#include "matx/operators/r2c.h"
#include "matx/operators/remap.h"
#include "matx/operators/repmat.h"
#include "matx/operators/resample_poly.h"
#include "matx/operators/reshape.h"
#include "matx/operators/reverse.h"
#include "matx/operators/select.h"
#include "matx/operators/self.h"
#include "matx/operators/set.h"
#include "matx/operators/shift.h"
#include "matx/operators/sign.h"
#include "matx/operators/slice.h"
#include "matx/operators/sort.h"
#include "matx/operators/sph2cart.h"
#include "matx/operators/stack.h"
#include "matx/operators/stdd.h"
#include "matx/operators/svd.h"
#include "matx/operators/toeplitz.h"
#include "matx/operators/trace.h"
#include "matx/operators/transpose.h"
#include "matx/operators/unique.h"
#include "matx/operators/updownsample.h"
#include "matx/operators/var.h"

#include "matx/operators/softmax.h"
#include "matx/operators/sum.h"
#include "matx/operators/median.h"
#include "matx/operators/mean.h"
#include "matx/operators/prod.h"
#include "matx/operators/min.h"
#include "matx/operators/max.h"
#include "matx/operators/argmin.h"
#include "matx/operators/argmax.h"
#include "matx/operators/all.h"
#include "matx/operators/any.h"
