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
#include "matx.h"
#include <memory>
#include <stdint.h>

using namespace matx;

class MVDRBeamformer {
public:
  using complex = cuda::std::complex<float>;
  /**
   * Constructs and MVDRBeamformer object, and allocates all handles needed to
   * beamform.
   */
  MVDRBeamformer(uint32_t num_beams, uint32_t num_el, uint32_t data_len,
                 uint32_t snap_len)
      : num_beams_(num_beams), num_el_(num_el), data_len_(data_len),
        snap_len_(snap_len)
  {
    // Create data objects and views
    vView = new tensor_t<complex, 2>({num_el, num_beams});
    vhView = new tensor_t<complex, 2>({num_beams, num_el});
    cbfView = new tensor_t<complex, 2>({num_beams, data_len});
    inVecView = new tensor_t<complex, 2>({num_el, data_len});

    ivsView = new tensor_t<complex, 2>({num_el, snap_len});
    ivshView = new tensor_t<complex, 2>({snap_len, num_el});
    covMatView = new tensor_t<complex, 2>({num_el, num_el});
    invCovMatView = new tensor_t<complex, 2>({num_el, num_el});
    abfBView = new tensor_t<complex, 2>({num_el, num_beams});
    abfAView = new tensor_t<complex, 2>({num_beams, num_beams});
    abfAInvView = new tensor_t<complex, 2>({num_beams, num_beams});
    abfWeightsView = new tensor_t<complex, 2>({num_el, num_beams});

    cbf_mm = new matxMatMulHandle_t(*cbfView, *vhView, *inVecView);
    cov_mat_mm = new matxMatMulHandle_t(*covMatView, *ivsView, *ivshView);
  }

  /**
   * Prefetch all data onto the device
   */
  void Prefetch(cudaStream_t stream)
  {
    covMatView->PrefetchDevice(stream);
    vView->PrefetchDevice(stream);
    vhView->PrefetchDevice(stream);
    cbfView->PrefetchDevice(stream);
    inVecView->PrefetchDevice(stream);
    ivsView->PrefetchDevice(stream);
    ivshView->PrefetchDevice(stream);
    invCovMatView->PrefetchDevice(stream);
    abfBView->PrefetchDevice(stream);
    abfAView->PrefetchDevice(stream);
    abfAInvView->PrefetchDevice(stream);
  }

  /**
   *  Run the entire beamformer
   */
  void Run(cudaStream_t stream)
  {
    (*vhView = hermitianT(*vView)).run(stream);

    cbf_mm->Exec(*cbfView, *vhView, *inVecView, stream);

    copy(*ivsView, inVecView->Slice({0, 0}, {matxEnd, snap_len_}), stream);

    (*ivshView = hermitianT(*ivsView)).run(stream);

    cov_mat_mm->Exec(*covMatView, *ivsView, *ivshView, stream);

    (*covMatView = (*covMatView * (1.0f / static_cast<float>(snap_len_))) +
                   eye<complex>({num_el_, num_el_}) * load_coeff_)
        .run(stream);
    inv(*invCovMatView, *covMatView, stream);

    // Find A and B to solve xA=B. Matlab uses A/B to solve for x, which is the
    // same as x = BA^-1
    matmul(*abfBView, *invCovMatView, *vView, stream);
    matmul(*abfAView, *vhView, *abfBView, stream);

    inv(*abfAInvView, *abfAView, stream);
    matmul(*abfWeightsView, *abfBView, *abfAInvView, stream);
  }

  auto GetInVec() { return *inVecView; }
  auto GetCBFView() { return *cbfView; }
  auto GetV() { return *vView; }
  auto GetCovMatView() { return *covMatView; }
  auto GetCovMatInvView() { return *invCovMatView; }

private:
  uint32_t num_beams_;
  uint32_t num_el_;
  uint32_t data_len_;
  uint32_t snap_len_;
  cuda::std::complex<float> load_coeff_ = {0.1f, 0.f};

  tensor_t<complex, 2> *vView;
  tensor_t<complex, 2> *vhView;
  tensor_t<complex, 2> *cbfView;
  tensor_t<complex, 2> *inVecView;
  tensor_t<complex, 2> *ivsView;
  tensor_t<complex, 2> *ivshView;
  tensor_t<complex, 2> *covMatView;
  tensor_t<complex, 2> *invCovMatView;
  tensor_t<complex, 2> *abfWeightsView;

  tensor_t<complex, 2> *abfAView;
  tensor_t<complex, 2> *abfBView;
  tensor_t<complex, 2> *abfAInvView;

  matxMatMulHandle_t<complex, complex, complex, 2> *cbf_mm;
  matxMatMulHandle_t<complex, complex, complex, 2> *cov_mat_mm;
};