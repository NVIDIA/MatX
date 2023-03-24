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

/**
 * @brief Custom operator for calculating detection positions
 * 
 * @tparam O output tensor type
 * @tparam I1 Input power tensor type
 * @tparam I2 Input burst averages tensor type
 * @tparam I3 Input norm tensor type
 * @tparam I4 Input probability of false alarm tensor type
 */
template <class O, class I1, class I2, class I3, class I4>
class calcDets : public BaseOp<calcDets<O, I1, I2, I3, I4>> {
private:
  O out_;
  I1 xpow_;
  I2 ba_;
  I3 norm_;
  I4 pfa_;

public:

  /**
   * @brief Construct a new calcDets object
   * 
  * @param out output tensor
  * @param xpow Input power tensor
  * @param ba Input burst averages tensor
  * @param norm Input norm tensor
  * @param pfa Input probability of false alarm tensor
  */
  calcDets(O out, I1 xpow, I2 ba, I3 norm, I4 pfa)
      : out_(out), xpow_(xpow), ba_(ba), norm_(norm), pfa_(pfa)
  {
  }

  /**
   * @brief Get detection value at position
   * 
   * @param idz Z position
   * @param idy Y position
   * @param idx X position
   * @return Detection value 
   */
  __device__ inline void operator()(index_t idz, index_t idy, index_t idx)
  {
    typename I1::type xpow = xpow_(idz, idy, idx);
    typename I2::type ba = ba_(idz, idy, idx);
    typename I2::type norm = norm_(idz, idy, idx);
    typename I2::type alpha = norm * (std::pow(pfa_, -1.0 / norm) - 1);
    out_(idz, idy, idx) = (xpow > alpha * ba) ? 1 : 0;
  }

  /**
   * @brief Get size of detection tensor across dimension
   * 
   * @param i dimension
   * @return Size of dimension 
   */
  __host__ __device__ inline index_t Size(uint32_t i) const
  {
    return out_.Size(i);
  }

  /**
   * @brief Return rank of detection tensor
   * 
   * @return Rank of tensor
   */
  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return O::Rank();
  }
};

/**
 * @brief Radar Pipeline object
 * 
 * @tparam ComplexType type of complex value
 */
template <typename ComplexType = cuda::std::complex<float>>
class RadarPipeline {
public:
  RadarPipeline() = delete;
  ~RadarPipeline()
  {

  }

  /**
   * @brief Construct a new Radar Pipeline object
   * 
   * @param _numPulses Number of pulses
   * @param _numSamples Number of samples per pulse
   * @param _wfLen Waveform length
   * @param _numChannels Number of channels
   * @param _stream CUDA stream
   */
  RadarPipeline(const index_t _numPulses, const index_t _numSamples,
                index_t _wfLen, index_t _numChannels, cudaStream_t _stream)
      : numPulses(_numPulses), numSamples(_numSamples), waveformLength(_wfLen),
        numChannels(_numChannels), stream(_stream)
  {
    numSamplesRnd = 1;
    while (numSamplesRnd < numSamples) {
      numSamplesRnd *= 2;
    }

    numPulsesRnd = 1;
    while (numPulsesRnd <= numPulses) {
      numPulsesRnd *= 2;
    }

    numCompressedSamples = numSamples - waveformLength + 1;

    // waveform is of length waveform data but we pad to numSamples for fft
    make_tensor(waveformView, {numSamplesRnd});
    make_tensor(norms);
    make_tensor(inputView,
        {numChannels, numPulses, numSamplesRnd});
    make_tensor(tpcView,
        {numChannels, numPulsesRnd, numCompressedSamples});
    make_tensor(cancelMask, {3});
    make_tensor(normT, 
        {numChannels, numPulsesRnd + cfarMaskY - 1,
         numCompressedSamples + cfarMaskX - 1});
    make_tensor(ba, 
        {numChannels, numPulsesRnd + cfarMaskY - 1,
         numCompressedSamples + cfarMaskX - 1});
    make_tensor(dets, 
        {numChannels, numPulsesRnd, numCompressedSamples});
    make_tensor(xPow, 
        {numChannels, numPulsesRnd, numCompressedSamples});

    cudaMemset(waveformView.Data(), 0, numSamplesRnd * sizeof(ComplexType));
    cudaMemset(inputView.Data(), 0,
               inputView.TotalSize() * sizeof(ComplexType));
    cudaMemset(tpcView.Data(), 0, tpcView.TotalSize() * sizeof(ComplexType));

    cancelMask.SetVals({1, -2, 1});

    make_tensor(cfarMaskView, 
        {cfarMaskY, cfarMaskX});
    // Mask for cfar detection
    // G == guard, R == reference, C == CUT
    // mask = [
    //    R R R R R ;
    //    R R R R R ;
    //    R R R R R ;
    //    R R R R R ;
    //    R R R R R ;
    //    R G G G R ;
    //    R G C G R ;
    //    R G G G R ;
    //    R R R R R ;
    //    R R R R R ;
    //    R R R R R ;
    //    R R R R R ;
    //    R R R R R ];
    //  }
    cfarMaskView.SetVals({{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});

    // Pre-process CFAR convolution
    conv2d(normT, ones({numChannels, numPulsesRnd, numCompressedSamples}),
           cfarMaskView, matxConvCorrMode_t::MATX_C_MODE_FULL, stream);

    cancelMask.PrefetchDevice(stream);
    ba.PrefetchDevice(stream);
    normT.PrefetchDevice(stream);
    cfarMaskView.PrefetchDevice(stream);
    dets.PrefetchDevice(stream);
    waveformView.PrefetchDevice(stream);
    norms.PrefetchDevice(stream);
    inputView.PrefetchDevice(stream);
    tpcView.PrefetchDevice(stream);
    xPow.PrefetchDevice(stream);
  }

  /**
   * @brief Stage 1 - Pulse compression - convolution via FFTs
   * 
   * Pulse compression achieves high range resolution by applying intra-pulse
   * modulation during transmit followed by applying a matched filter after
   * reception. References:
   *    Richards, M. A., Scheer, J. A., Holm, W. A., "Principles of Modern
   *    Radar: Basic Principles", SciTech Publishing, Inc., 2010.  Chapter 20.
   *    Also, http://en.wikipedia.org/wiki/Pulse_compression
   */
  void PulseCompression()
  {
    // reshape waveform to be waveformLength
    auto waveformPart = waveformView.Slice({0}, {waveformLength});
    auto waveformT =
        waveformView.template Clone<3>({numChannels, numPulses, matxKeepDim});

    auto waveformFull = waveformView.Slice({0}, {numSamplesRnd});

    auto x = inputView;

    // create waveform (assuming waveform is the same for every pulse)
    // this allows us to precompute waveform in frequency domain
    // Apply a Hamming window to the waveform to suppress sidelobes. Other
    // windows could be used as well (e.g., Taylor windows). Ultimately, it is
    // just an element-wise weighting by a pre-computed window function.
    (waveformPart = waveformPart * hamming<0>({waveformLength})).run(stream);

    // compute L2 norm
    sum(norms, norm(waveformPart), stream);
    (norms = sqrt(norms)).run(stream);

    (waveformPart = waveformPart / norms).run(stream);
    fft(waveformFull, waveformPart, 0, stream);
    (waveformFull = conj(waveformFull)).run(stream);

    fft(x, x, 0, stream);
    (x = x * waveformT).run(stream);
    ifft(x, x, 0, stream);
  }


  /**
   * @brief Stage 2 - Three-pulse canceller - 1D convolution
   * 
   * The three-pulse canceller is a simple high-pass filter designed to suppress
   * background, or "clutter", such as the ground and other non-moving objects.
   * The three-pulse canceller is a pair of two-pulse cancellers implemented in
   * a single stage. A two-pulse canceller just computes the difference between
   * two subsequent pulses at each range bin. Thus, the two pulse canceller is
   * equivalent to convolution in the pulse dimension with [1 -1] and the
   * three-pulse canceller is convolution in the pulse dimension with [1 -2 1]
   * ([1 -2 1] is just the convolution of [1 -1] with [1 -1], so it is
   * effectively a sequence of two two-pulse cancellers).
   * References:
   *   Richards, M. A., Scheer, J. A., Holm, W. A., "Principles of Modern Radar:
   *   Basic Principles",
   *       SciTech Publishing, Inc., 2010.  Section 17.4.
   */
  void ThreePulseCanceller()
  {
    auto x = inputView.Permute({0, 2, 1}).Slice(
        {0, 0, 0}, {numChannels, numCompressedSamples, numPulses});
    auto xo = tpcView.Permute({0, 2, 1}).Slice(
        {0, 0, 0}, {numChannels, numCompressedSamples, numPulses});
    conv1d(xo, x, cancelMask, matxConvCorrMode_t::MATX_C_MODE_SAME, stream);
  }

  /**
   * @brief Stage 3 - Doppler Processing - FFTs in pulse
   * 
   * Doppler processing converts the range-pulse data to range-Doppler data via
   * an FFT in the Doppler dimension. Explicit spectral analysis can then be
   * performed, such as the detector that will follow as stage 4.
   * References:
   *   Richards, M. A., Scheer, J. A., Holm, W. A., "Principles of Modern Radar:
   *   Basic Principles",
   *       SciTech Publishing, Inc., 2010.  Section 17.5.
   *
   * Apply a window in pulse to suppress sidelobes. Using a Hamming window for
   * simplicity, but others would work. repmat().
   */
  void DopplerProcessing()
  {
    const index_t cpulses = numPulses - (cancelMask.Size(0) - 1);

    auto xc =
        tpcView.Slice({0, 0, 0}, {numChannels, cpulses, numCompressedSamples});

    auto xf = tpcView.Permute({0, 2, 1});

    (xc = xc * hamming<1>({numChannels, numPulses - (cancelMask.Size(0) - 1),
                          numCompressedSamples}))
        .run(stream);
    fft(xf, xf, 0, stream);
  }

  /**
   * @brief Stage 4 - Constant False Alarm Rate (CFAR) Detector - averaging or median
   * 
   * filter CFAR detectors in general are designed to provide constant false
   * alarm rates by dynamically adjusting detection thresholds based on certain
   * statistical assumptions and interference estimates made from the data.
   * References:
   *   Richards, M. A., Scheer, J. A., Holm, W. A., "Principles of Modern Radar:
   *   Basic Principles",
   *       SciTech Publishing, Inc., 2010.  Section 16.4.
   *   Richards, M. A., "Fundamentals of Radar Signal Processing", McGraw-Hill,
   *   2005.
   *       Chapter 7. alpha below corresponds to equation (7.17)
   *   Also, http://en.wikipedia.org/wiki/Constant_false_alarm_rate

   * CFAR works by using a training window to average cells "near" a cell
   * under test (CUT) to estimate the background power for that cell. It is
   * an assumption that the average of the nearby cells represents a
   * reasonable background estimate. In general, there are guard cells (G)
   * and reference cells (R) around the CUT. The guard cells prevent
   * contributions of a potential target in the CUT from corrupting the
   * background estimate. More reference cells are preferred to better
   * estimate the background average. As implemented below, the CUT and
   * guard cells form a hole within the training window, but CA-CFAR is
   * largely just an averaging filter otherwise with a threshold check
   * at each pixel after applying the filter.
   * Currently, the window below is defined statically because it is then
   * easy to visualize, but more typically the number of guard and
   * reference cells would be given as input and the window would be
   * constructed; we could update to such an approach, but I'm keeping
   * it simple for now.

   * We apply CFAR to the power of X; X is still complex until this point
   * Xpow = abs(X).^2;
   */
  void CFARDetections()
  {
    (xPow = norm(tpcView)).run(stream);

    // Estimate the background average power in each cell
    // background_averages = conv2(Xpow, mask, 'same') ./ norm;
    conv2d(ba, xPow, cfarMaskView, matxConvCorrMode_t::MATX_C_MODE_FULL,
           stream);

    // Computing number of cells contributing to each cell.
    // This can be done with a convolution of the cfarMask with
    // ones.
    // norm = conv2(ones(size(X)), mask, 'same');
    auto normTrim = normT.Slice({0, cfarMaskY / 2, cfarMaskX / 2},
                                 {numChannels, numPulsesRnd + cfarMaskY / 2,
                                  numCompressedSamples + cfarMaskX / 2});

    auto baTrim = ba.Slice({0, cfarMaskY / 2, cfarMaskX / 2},
                            {numChannels, numPulsesRnd + cfarMaskY / 2,
                             numCompressedSamples + cfarMaskX / 2});
    (baTrim = baTrim / normTrim).run(stream);

    // The scalar alpha is used as a multiplier on the background averages
    // to achieve a constant false alarm rate (under certain assumptions);
    // it is based upon the desired probability of false alarm (Pfa) and
    // number of reference cells used to estimate the background for the
    // CUT. For the purposes of computation, it can be assumed as a given
    // constant, although it does vary at the edges due to the different
    // training windows.
    // Declare a detection if the power exceeds the background estimate
    // times alpha for a particular cell.
    // dets(find(Xpow > alpha.*background_averages)) = 1;

    // These 2 branches are functionally equivalent.  A custom op is more
    // efficient as it can avoid repeated loads.
#if 0
    IFELSE(xPow > normTrim*(pow(pfa, -1.0f/normTrim) - 1.0f)*baTrim,
                dets = 1, dets = 0).run(stream);
#else
    calcDets(dets, xPow, baTrim, normTrim, pfa).run(stream);
#endif
  }

  /**
   * @brief Get the Input View object
   * 
   * @return tensor_t view 
   */
  auto GetInputView() { return inputView; }

  /**
   * @brief Get waveform view
   * 
   * @return tensor_t view 
   */
  auto GetwaveformView() { return waveformView; }

  /**
   * @brief Get TPC view
   * 
   * @return tensor_t view 
   */
  auto GetTPCView() { return tpcView; }

  /**
   * @brief Get the Detections object
   * 
   * @return tensor_t view 
   */
  auto GetDetections() { return dets; }

  /**
   * @brief Get the Background Averages object
   * 
   * @return tensor_t view 
   */
  auto GetBackgroundAverages() { return ba; }

  /**
   * @brief Get norm object
   * 
   * @return tensor_t view 
   */
  auto GetnormT() { return normT; }

private:
  index_t numPulses;
  index_t numSamples;
  index_t waveformLength;
  index_t numSamplesRnd;
  index_t numPulsesRnd;
  index_t numCompressedSamples;
  index_t numChannels;
  const index_t cfarMaskX = 13;
  const index_t cfarMaskY = 5;

  static const constexpr float pfa = 1e-5f;

  tensor_t<typename ComplexType::value_type, 3> normT;
  tensor_t<typename ComplexType::value_type, 3> ba;
  tensor_t<int, 3> dets;
  tensor_t<typename ComplexType::value_type, 1> cancelMask;
  tensor_t<typename ComplexType::value_type, 3> xPow;
  tensor_t<ComplexType, 1> waveformView;
  tensor_t<typename ComplexType::value_type, 0> norms;
  tensor_t<ComplexType, 3> inputView;
  tensor_t<ComplexType, 3> tpcView;
  tensor_t<typename ComplexType::value_type, 2> cfarMaskView;

  cudaStream_t stream;
};
