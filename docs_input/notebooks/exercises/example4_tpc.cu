/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include "simple_radar_pipeline.h"

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
    index_t numChannels = 16;
    index_t numPulses = 128;
    index_t numSamples = 9000;
    index_t waveformLength = 1000;

    // cuda stream to place work in
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // create some events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    auto radar = RadarPipeline(numPulses, numSamples, waveformLength, numChannels, stream);
    randomGenerator_t<float> rfloat(radar.GetInputView().TotalSize(), 0);
    auto t3fu = rfloat.GetTensorView<3>({radar.GetInputView().TotalSize()}, NORMAL);
    (*(radar.GetInputView()) = t3fu).run();
    radar.ThreePulseCanceller();

    printf("x input:\n");
    radar.GetInputView().Slice<1>({0, 0, 0}, {matxSliceDim, matxSliceDim, 16}).Print();
    printf("Convolution output:\n");
    radar.GetTPCView()->Slice<1>({0,0,0}, {matxSliceDim, matxSliceDim, 10}).Print();     
    cudaStreamDestroy(stream);

    return 0;
}
