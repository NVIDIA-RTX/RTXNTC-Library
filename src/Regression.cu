/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "Regression.h"
#include "MlpDesc.h"
#include <cassert>
#include <cooperative_groups.h>

#include "kernels/kernel_permutations.h"

namespace ntc::cuda
{

void Regression(
    size_t   pixelsPerBatch,
    bool  stableTraining,
    MlpDesc const& kernelVersion,
    RegressionKernelParams const& params)
{
    auto threadBlockSize = dim3(TB_SIZE_X, TB_SIZE_Y, 1);
    uint32_t pixelsPerThreadBlock = threadBlockSize.x * threadBlockSize.y * Y_ITERS;
    auto gridSize = dim3((pixelsPerBatch + pixelsPerThreadBlock - 1) / pixelsPerThreadBlock, 1, 1);

    switch(kernelVersion.networkVersion)
    {
        case NTC_NETWORK_SMALL:
            if (stableTraining)
                RegressionKernel_small_stable1<<<gridSize, threadBlockSize>>>(params);
            else
                RegressionKernel_small_stable0<<<gridSize, threadBlockSize>>>(params);
            break;
        case NTC_NETWORK_MEDIUM:
            if (stableTraining)
                RegressionKernel_medium_stable1<<<gridSize, threadBlockSize>>>(params);
            else
                RegressionKernel_medium_stable0<<<gridSize, threadBlockSize>>>(params);
            break;
        case NTC_NETWORK_LARGE:
            if (stableTraining)
                RegressionKernel_large_stable1<<<gridSize, threadBlockSize>>>(params);
            else
                RegressionKernel_large_stable0<<<gridSize, threadBlockSize>>>(params);
            break;
        case NTC_NETWORK_XLARGE:
            if (stableTraining)
                RegressionKernel_xlarge_stable1<<<gridSize, threadBlockSize>>>(params);
            else
                RegressionKernel_xlarge_stable0<<<gridSize, threadBlockSize>>>(params);
            break;
    }
}

void Inference(
    MlpDesc const& kernelVersion,
    InferenceKernelParams const& params)
{
    auto gridSize = dim3(
        std::ceil(params.referenceWidth / float(TILE_SIZE_X)),
        std::ceil(params.referenceHeight / float(TILE_SIZE_Y)),
        1);
    auto threadBlockSize = dim3(TB_SIZE_X, TB_SIZE_Y, 1);

    switch(kernelVersion.networkVersion)
    {
        case NTC_NETWORK_SMALL:
            InferenceKernel_small<<<gridSize, threadBlockSize>>>(params);
            break;
        case NTC_NETWORK_MEDIUM:
            InferenceKernel_medium<<<gridSize, threadBlockSize>>>(params);
            break;
        case NTC_NETWORK_LARGE:
            InferenceKernel_large<<<gridSize, threadBlockSize>>>(params);
            break;
        case NTC_NETWORK_XLARGE:
            InferenceKernel_xlarge<<<gridSize, threadBlockSize>>>(params);
            break;
    }
}

__constant__ MipInfo g_MipInfo[NTC_MAX_MIPS];

void SetMipInfos(const MipInfo* data, int count)
{
    cudaMemcpyToSymbol(g_MipInfo, data, sizeof(MipInfo) * count);
}

__constant__ ChannelInfo g_ChannelInfo[NTC_MAX_CHANNELS];

void SetChannelInfos(const ChannelInfo* data, int count)
{
    cudaMemcpyToSymbol(g_ChannelInfo, data, sizeof(ChannelInfo) * count);
}

}