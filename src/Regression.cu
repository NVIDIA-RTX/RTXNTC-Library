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
#include "CudaUtils.h"
#include <cassert>
#include <cooperative_groups.h>

namespace ntc::cuda
{

__global__ void RegressionKernelFast(RegressionKernelParams const params);
__global__ void RegressionKernelStable(RegressionKernelParams const params);

void Regression(
    size_t   pixelsPerBatch,
    bool  stableTraining,
    RegressionKernelParams const& params)
{
    auto threadBlockSize = dim3(TB_SIZE_X, TB_SIZE_Y, 1);
    uint32_t pixelsPerThreadBlock = threadBlockSize.x * threadBlockSize.y * Y_ITERS;
    auto gridSize = dim3(DivRoundUp(pixelsPerBatch, pixelsPerThreadBlock), 1, 1);

    if (stableTraining)
        RegressionKernelStable<<<gridSize, threadBlockSize>>>(params);
    else
        RegressionKernelFast<<<gridSize, threadBlockSize>>>(params);
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