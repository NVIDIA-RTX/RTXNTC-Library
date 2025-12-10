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

#define OUTPUT_FORMAT uint2
#include "BlockCompressCommon.hlsli"

[numthreads(BLOCK_COMPRESS_CS_GROUP_WIDTH, BLOCK_COMPRESS_CS_GROUP_HEIGHT, 1)]
void main(uint2 globalIdx : SV_DispatchThreadID)
{
    if (globalIdx.x >= g_Const.widthInBlocks || globalIdx.y >= g_Const.heightInBlocks)
        return;

    // Load the input pixels.

    float4 colors[PIXELS_PER_BLOCK];
    FOREACH_PIXEL(idx)
    {
        const float4 color = t_Input[BCn_GetPixelPos(globalIdx, idx) + uint2(g_Const.srcLeft, g_Const.srcTop)];
        colors[idx] = color;
    }

    // Try different encoding schemes for the block and select one with the lowest L2 error.

    float bestError;
    uint2 bestBlock = BC1a_TransparentBlock(colors, bestError);

    {
        float error;
        uint2 block = BC1_EncodeSingleColor(colors, error);
        if (error < bestError)
        {
            bestError = error;
            bestBlock = block;
        }
    }
    
    {
        float error;
        uint2 block = BC1_CompressFast(colors, error);
        if (error < bestError)
        {
            bestError = error;
            bestBlock = block;
        }
    }

    {
        float error;
        uint2 block = BC1a_CompressFast(colors, error);
        if (error < bestError)
        {
            bestError = error;
            bestBlock = block;
        }
    }
    
    // Write the output.

    u_Output[globalIdx + uint2(g_Const.dstOffsetX, g_Const.dstOffsetY)] = bestBlock;
}