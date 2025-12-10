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

#define OUTPUT_FORMAT uint4
#include "BlockCompressCommon.hlsli"

[numthreads(BLOCK_COMPRESS_CS_GROUP_WIDTH, BLOCK_COMPRESS_CS_GROUP_HEIGHT, 1)]
void main(uint2 globalIdx : SV_DispatchThreadID)
{
    if (globalIdx.x >= g_Const.widthInBlocks || globalIdx.y >= g_Const.heightInBlocks)
        return;

    // Load the input pixels.
    
    float4 colors3[PIXELS_PER_BLOCK];
    float alphas[PIXELS_PER_BLOCK];
    FOREACH_PIXEL(idx)
    {
        const float4 color = t_Input[BCn_GetPixelPos(globalIdx, idx) + uint2(g_Const.srcLeft, g_Const.srcTop)];
        colors3[idx] = float4(color.rgb, 1.0);
        alphas[idx] = color.a;
    }

    // Compress color as BC1 and compress alpha as BC4.

    float bestError;
    uint2 bestColorBlock = BC1_EncodeSingleColor(colors3, bestError);
    
    {
        float error;
        uint2 colorBlock = BC1_CompressFast(colors3, error);
        if (error < bestError)
        {
            bestError = error;
            bestColorBlock = colorBlock;
        }
    }

    uint2 compressedAlpha = BC4_CompressFast(alphas);
    
    // Write the output.

    u_Output[globalIdx + uint2(g_Const.dstOffsetX, g_Const.dstOffsetY)] = uint4(compressedAlpha, bestColorBlock);
}