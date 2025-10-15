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

#include "libntc/shaders/ImageDifferenceConstants.h"
#include "libntc/shaders/Bindings.h"
#include "BindingHelpers.hlsli"

#ifdef __cplusplus
static const NtcImageDifferenceConstants g_Const;
#else
NTC_DECLARE_CBUFFER(ConstantBuffer<NtcImageDifferenceConstants> g_Const, NTC_BINDING_IMAGE_DIFF_CONSTANT_BUFFER, 0);
#endif
NTC_DECLARE_SRV(Texture2D<float4> t_Input1,   NTC_BINDING_IMAGE_DIFF_INPUT_TEXTURE_A, 0);
NTC_DECLARE_SRV(Texture2D<float4> t_Input2,   NTC_BINDING_IMAGE_DIFF_INPUT_TEXTURE_B, 0);
NTC_DECLARE_UAV(RWByteAddressBuffer u_Output, NTC_BINDING_IMAGE_DIFF_OUTPUT_BUFFER, 0);

// Storage for error reduction - enough for the worst case of 4-lane SIMD.
groupshared float4 s_Error[IMAGE_DIFFERENCE_CS_BLOCK_WIDTH * IMAGE_DIFFERENCE_CS_BLOCK_HEIGHT / 4];

// Simulate a 64-bit uint atomic addition
void InterlockedAddUint64(uint offset, uint high, uint low)
{
    uint oldlow, oldhigh;
    u_Output.InterlockedAdd(offset, low, oldlow);
    uint newlow = oldlow + low;
    uint carry = newlow < oldlow ? 1 : 0;
    high += carry;
    if (high != 0)
        u_Output.InterlockedAdd(offset + 4, high, oldhigh);
}

[numthreads(IMAGE_DIFFERENCE_CS_BLOCK_WIDTH * IMAGE_DIFFERENCE_CS_BLOCK_HEIGHT, 1, 1)]
void main(uint2 groupIdx : SV_GroupID, uint linearIdx: SV_GroupThreadID)
{
    // Note: the group size is (W*H, 1, 1) so that we can get a well-defined mapping from SV_GroupThreadID to waves.
    // In a 2D group, the mapping is undefined, and the reduction code below breaks on Intel ARC.

    // Map the 1D thread index into 2D and recalculate globalIdx.
    uint2 threadIdx = uint2(linearIdx / IMAGE_DIFFERENCE_CS_BLOCK_WIDTH, linearIdx % IMAGE_DIFFERENCE_CS_BLOCK_WIDTH);
    uint2 globalIdx = groupIdx * uint2(IMAGE_DIFFERENCE_CS_BLOCK_WIDTH, IMAGE_DIFFERENCE_CS_BLOCK_HEIGHT) + threadIdx;
    uint2 const basePos = globalIdx * uint2(IMAGE_DIFFERENCE_CS_PIXELS_PER_THREAD_X, IMAGE_DIFFERENCE_CS_PIXELS_PER_THREAD_Y);
    uint2 const offset = uint2(g_Const.left, g_Const.top);
    
    // Calculate and accumulate MSE for all pixels processed by this thread.
    // We process more than pixel per thread to amortize the cost of further reductions.
    float4 mse = 0;
    for (int dy = 0; dy < IMAGE_DIFFERENCE_CS_PIXELS_PER_THREAD_Y; ++dy)
    {
        for (int dx = 0; dx < IMAGE_DIFFERENCE_CS_PIXELS_PER_THREAD_X; ++dx)
        {
            uint2 pos = basePos + uint2(dx, dy);
            if (pos.x < uint(g_Const.width) && pos.y < uint(g_Const.height))
            {
                float4 t1 = t_Input1[pos + offset];
                float4 t2 = t_Input2[pos + offset];
                float4 diff = g_Const.useMSLE ? log(abs(t1) + 1) - log(abs(t2) + 1) : t1 - t2;

                // If alpha threshold is enabled, and both images are transparent in this pixel, ignore the color difference.
                if (g_Const.useAlphaThreshold && t1.a < g_Const.alphaThreshold && t2.a < g_Const.alphaThreshold)
                    diff.rgb = 0;

                mse += diff * diff;
            }
        }
    }

    // Parallel reduction of the error values from the entire block into thread 0,
    // using wave intrinsics. Generic algorithm that covers all wavefront sizes.
    uint const waveIdx = linearIdx / WaveGetLaneCount();
    uint groupSize = IMAGE_DIFFERENCE_CS_BLOCK_WIDTH * IMAGE_DIFFERENCE_CS_BLOCK_HEIGHT;
    while(true)
    {
        if (linearIdx < groupSize)
        {
            mse = WaveActiveSum(mse);

            if (WaveIsFirstLane())
                s_Error[waveIdx] = mse;
        }

        groupSize /= WaveGetLaneCount();

        if (groupSize <= 1)
            break;

        GroupMemoryBarrierWithGroupSync();
        
        if (linearIdx < groupSize)
            mse = s_Error[linearIdx];
        else
            mse = 0;

        GroupMemoryBarrierWithGroupSync();
    }
    
    if (linearIdx == 0)
    {
        // Normalize the MSE value
        mse /= (g_Const.width * g_Const.height);

        // Convert to 16.48 fixed point format.
        // Note: the format is chosen to support a wide range of possible error values and atomic accumulation.
        // For example, the smallest nonzero squared error for an 8-bit image is 2^-16.
        // We're normalizing the error by dividing the sum by the total pixel count in the image.
        // If we want to preserve a single bit error in a 16Kx16K image (max supported by graphics APIs),
        // we need 16+(14*2) = 44 bits after the decimal point. Call it 48 for cases when image dimensions
        // are not nice round powers of 2.
        // On the high end, if we're processing FP16 numbers, max error is (65504*2) = 131008. Squared that is 
        // 1.7*10^10, or 0x3FF001000, which needs 34 bits. That implies we need something like 48.48 format
        // to cover all possible ranges, but let's just assume that extreme errors don't happen very often.
        mse *= 0x1p16f;
        uint4 high = uint4(floor(mse));
        uint4 low = uint4(floor(frac(mse) * 0x1p32f));

        // Atomically add to the output buffer
        InterlockedAddUint64(g_Const.outputOffset + 0, high.x, low.x);
        InterlockedAddUint64(g_Const.outputOffset + 8, high.y, low.y);
        InterlockedAddUint64(g_Const.outputOffset + 16, high.z, low.z);
        InterlockedAddUint64(g_Const.outputOffset + 24, high.w, low.w);
    }
}