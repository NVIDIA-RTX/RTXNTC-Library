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

#define USE_COOPVEC

#include "DecompressCommon.hlsli"
#include "libntc/shaders/InferenceCoopVec.hlsli"

groupshared float s_Outputs[DECOMPRESS_CS_BLOCK_HEIGHT * DECOMPRESS_CS_BLOCK_WIDTH * NTC_MLP_OUTPUT_CHANNELS];

void DecompressPixel_CoopVec(uint2 globalIndex, uint2 threadIndex)
{
    const int2 pixelPosition = int2(globalIndex) + int2(g_Const.srcLeft, g_Const.srcTop);
    const int2 dstPosition = pixelPosition + int2(g_Const.dstLeft - g_Const.srcLeft, g_Const.dstTop - g_Const.srcTop);
    const NtcColorMipConstants colorMip = NtcUnpackColorMipConstants(g_Const.colorMip);
    const float2 colorMipSize = float2(g_Const.imageWidth, g_Const.imageHeight);

    const float2 uv = (float2(pixelPosition)+0.5) / float2(g_Const.imageWidth, g_Const.imageHeight);

    CoopVec<float16_t, NTC_MLP_INPUT_CHANNELS> networkInputs;
    NtcPrepareNetworkInputsInternal_FP16(t_Latents, s_LatentSampler,
        pixelPosition, uv, colorMip, networkInputs);

    // Evaluate the MLP layers:
    // Input layer
    CoopVec<float16_t, NTC_MLP_HIDDEN_CHANNELS> hiddenOutput1;
    NtcEvaluateLayer_CoopVec_FP8<NTC_MLP_INPUT_CHANNELS, NTC_MLP_HIDDEN_CHANNELS, true>(
        t_WeightBuffer,
        g_Const.networkWeightOffsets.x,
        g_Const.networkBiasOffsets.x,
        false, networkInputs, hiddenOutput1);

    // Hidden layer 1
    CoopVec<float16_t, NTC_MLP_HIDDEN_CHANNELS> hiddenOutput2;
    NtcEvaluateLayer_CoopVec_FP8<NTC_MLP_HIDDEN_CHANNELS, NTC_MLP_HIDDEN_CHANNELS, true>(
        t_WeightBuffer,
        g_Const.networkWeightOffsets.y,
        g_Const.networkBiasOffsets.y,
        false, hiddenOutput1, hiddenOutput2);

    // Hidden layer 2
    CoopVec<float16_t, NTC_MLP_HIDDEN_CHANNELS> hiddenOutput3;
    NtcEvaluateLayer_CoopVec_FP8<NTC_MLP_HIDDEN_CHANNELS, NTC_MLP_HIDDEN_CHANNELS, true>(
        t_WeightBuffer,
        g_Const.networkWeightOffsets.z,
        g_Const.networkBiasOffsets.z,
        true, hiddenOutput2, hiddenOutput3);

    // Output layer
    CoopVec<float, NTC_MLP_OUTPUT_CHANNELS> networkOutputs;
    NtcEvaluateOutputLayer_CoopVec_FP8<NTC_MLP_HIDDEN_CHANNELS, NTC_MLP_OUTPUT_CHANNELS>(
        t_WeightBuffer,
        g_Const.networkWeightOffsets.w,
        g_Const.networkBiasOffsets.w,
        g_Const.networkScaleOffsets.w,
        hiddenOutput3, networkOutputs);

    // Store the outputs into shared memory for efficient indexed access later.
    // Note: there is no need for a barrier before or after this store because we're using a dedicated
    // shared memory array, and each thread only reads the data it's written - nothing from other threads.
    int threadOffset = (threadIndex.y * DECOMPRESS_CS_BLOCK_WIDTH + threadIndex.x) * NTC_MLP_OUTPUT_CHANNELS;
#if __SLANG__
    networkOutputs.store(s_Outputs, threadOffset * sizeof(float));
#else
    for (int i = 0; i < NTC_MLP_OUTPUT_CHANNELS; ++i)
        s_Outputs[threadOffset + i] = networkOutputs[i];
#endif
    
    HashBasedRNG rng = HashBasedRNG::Create(pixelPosition.x + pixelPosition.y * g_Const.imageWidth, 0);

    // Exit if this pixel is outside of the specified rectangle
    if (pixelPosition.x < g_Const.srcLeft || pixelPosition.y < g_Const.srcTop ||
        pixelPosition.x >= g_Const.srcRight || pixelPosition.y >= g_Const.srcBottom)
        return;
    
    // Shuffle the output data into destination textures
    for (int outputIndex = 0; outputIndex < g_Const.numOutputs; ++outputIndex)
    {
        const NtcDecompressOutputDesc outputDesc = g_Const.outputs[outputIndex];
        
        // Read 4 channels from the network output
        float4 texelValue;
        int firstOffset = threadOffset + outputDesc.firstChannel;

        texelValue.r = s_Outputs[firstOffset];
        if (outputDesc.numChannels >= 2)
            texelValue.g = s_Outputs[++firstOffset];
        if (outputDesc.numChannels >= 3)
            texelValue.b = s_Outputs[++firstOffset];
        if (outputDesc.numChannels == 4)
            texelValue.a = s_Outputs[++firstOffset];

        // Perform color space conversion, if needed
        texelValue.rgb = NtcConvertColorSpace(texelValue.rgb, outputDesc.srcRgbColorSpace, outputDesc.dstRgbColorSpace);
        texelValue.a = NtcConvertColorSpace(texelValue.a, outputDesc.srcAlphaColorSpace, outputDesc.dstAlphaColorSpace);
        
        // Apply dithering. Making the loop conditional on ditherScale makes the shader slower,
        // so just multiply by 0 if dithering is not needed.
        float4 dither = (rng.Next4LowPrecisionFloats() - 0.5f) * outputDesc.ditherScale;
        texelValue += dither;

        // If fewer than 4 channels are requested, set the remaining ones to default values
        if (outputDesc.numChannels <= 1) texelValue.y = 0;
        if (outputDesc.numChannels <= 2) texelValue.z = 0;
        if (outputDesc.numChannels <= 3) texelValue.w = 1;

        // Write out the texel to the UAV
        u_Outputs[outputDesc.textureIndex][dstPosition] = texelValue;
    }
}

[numthreads(DECOMPRESS_CS_BLOCK_WIDTH, DECOMPRESS_CS_BLOCK_HEIGHT, 1)]
void main(uint2 globalIndex : SV_DispatchThreadID, uint2 threadIndex : SV_GroupThreadID)
{
    DecompressPixel_CoopVec(globalIndex, threadIndex);
}
