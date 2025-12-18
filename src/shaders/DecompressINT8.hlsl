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

#include "DecompressCommon.hlsli"

groupshared float4 s_Scale[MAX_OUTPUT_SIZE / 4];
groupshared int4 s_Bias[MAX_OUTPUT_SIZE / 4];
groupshared uint s_MatrixB[MAX_OUTPUT_SIZE][MAX_INPUT_SIZE / 4];

template<int IN, int OUT, bool OUTPUT_LAYER>
void EvaluateLayerINT8_SharedMem(
    int weightOffset,
    int biasOffset,
    int scaleOffset,
    uint inputArray[IN / 4],
    out uint outputArray[OUTPUT_LAYER ? OUT : OUT / 4],
    int2 threadIndex)
{
    GroupMemoryBarrierWithGroupSync();

    // Preload the bias values into shared memory.
    // Note: this 'if' assumes that there are enough threads in the group to load all bias values in one pass.
    // If that ever changes, use a loop like one used for the weights below.
    const int linearThreadIndex = threadIndex.x + threadIndex.y * DECOMPRESS_CS_BLOCK_WIDTH;
    if (linearThreadIndex < OUT/4)
    {
        float4 scale = t_WeightBuffer.Load<float4>(scaleOffset + linearThreadIndex * sizeof(float4));
        int4 bias = t_WeightBuffer.Load<int4>(biasOffset + linearThreadIndex * sizeof(int4));

        s_Scale[linearThreadIndex] = scale;
        s_Bias[linearThreadIndex] = bias;
    }

    // Preload the weights into shared memory.
    // The weights form a matrix with IN rows and OUT columns, stored in a row-major layout
    // (i.e. elements of a row are continuous).
    // Each element is an int8, so 4 of these are packed into a uint32.
    int preloadIndex = linearThreadIndex;
    while (preloadIndex < (IN * OUT) / 4)
    {
        int row = (preloadIndex) % (IN/4);
        int col = (preloadIndex) / (IN/4);
        
        s_MatrixB[col][row] = t_WeightBuffer.Load(weightOffset + preloadIndex * 4);

        preloadIndex += DECOMPRESS_CS_BLOCK_WIDTH * DECOMPRESS_CS_BLOCK_HEIGHT;
    }

    GroupMemoryBarrierWithGroupSync();

    // Note: not unrolling the outer loop for performance and smaller shaders.
    [loop]
    for (uint col = 0; col < OUT; col += 4)
    {
        int4 acc = s_Bias[col / 4];
        
        [unroll]
        for (uint row4 = 0; row4 < IN / 4; row4++)
        {
            acc.x = dot4add_i8packed(inputArray[row4], s_MatrixB[col + 0][row4], acc.x);
            acc.y = dot4add_i8packed(inputArray[row4], s_MatrixB[col + 1][row4], acc.y);
            acc.z = dot4add_i8packed(inputArray[row4], s_MatrixB[col + 2][row4], acc.z);
            acc.w = dot4add_i8packed(inputArray[row4], s_MatrixB[col + 3][row4], acc.w);
        }

        float4 scales = s_Scale[col / 4];
        float4 results = float4(acc) * scales;

        if (OUTPUT_LAYER)
        {
            outputArray[col + 0] = asuint(results.x);
            outputArray[col + 1] = asuint(results.y);
            outputArray[col + 2] = asuint(results.z);
            outputArray[col + 3] = asuint(results.w);
        }
        else
        {
            results = NtcHGELUClamp_ForwardFloat(results);
            int4 iresults = NtcHGELUClamp_QuantizeFloat(results);

            outputArray[col / 4] = NtcPackInt8x4(iresults);
        }
    }
}

void DecompressPixel(uint2 globalIndex, uint2 threadIndex)
{
    const int2 pixelPosition = int2(globalIndex) + int2(g_Const.srcLeft, g_Const.srcTop);
    const int2 dstPosition = pixelPosition + int2(g_Const.dstLeft - g_Const.srcLeft, g_Const.dstTop - g_Const.srcTop);
    const NtcColorMipConstants colorMip = NtcUnpackColorMipConstants(g_Const.colorMip);
    const float2 colorMipSize = float2(g_Const.imageWidth, g_Const.imageHeight);
    
    const float2 uv = (float2(pixelPosition) + 0.5) / colorMipSize;

    uint networkInputs[NTC_MLP_INPUT_CHANNELS / 4];
    NtcPrepareNetworkInputsInternal(t_Latents, s_LatentSampler,
        pixelPosition, uv, colorMip, networkInputs);

    // Evaluate the MLP layers:
    
    // Input layer
    uint hiddenOutput0[NTC_MLP_HIDDEN0_CHANNELS / 4];
    EvaluateLayerINT8_SharedMem<NTC_MLP_INPUT_CHANNELS, NTC_MLP_HIDDEN0_CHANNELS, false>(
        g_Const.networkWeightOffsets.x,
        g_Const.networkBiasOffsets.x,
        g_Const.networkScaleOffsets.x,
        networkInputs, hiddenOutput0, threadIndex);

    // Hidden layer 1
    uint hiddenOutput1[NTC_MLP_HIDDEN1_CHANNELS / 4];
    EvaluateLayerINT8_SharedMem<NTC_MLP_HIDDEN0_CHANNELS, NTC_MLP_HIDDEN1_CHANNELS, false>(
        g_Const.networkWeightOffsets.y,
        g_Const.networkBiasOffsets.y,
        g_Const.networkScaleOffsets.y,
        hiddenOutput0, hiddenOutput1, threadIndex);

#if NTC_MLP_LAYERS == 4
    // Hidden layer 2
    uint hiddenOutput2[NTC_MLP_HIDDEN2_CHANNELS / 4];
    EvaluateLayerINT8_SharedMem<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_HIDDEN2_CHANNELS, false>(
        g_Const.networkWeightOffsets.z,
        g_Const.networkBiasOffsets.z,
        g_Const.networkScaleOffsets.z,
        hiddenOutput1, hiddenOutput2, threadIndex);

    // Output layer
    uint networkOutputs[NTC_MLP_OUTPUT_CHANNELS];
    EvaluateLayerINT8_SharedMem<NTC_MLP_HIDDEN2_CHANNELS, NTC_MLP_OUTPUT_CHANNELS, true>(
        g_Const.networkWeightOffsets.w,
        g_Const.networkBiasOffsets.w,
        g_Const.networkScaleOffsets.w,
        hiddenOutput2, networkOutputs, threadIndex);
#else
    // Output layer
    uint networkOutputs[NTC_MLP_OUTPUT_CHANNELS];
    EvaluateLayerINT8_SharedMem<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_OUTPUT_CHANNELS, true>(
        g_Const.networkWeightOffsets.z,
        g_Const.networkBiasOffsets.z,
        g_Const.networkScaleOffsets.z,
        hiddenOutput1, networkOutputs, threadIndex);
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
        
        // Read 4 channels from the shared buffer
        float4 texelValue;
        [unroll]
        for (int ch = 0; ch < 4; ++ch)
        {
            int srcChannel = min(outputDesc.firstChannel + ch, NTC_MLP_OUTPUT_CHANNELS - 1);
            texelValue[ch] = asfloat(networkOutputs[srcChannel]);
        }

        // Perform color space conversion, if needed
        texelValue.rgb = NtcConvertColorSpace(texelValue.rgb, outputDesc.srcRgbColorSpace, outputDesc.dstRgbColorSpace);
        texelValue.a = NtcConvertColorSpace(texelValue.a, outputDesc.srcAlphaColorSpace, outputDesc.dstAlphaColorSpace);
        
        // Apply dithering
        float4 dither = (rng.Next4LowPrecisionFloats() - 0.5f) * outputDesc.ditherScale;
        texelValue += dither;

        // If fewer than 4 channels are requested, set the remaining ones to default values
        if (outputDesc.numChannels <= 1) texelValue.y = 0;
        if (outputDesc.numChannels <= 2) texelValue.z = 0;
        if (outputDesc.numChannels <= 3) texelValue.w = 1;

        // Apply quantization, e.g. manual rounding to UNORM8, because DX doesn't require
        // correct rounding for UAV writes
        if (outputDesc.quantizationScale != 0.f)
        {
            texelValue = round(texelValue * outputDesc.invQuantizationScale) * outputDesc.quantizationScale;
        }

        // Write out the texel to the UAV
        u_Outputs[outputDesc.textureIndex][dstPosition] = texelValue;
    }
}

[numthreads(DECOMPRESS_CS_BLOCK_WIDTH, DECOMPRESS_CS_BLOCK_HEIGHT, 1)]
void main(uint2 globalIndex : SV_DispatchThreadID, uint2 threadIndex : SV_GroupThreadID)
{
    DecompressPixel(globalIndex, threadIndex);
}
