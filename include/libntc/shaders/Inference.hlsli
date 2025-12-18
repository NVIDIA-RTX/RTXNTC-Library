/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NTC_INFERENCE_HLSLI
#define NTC_INFERENCE_HLSLI

#include "InferenceConstants.h"
#include "ColorSpaces.hlsli"

// Helper macros used to declare templated functions with different t-parameter counts in Slang and HLSL.
#if __SLANG__
#define NTC_TEMPLATE_FN_1(ReturnType, FnName, ArgType1, ArgName1) \
    ReturnType FnName <let ArgName1: ArgType1>
#define NTC_TEMPLATE_FN_2(ReturnType, FnName, ArgType1, ArgName1, ArgType2, ArgName2) \
    ReturnType FnName <let ArgName1: ArgType1, let ArgName2: ArgType2>
#define NTC_TEMPLATE_FN_3(ReturnType, FnName, ArgType1, ArgName1, ArgType2, ArgName2, ArgType3, ArgName3) \
    ReturnType FnName <let ArgName1: ArgType1, let ArgName2: ArgType2, let ArgName3: ArgType3>
#else
#define NTC_TEMPLATE_FN_1(ReturnType, FnName, ArgType1, ArgName1) \
    template<ArgType1 ArgName1> ReturnType FnName
#define NTC_TEMPLATE_FN_2(ReturnType, FnName, ArgType1, ArgName1, ArgType2, ArgName2) \
    template<ArgType1 ArgName1, ArgType2 ArgName2> ReturnType FnName
#define NTC_TEMPLATE_FN_3(ReturnType, FnName, ArgType1, ArgName1, ArgType2, ArgName2, ArgType3, ArgName3) \
    template<ArgType1 ArgName1, ArgType2 ArgName2, ArgType3 ArgName3> ReturnType FnName
#endif

// Activation function parameters
#define NTC_HGELU_MAXVALUE 3
#define NTC_HGELU_INVSTEP 80
#define NTC_HGELU_BIAS -113

uint NtcPackInt8x4(int4 vec)
{    
    return uint(vec.x & 0xff) 
        | (uint(vec.y & 0xff) << 8) 
        | (uint(vec.z & 0xff) << 16) 
        | (uint(vec.w) << 24);
}

// Converts the int4 packed version of ColorMipConstants into a struct
NtcColorMipConstants NtcUnpackColorMipConstants(int4 i)
{
    NtcColorMipConstants result;
    result.neuralMip = i.x;
    result.positionLod = asfloat(i.y);
    result.positionScale = asfloat(i.z);
    result.pad = i.w;
    return result;
}

static const float c_InputScale = 127.5f; // Inputs are in the [-1, 1] range, scale matches tin::InputQuant

bool NtcSampleLatentGrid(
    Texture2DArray latentTexture,
    SamplerState latentSampler,
    float2 uv,
    int neuralLod,
    int featureOffset,
    inout uint outputArray[NTC_MLP_INPUT_CHANNELS / 4])
{
    int width, height, arraySize;
    latentTexture.GetDimensions(width, height, arraySize);

    [unroll]
    for (int layerIndex = 0; layerIndex < NTC_MLP_FEATURES / NTC_FEATURES_PER_LAYER; ++layerIndex)
    {
        const bool mask = (layerIndex == 0) || (layerIndex < arraySize);
        
        float4 sampledValue = latentTexture.SampleLevel(latentSampler, float3(uv, layerIndex), neuralLod);
        sampledValue = sampledValue.bgra; // The texture format is BGRA4, unswizzle that
        sampledValue = mask ? sampledValue * (2.f * c_InputScale) - c_InputScale - 0.5f : 0.f;
        
        const uint packedValues = NtcPackInt8x4(int4(floor(sampledValue)));
        outputArray[featureOffset / 4 + layerIndex] = packedValues;
    }

    return true;
}

float4 NtcEvaluatePositionalEncoding(float2 posf)
{
    float4 result;

    result.x = frac(posf.x) * 2 - 1;
    result.y = frac(posf.y) * 2 - 1;
    result.z = frac(posf.x + 0.25f) * 2 - 1;
    result.w = frac(posf.y + 0.25f) * 2 - 1;

    return result;
}

void NtcEncodeSamplePosition(
    float2 posf, float lod, int featureOffset,
    inout uint outputArray[NTC_MLP_INPUT_CHANNELS / 4])
{
    int idx = featureOffset / 4;
    
    [unroll]
    for (int wave = 0; wave < NTC_MLP_POS_ENC_WAVES; ++wave)
    {
        float4 enc = NtcEvaluatePositionalEncoding(posf);
        uint packedPositionalEncoding = NtcPackInt8x4(int4(enc * c_InputScale));
        outputArray[idx] = packedPositionalEncoding;
        ++idx;
        posf *= 2.f;
    }

    int iLod = int(lod * c_InputScale);
    uint packedLod = NtcPackInt8x4(int4(iLod.xx, 0, 0));
    outputArray[idx] = packedLod;
}

// HGELU activation function with clamping, forward evaluation
float4 NtcHGELUClamp_ForwardFloat(float4 x)
{
    return min(x, float(NTC_HGELU_MAXVALUE)) * saturate(mad(x, 0.333333f, 0.5f));
}

int4 NtcHGELUClamp_QuantizeFloat(float4 x)
{
    return int4(round(x * float(NTC_HGELU_INVSTEP)) + float(NTC_HGELU_BIAS));
}

NTC_TEMPLATE_FN_3(void, NtcEvaluateLayerINT8, int, IN, int, OUT, bool, OUTPUT_LAYER)
    (ByteAddressBuffer weightBuffer,
    int weightOffset,
    int biasOffset,
    int scaleOffset,
    uint inputArray[IN / 4],
    out uint outputArray[OUTPUT_LAYER ? OUT : OUT / 4]
)
{
    // See the comment block in the beginning of WeightLayout.cpp for the weight layouts

    // Note: not unrolling the outer loop for performance and smaller shaders.
    [loop]
    for (uint col = 0; col < OUT; col += 4)
    {
        int4 acc = weightBuffer.Load<int4>(biasOffset + col * 4);
        
        [unroll]
        for (uint row4 = 0; row4 < IN / 4; row4++)
        {
            const uint weights0 = weightBuffer.Load(weightOffset + (col + 0) * IN + row4 * sizeof(uint));
            const uint weights1 = weightBuffer.Load(weightOffset + (col + 1) * IN + row4 * sizeof(uint));
            const uint weights2 = weightBuffer.Load(weightOffset + (col + 2) * IN + row4 * sizeof(uint));
            const uint weights3 = weightBuffer.Load(weightOffset + (col + 3) * IN + row4 * sizeof(uint));
            
            acc.x = dot4add_i8packed(inputArray[row4], weights0, acc.x);
            acc.y = dot4add_i8packed(inputArray[row4], weights1, acc.y);
            acc.z = dot4add_i8packed(inputArray[row4], weights2, acc.z);
            acc.w = dot4add_i8packed(inputArray[row4], weights3, acc.w);
        }
        
        float4 scales = weightBuffer.Load<float4>(scaleOffset + col * 4);
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

int2 NtcGetTextureDimensions(NtcTextureSetConstants desc, int mipLevel)
{
    return max(int2(desc.imageWidth, desc.imageHeight) >> mipLevel, 1);
}

int NtcGetTextureMipLevels(NtcTextureSetConstants desc)
{
    return desc.imageMips;
}

uint NtcGetChannelMask(int firstChannel, int numChannels = 1)
{
    return ((1u << numChannels) - 1u) << firstChannel;
}

// Returns the bit mask of channels in the texture set that have some texture data.
// If a channel's bit in this mask is 0, then its contents are undefined.
// Use GetChannelMask(first, num) to get the expected mask for a given set of channels.
uint NtcGetValidChannelMask(NtcTextureSetConstants desc)
{
    return desc.validChannelMask;
}

bool NtcTextureSetHasChannels(NtcTextureSetConstants desc, int firstChannel, int numChannels = 1)
{
    uint mask = NtcGetChannelMask(firstChannel, numChannels);
    return (NtcGetValidChannelMask(desc) & mask) == mask;
}

bool NtcPrepareNetworkInputsInternal(
    Texture2DArray latentTexture,
    SamplerState latentSampler,
    int2 texel,
    float2 uv,
    const NtcColorMipConstants colorMip,
    out uint networkInputs[NTC_MLP_INPUT_CHANNELS / 4])
{
    // Zero init the array
    [unroll]
    for (int i = 0; i < NTC_MLP_INPUT_CHANNELS / 4; ++i)
        networkInputs[i] = 0;

    if (colorMip.neuralMip < 0)
        return false;

    // Sample the latent grids
    if (!NtcSampleLatentGrid(latentTexture, latentSampler, uv, colorMip.neuralMip, 0, networkInputs))
        return false;

    if (!NtcSampleLatentGrid(latentTexture, latentSampler, uv, colorMip.neuralMip + 1, NTC_MLP_FEATURES, networkInputs))
        return false;

    // Encode the sample position
    NtcEncodeSamplePosition(float2(texel) * colorMip.positionScale,
        colorMip.positionLod, NTC_MLP_FEATURES * 2, networkInputs);

    return true;
}

bool NtcPrepareNetworkInputs(
    NtcTextureSetConstants desc,
    Texture2DArray latentTexture,
    SamplerState latentSampler,
    int2 texel,
    int mipLevel,
    out uint networkInputs[NTC_MLP_INPUT_CHANNELS / 4])
{
    const int2 imageSize = NtcGetTextureDimensions(desc, mipLevel);
    const float2 uv = (float2(texel) + 0.5) / imageSize;

    // Zero init the array - in some cases, OUTPUT_SIZE is rounded up from the actual used size.
    [unroll]
    for (int i = 0; i < NTC_MLP_INPUT_CHANNELS / 4; ++i)
        networkInputs[i] = 0;

    const NtcColorMipConstants colorMip = NtcUnpackColorMipConstants(desc.colorMips[mipLevel]);

    return NtcPrepareNetworkInputsInternal(latentTexture, latentSampler, texel, uv, colorMip, networkInputs);
}

float NtcConvertChannelToLinearColorSpace(NtcTextureSetConstants desc, int channel, float storedValue)
{
    int colorSpace = (desc.channelColorSpaces >> (channel * 2)) & 3;
    
    switch (colorSpace)
    {
        case NtcColorSpace_sRGB:
            return NtcSrgbColorSpace::Decode(storedValue);
        case NtcColorSpace_HLG:
            return NtcHybridLogGammaColorSpace::Decode(storedValue);
        default:
            return storedValue;
    }
}

// NtcSampleTextureSet - this is the main NTC function for applications.
// Use like NtcSampleTextureSet(Constants, LatentsBuffer, ...)
// Returns true if the mip level is valid; out-of-bounds texel positions are clamped.
bool NtcSampleTextureSet(
    NtcTextureSetConstants desc,
    Texture2DArray latentTexture,
    SamplerState latentSampler,
    ByteAddressBuffer weightsBuffer,
    uint weightsOffset, // Offset of the weight chunk in weightsBuffer if packing multiple textures together
    int2 texel,
    int mipLevel,
    bool convertToLinearColorSpace,
    inout float outputs[NTC_MLP_OUTPUT_CHANNELS])
{
    uint networkInputs[NTC_MLP_INPUT_CHANNELS / 4];
    if (!NtcPrepareNetworkInputs(desc, latentTexture, latentSampler, texel, mipLevel, networkInputs))
        return false;

    // Evaluate the MLP layers:
    // Input layer
    uint hiddenOutput0[NTC_MLP_HIDDEN0_CHANNELS / 4];
    NtcEvaluateLayerINT8<NTC_MLP_INPUT_CHANNELS, NTC_MLP_HIDDEN0_CHANNELS, false>(
        weightsBuffer,
        weightsOffset + desc.networkWeightOffsets.x,
        weightsOffset + desc.networkBiasOffsets.x,
        weightsOffset + desc.networkScaleOffsets.x,
        networkInputs, hiddenOutput0);

    // Hidden layer 1
    uint hiddenOutput1[NTC_MLP_HIDDEN1_CHANNELS / 4];
    NtcEvaluateLayerINT8<NTC_MLP_HIDDEN0_CHANNELS, NTC_MLP_HIDDEN1_CHANNELS, false>(
        weightsBuffer,
        weightsOffset + desc.networkWeightOffsets.y,
        weightsOffset + desc.networkBiasOffsets.y,
        weightsOffset + desc.networkScaleOffsets.y,
        hiddenOutput0, hiddenOutput1);

#if NTC_MLP_LAYERS == 4
    // Hidden layer 2
    uint hiddenOutput2[NTC_MLP_HIDDEN2_CHANNELS / 4];
    NtcEvaluateLayerINT8<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_HIDDEN2_CHANNELS, false>(
        weightsBuffer,
        weightsOffset + desc.networkWeightOffsets.z,
        weightsOffset + desc.networkBiasOffsets.z,
        weightsOffset + desc.networkScaleOffsets.z,
        hiddenOutput1, hiddenOutput2);

    // Output layer
    uint networkOutputs[NTC_MLP_OUTPUT_CHANNELS];
    NtcEvaluateLayerINT8<NTC_MLP_HIDDEN2_CHANNELS, NTC_MLP_OUTPUT_CHANNELS, true>(
        weightsBuffer,
        weightsOffset + desc.networkWeightOffsets.w,
        weightsOffset + desc.networkBiasOffsets.w,
        weightsOffset + desc.networkScaleOffsets.w,
        hiddenOutput2, networkOutputs);
#else
    // Output layer
    uint networkOutputs[NTC_MLP_OUTPUT_CHANNELS];
    NtcEvaluateLayerINT8<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_OUTPUT_CHANNELS, true>(
        weightsBuffer,
        weightsOffset + desc.networkWeightOffsets.z,
        weightsOffset + desc.networkBiasOffsets.z,
        weightsOffset + desc.networkScaleOffsets.z,
        hiddenOutput1, networkOutputs);
#endif

    [unroll]
    for (int ch = 0; ch < NTC_MLP_OUTPUT_CHANNELS; ++ch)
    {
        outputs[ch] = asfloat(networkOutputs[ch]);

        if (convertToLinearColorSpace)
        {
            outputs[ch] = NtcConvertChannelToLinearColorSpace(desc, ch, outputs[ch]);
        }
    }
    
    return true;
}

#endif
