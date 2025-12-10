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

float16_t2 NtcUintToHalf2(uint u)
{
    return asfloat16(uint16_t2(uint16_t(u), uint16_t(u >> 16)));
}

uint NtcHalf2ToUint(float16_t2 h)
{
    uint16_t2 u = asuint16(h);
    return uint(u.x) | (uint(u.y) << 16);
}

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
        sampledValue = mask ? sampledValue * (2.f * c_InputScale) - c_InputScale : 0.f;

        const uint packedValues = NtcPackInt8x4(int4(sampledValue));
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

struct NtcHGELUParams
{
    float maxval;
    float invStep;
    float bias;
};

NtcHGELUParams NtcGetHGELUParams()
{
    const float minval = -3.0 / 16.0;
    const float maxval = 3.0;

    const int bins = 256;
    const float step = (maxval - minval) / float(bins - 1);
    const float invStep = 1.0 / step;
    const int qmax = int(maxval / step);
    const int qmin = qmax - bins + 1;
    const int bias = -(bins / 2) - qmin;

    NtcHGELUParams params;
    params.maxval = maxval;
    params.invStep = invStep;
    params.bias = bias;
    return params;
}

// HGELU activation function with clamping, forward evaluation
float16_t4 NtcHGELUClamp_ForwardHalf(float16_t4 x)
{
    const NtcHGELUParams params = NtcGetHGELUParams();
    return min(x, float16_t(params.maxval)) * clamp(float16_t(1/3.f) * x + 0.5h, 0.h, 1.h);
}

float4 NtcHGELUClamp_ForwardFloat(float4 x)
{
    const NtcHGELUParams params = NtcGetHGELUParams();
    return min(x, params.maxval) * clamp((1/3.f) * x + 0.5f, 0.f, 1.f);
}

int4 NtcHGELUClamp_QuantizeHalf(float16_t4 x)
{
    const NtcHGELUParams params = NtcGetHGELUParams();
    return int4(round(x * float16_t(params.invStep) + float16_t(params.bias)));
}

int4 NtcHGELUClamp_QuantizeFloat(float4 x)
{
    const NtcHGELUParams params = NtcGetHGELUParams();
    return int4(round(x * params.invStep + params.bias));
}

NTC_TEMPLATE_FN_3(void, NtcEvaluateLayerINT8, int, IN, int, OUT, bool, OUTPUT_LAYER)
    (ByteAddressBuffer weightBuffer,
    int weightOffset,
    int biasOffset,
    int scaleOffset,
    bool activation,
    uint inputArray[IN / 4],
    out uint outputArray[OUTPUT_LAYER ? OUT / 2 : OUT / 4]
)
{
    // See the comment block in the beginning of WeightLayout.cpp for the weight layouts

    // Note: not unrolling the outer loop.
    // If we do, DXC/SPIR-V crashes.
    // DXC/DXIL compiles the unrolled loop successfully, but then creating a pipeline with it takes seconds,
    // and the resulting code works slower than a regular loop.
    for (uint c = 0; c < OUT; c += 4)
    {
        int4 biases = weightBuffer.Load<int4>(biasOffset + c * 4);
        int acc0 = biases.x;
        int acc1 = biases.y;
        int acc2 = biases.z;
        int acc3 = biases.w;
        
        [unroll]
        for (uint k = 0; k < IN / 4; k++)
        {
            const uint weights0 = weightBuffer.Load(weightOffset + (c + 0) * IN + k * 4);
            const uint weights1 = weightBuffer.Load(weightOffset + (c + 1) * IN + k * 4);
            const uint weights2 = weightBuffer.Load(weightOffset + (c + 2) * IN + k * 4);
            const uint weights3 = weightBuffer.Load(weightOffset + (c + 3) * IN + k * 4);
            
            acc0 = dot4add_i8packed(inputArray[k], weights0, acc0);
            acc1 = dot4add_i8packed(inputArray[k], weights1, acc1);
            acc2 = dot4add_i8packed(inputArray[k], weights2, acc2);
            acc3 = dot4add_i8packed(inputArray[k], weights3, acc3);
        }
        
        float4 results = float4(acc0, acc1, acc2, acc3);
        float4 scales = weightBuffer.Load<float4>(scaleOffset + c * 4);

        float16_t4 hresults = float16_t4(results * scales);
        
        if (activation)
        {
            hresults = NtcHGELUClamp_ForwardHalf(hresults);
        }

        if (OUTPUT_LAYER)
        {
            outputArray[c / 2 + 0] = NtcHalf2ToUint(hresults.xy);
            outputArray[c / 2 + 1] = NtcHalf2ToUint(hresults.zw);
        }
        else
        {
            int4 iresults = NtcHGELUClamp_QuantizeHalf(hresults);

            outputArray[c / 4] = NtcPackInt8x4(iresults);
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
        true, networkInputs, hiddenOutput0);

    // Hidden layer 1
    uint hiddenOutput1[NTC_MLP_HIDDEN1_CHANNELS / 4];
    NtcEvaluateLayerINT8<NTC_MLP_HIDDEN0_CHANNELS, NTC_MLP_HIDDEN1_CHANNELS, false>(
        weightsBuffer,
        weightsOffset + desc.networkWeightOffsets.y,
        weightsOffset + desc.networkBiasOffsets.y,
        weightsOffset + desc.networkScaleOffsets.y,
        true, hiddenOutput0, hiddenOutput1);

#if NTC_MLP_LAYERS == 4
    // Hidden layer 2
    uint hiddenOutput2[NTC_MLP_HIDDEN2_CHANNELS / 4];
    NtcEvaluateLayerINT8<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_HIDDEN2_CHANNELS, false>(
        weightsBuffer,
        weightsOffset + desc.networkWeightOffsets.z,
        weightsOffset + desc.networkBiasOffsets.z,
        weightsOffset + desc.networkScaleOffsets.z,
        true, hiddenOutput1, hiddenOutput2);

    // Output layer
    uint networkOutputs[NTC_MLP_OUTPUT_CHANNELS / 2];
    NtcEvaluateLayerINT8<NTC_MLP_HIDDEN2_CHANNELS, NTC_MLP_OUTPUT_CHANNELS, true>(
        weightsBuffer,
        weightsOffset + desc.networkWeightOffsets.w,
        weightsOffset + desc.networkBiasOffsets.w,
        weightsOffset + desc.networkScaleOffsets.w,
        false, hiddenOutput2, networkOutputs);
#else
    // Output layer
    uint networkOutputs[NTC_MLP_OUTPUT_CHANNELS / 2];
    NtcEvaluateLayerINT8<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_OUTPUT_CHANNELS, true>(
        weightsBuffer,
        weightsOffset + desc.networkWeightOffsets.z,
        weightsOffset + desc.networkBiasOffsets.z,
        weightsOffset + desc.networkScaleOffsets.z,
        false, hiddenOutput1, networkOutputs);
#endif

    [unroll]
    for (int ch = 0; ch < NTC_MLP_OUTPUT_CHANNELS / 2; ++ch)
    {
        uint twoCh = networkOutputs[ch];
        int ch0 = ch * 2 + 0;
        int ch1 = ch * 2 + 1;
        outputs[ch0] = asfloat16(uint16_t(twoCh));
        outputs[ch1] = asfloat16(uint16_t(twoCh >> 16));

        if (convertToLinearColorSpace)
        {
            outputs[ch0] = NtcConvertChannelToLinearColorSpace(desc, ch0, outputs[ch0]);
            outputs[ch1] = NtcConvertChannelToLinearColorSpace(desc, ch1, outputs[ch1]);
        }
    }
    
    return true;
}

#endif
