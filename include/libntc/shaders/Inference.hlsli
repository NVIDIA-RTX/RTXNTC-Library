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

// Define this macro before including the header to set the DP4a support flag for compatibility with older GPUs
#ifndef NTC_USE_DP4A
    #define NTC_USE_DP4A 1
#endif

// Define this macro before including the header to set the FP16 support flag for compatibility with older GPUs
#ifndef NTC_USE_FLOAT16
    #define NTC_USE_FLOAT16 1
#endif

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

// The NtcNetworkParams structure is used to derive the MLP geometry from network version
#if __SLANG__
struct NtcNetworkParams<let _NETWORK_VERSION: int>
#else
template<int _NETWORK_VERSION> struct NtcNetworkParams
#endif
{
    static const int INPUT_CHANNELS = 
        (_NETWORK_VERSION == NTC_NETWORK_SMALL) ? NTC_MLP_INPUT_CHANNELS_SMALL :
        (_NETWORK_VERSION == NTC_NETWORK_MEDIUM) ? NTC_MLP_INPUT_CHANNELS_MEDIUM :
        (_NETWORK_VERSION == NTC_NETWORK_LARGE) ? NTC_MLP_INPUT_CHANNELS_LARGE :
        (_NETWORK_VERSION == NTC_NETWORK_XLARGE) ? NTC_MLP_INPUT_CHANNELS_XLARGE :
        0; // Unsupported value

    static const int HR_FEATURES = 
        (_NETWORK_VERSION == NTC_NETWORK_SMALL) ? NTC_MLP_HR_FEATURES_SMALL :
        (_NETWORK_VERSION == NTC_NETWORK_MEDIUM) ? NTC_MLP_HR_FEATURES_MEDIUM :
        (_NETWORK_VERSION == NTC_NETWORK_LARGE) ? NTC_MLP_HR_FEATURES_LARGE :
        (_NETWORK_VERSION == NTC_NETWORK_XLARGE) ? NTC_MLP_HR_FEATURES_XLARGE :
        0; // Unsupported value

    static const int LR_FEATURES = NTC_MLP_LR_FEATURES;

    static const int SAMPLED_FEATURES_HR = HR_FEATURES * 4;
    static const int SAMPLED_FEATURES_LR = LR_FEATURES;

    static const int HIDDEN_LAYER_CHANNELS = NTC_MLP_HIDDEN_CHANNELS;

    static const int OUTPUT_CHANNELS = NTC_MLP_OUTPUT_CHANNELS;
};

// The pack_clamp_s8 intrinsic should map well to an I2IP instruction on NV GPUs, but using it causes major slowdowns on Intel.
#define USE_PACKING_INTRINSICS 0

float16_t2 NtcUintToHalf2(uint u)
{
    return asfloat16(uint16_t2(uint16_t(u), uint16_t(u >> 16)));
}

uint NtcHalf2ToUint(float16_t2 h)
{
    uint16_t2 u = asuint16(h);
    return uint(u.x) | (uint(u.y) << 16);
}

uint NtcFloatToInt8(float h, float scale)
{
    return uint(int(clamp(h * scale, -128.f, 127.f)) & 0xff);
}

uint NtcPackFloat4(float4 h, float scale)
{
    return NtcFloatToInt8(h.x, scale)
        | (NtcFloatToInt8(h.y, scale) << 8)
        | (NtcFloatToInt8(h.z, scale) << 16)
        | (NtcFloatToInt8(h.w, scale) << 24);
}

uint NtcPackInt8x4(int4 vec)
{    
    #if USE_PACKING_INTRINSICS
    {
        return pack_s8(vec);
    }
    #else
    {
        return uint(vec.x & 0xff) 
            | (uint(vec.y & 0xff) << 8) 
            | (uint(vec.z & 0xff) << 16) 
            | (uint(vec.w) << 24);
    }
    #endif
}

int4 NtcUnpackInt8x4(uint packed)
{
    #if USE_PACKING_INTRINSICS
    {
        return unpack_s8s32(packed);
    }
    #else
    {
        int4 result;
        result.x = (int(packed) << 24) >> 24;
        result.y = (int(packed) << 16) >> 24;
        result.z = (int(packed) << 8) >> 24;
        result.w = int(packed) >> 24;
        return result;
    }
    #endif
}

// Software emulation of the dot4add_i8packed intrinsic
int NtcDotProductInt8x4(uint32_t a, uint32_t b)
{
    int ia = a;
    int ib = b;

    return (ia >> 24) * (ib >> 24)
        + ((ia << 8) >> 24) * ((ib << 8) >> 24)
        + ((ia << 16) >> 24) * ((ib << 16) >> 24)
        + ((ia << 24) >> 24) * ((ib << 24) >> 24);
}

// Converts the int4 packed version of LatentEncodingConstants into a struct
NtcLatentEncodingConstants NtcUnpackLatentEncodingConstants(int4 i[2])
{
    NtcLatentEncodingConstants result;
    result.numFeatures = i[0].x;
    result.quantBits = i[0].y;
    result.logElementsPerUint = i[0].z;
    result.pad = i[0].w;
    result.addressMask = uint(i[1].x);
    result.dataMask  = uint(i[1].y);
    result.quantizedScale = i[1].z;
    result.quantizedBias = i[1].w;
    return result;
}

// Converts the int4 packed version of NeuralMipConstants into a struct
NtcNeuralMipConstants NtcUnpackNeuralMipConstants(int4 i)
{
    NtcNeuralMipConstants result;
    result.dataOffset = uint(i.x);
    result.imageWidth = uint(i.y) & 0xffff;
    result.imageHeight = uint(i.y) >> 16;
    result.sliceLeft = uint(i.z) & 0xffff;
    result.sliceTop = uint(i.z) >> 16;
    result.sliceWidth = uint(i.w) & 0xffff;
    result.sliceHeight = uint(i.w) >> 16;
    return result;
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

uint4 NtcLoadFourRawLatents(
    ByteAddressBuffer buffer,
    uint bufferOffset,
    NtcLatentEncodingConstants encoding,
    NtcNeuralMipConstants neuralMip,
    int addr)
{
    uint word = buffer.Load(bufferOffset + neuralMip.dataOffset + (addr >> encoding.logElementsPerUint) * 4);
    const uint firstOffset = (addr & encoding.addressMask) * encoding.quantBits;

    word = word >> firstOffset;
    const uint bits0 = word & encoding.dataMask; word = word >> encoding.quantBits;
    const uint bits1 = word & encoding.dataMask; word = word >> encoding.quantBits;
    const uint bits2 = word & encoding.dataMask; word = word >> encoding.quantBits;
    const uint bits3 = word & encoding.dataMask;

    return uint4(bits0, bits1, bits2, bits3);
}

int4 NtcLoadFourInputQuantizedLatents(
    ByteAddressBuffer buffer,
    uint bufferOffset,
    NtcLatentEncodingConstants encoding,
    NtcNeuralMipConstants neuralMip,
    int addr)
{
    uint4 bits = NtcLoadFourRawLatents(buffer, bufferOffset, encoding, neuralMip, addr);

    return int4(bits.xyzw) * encoding.quantizedScale + encoding.quantizedBias;
}

void NtcSetupLatentBilinearFilter(
    NtcNeuralMipConstants neuralMip,
    float2 uv,
    out int2 topLeftPos,
    out float4 weights)
{
    const float2 pixelPos = uv * float2(neuralMip.imageWidth, neuralMip.imageHeight)
        - float2(neuralMip.sliceLeft, neuralMip.sliceTop) - 0.5f;

    topLeftPos = int2(floor(pixelPos));

    const float dx = pixelPos.x - topLeftPos.x;
    const float dy = pixelPos.y - topLeftPos.y;
    const float dxn = 1 - dx;
    const float dyn = 1 - dy;

    weights.x = dxn * dyn;
    weights.y = dx * dyn;
    weights.z = dxn * dy;
    weights.w = dx * dy;
}

NTC_TEMPLATE_FN_3(bool, NtcSampleLatentGrid, int, NUM_FEATURES, bool, ALL_CORNERS, int, OUTPUT_SIZE)
    (ByteAddressBuffer buffer,
    uint bufferOffset,
    NtcLatentEncodingConstants encoding,
    NtcNeuralMipConstants neuralMip,
    float2 uv,
    int outputOffset,
    inout uint outputArray[OUTPUT_SIZE])
{
    if (neuralMip.sliceWidth == 0 || neuralMip.sliceHeight == 0)
        return false;

    int2 topLeftPos;
    float4 weights;
    NtcSetupLatentBilinearFilter(neuralMip, uv, topLeftPos, weights);
    int4 iweights = int4(weights * 256.f);

    // Shift right the interpolated weights by 8 to undo the 256 factor above
    const int normalizationShift = 8;

    const int x0 = min(max(topLeftPos.x, 0), neuralMip.sliceWidth - 1);
    const int y0 = min(max(topLeftPos.y, 0), neuralMip.sliceHeight - 1);
    const int x1 = min(max(topLeftPos.x + 1, 0), neuralMip.sliceWidth - 1);
    const int y1 = min(max(topLeftPos.y + 1, 0), neuralMip.sliceHeight - 1);

    int a00 = (y0 * neuralMip.sliceWidth + x0) * encoding.numFeatures;
    int a01 = (y0 * neuralMip.sliceWidth + x1) * encoding.numFeatures;
    int a10 = (y1 * neuralMip.sliceWidth + x0) * encoding.numFeatures;
    int a11 = (y1 * neuralMip.sliceWidth + x1) * encoding.numFeatures;

#if __SLANG__
    [ForceUnroll]
#else
    [unroll]
#endif
    for (int i = 0; i < NUM_FEATURES / 4; i++)
    {
        if (i >= encoding.numFeatures / 4)
            break;

        const int4 x00 = NtcLoadFourInputQuantizedLatents(buffer, bufferOffset, encoding, neuralMip, a00) * iweights.x; a00 += 4;
        const int4 x01 = NtcLoadFourInputQuantizedLatents(buffer, bufferOffset, encoding, neuralMip, a01) * iweights.y; a01 += 4;
        const int4 x10 = NtcLoadFourInputQuantizedLatents(buffer, bufferOffset, encoding, neuralMip, a10) * iweights.z; a10 += 4;
        const int4 x11 = NtcLoadFourInputQuantizedLatents(buffer, bufferOffset, encoding, neuralMip, a11) * iweights.w; a11 += 4;

        if (ALL_CORNERS)
        {
            // Copy the latents for the 4 pixels into the network inputs.
            outputArray[outputOffset + i + (NUM_FEATURES / 4) * 0] = NtcPackInt8x4(x00 >> normalizationShift);
            outputArray[outputOffset + i + (NUM_FEATURES / 4) * 1] = NtcPackInt8x4(x01 >> normalizationShift);
            outputArray[outputOffset + i + (NUM_FEATURES / 4) * 2] = NtcPackInt8x4(x10 >> normalizationShift);
            outputArray[outputOffset + i + (NUM_FEATURES / 4) * 3] = NtcPackInt8x4(x11 >> normalizationShift);
        }
        else
        {
            // Blend the features of the 4 pixels.
            int4 d = (x00 + x01 + x10 + x11) >> normalizationShift;
            outputArray[outputOffset + i] = NtcPackInt8x4(d);
        }
    }

    return true;
}

float4 NtcEvaluatePositionalEncoding(float2 posf, float iscale)
{
    float4 result;

    result.x = frac(posf.x * iscale) * 4;
    result.x = abs(result.x - 2) - 1;
    result.y = frac(posf.y * iscale) * 4;
    result.y = abs(result.y - 2) - 1;

    result.z = frac(posf.x * iscale + 0.25f) * 4;
    result.z = abs(result.z - 2) - 1;
    result.w = frac(posf.y * iscale + 0.25f) * 4;
    result.w = abs(result.w - 2) - 1;

    return result;
}

NTC_TEMPLATE_FN_1(void, NtcEncodeSamplePosition, int, OUTPUT_SIZE)
    (float2 posf, float lod, int offset, inout uint outputArray[OUTPUT_SIZE])
{
    int idx = offset;
    int scale = NTC_MLP_POS_ENC_SCALE;
    float iscale = 1.f / scale;
    
    [unroll]
    for (; scale > 1; scale >>= 1)
    {
        float4 enc = NtcEvaluatePositionalEncoding(posf, iscale);
        outputArray[idx] = NtcPackFloat4(enc, c_InputScale);

        idx++;
        iscale *= 2;
    }
    
    outputArray[idx] = NtcPackFloat4(float4(lod.xx, 0, 0), c_InputScale);
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
    inout int scaleBiasOffset,
    int totalChannels,
    bool activation,
    uint inputArray[IN / 4],
#if NTC_USE_FLOAT16
    out uint outputArray[OUTPUT_LAYER ? OUT / 2 : OUT / 4]
#else
    out uint outputArray[OUTPUT_LAYER ? OUT : OUT / 4]
#endif
)
{
    // See the comment block in the beginning of TextureSet.cpp for the weight layouts

    // Note: not unrolling the outer loop.
    // If we do, DXC/SPIR-V crashes.
    // DXC/DXIL compiles the unrolled loop successfully, but then creating a pipeline with it takes seconds,
    // and the resulting code works slower than a regular loop.
    for (uint c = 0; c < OUT; c += 4)
    {
        int4 biases = weightBuffer.Load<int4>(scaleBiasOffset + (totalChannels + c) * 4);
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
            
#if NTC_USE_DP4A
            acc0 = dot4add_i8packed(inputArray[k], weights0, acc0);
            acc1 = dot4add_i8packed(inputArray[k], weights1, acc1);
            acc2 = dot4add_i8packed(inputArray[k], weights2, acc2);
            acc3 = dot4add_i8packed(inputArray[k], weights3, acc3);
#else
            acc0 += DotProductInt8x4(inputArray[k], weights0);
            acc1 += DotProductInt8x4(inputArray[k], weights1);
            acc2 += DotProductInt8x4(inputArray[k], weights2);
            acc3 += DotProductInt8x4(inputArray[k], weights3);
#endif
        }
        
        float4 results = float4(acc0, acc1, acc2, acc3);
        float4 scales = weightBuffer.Load<float4>(scaleBiasOffset + c * 4);

#if NTC_USE_FLOAT16
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
#else
        float4 hresults = results * scales;
        
        if (activation)
        {
            hresults = NtcHGELUClamp_ForwardFloat(hresults);
        }

        if (OUTPUT_LAYER)
        {
            outputArray[c + 0] = asuint(hresults.x);
            outputArray[c + 1] = asuint(hresults.y);
            outputArray[c + 2] = asuint(hresults.z);
            outputArray[c + 3] = asuint(hresults.w);
        }
        else
        {
            int4 iresults = NtcHGELUClamp_QuantizeFloat(hresults);

            outputArray[c / 4] = PackInt8x4(iresults);
        }
#endif
    }

    // Advance the input offsets to point at the next layer.
    scaleBiasOffset += OUT * sizeof(float);
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

NTC_TEMPLATE_FN_1(bool, NtcPrepareNetworkInputs, int, VERSION)
    (NtcTextureSetConstants desc,
    ByteAddressBuffer latentsBuffer,
    uint latentsOffset,
    int2 texel,
    int mipLevel,
    out uint networkInputs[NtcNetworkParams<VERSION>::INPUT_CHANNELS / 4])
{
    typedef NtcNetworkParams<VERSION> Params;

    const int2 imageSize = NtcGetTextureDimensions(desc, mipLevel);
    const float2 uv = (float2(texel) + 0.5) / imageSize;

    // Zero init the array - in some cases, OUTPUT_SIZE is rounded up from the actual used size.
    [unroll]
    for (int i = 0; i < Params::INPUT_CHANNELS / 4; ++i)
        networkInputs[i] = 0;

    const NtcColorMipConstants colorMip = NtcUnpackColorMipConstants(desc.colorMips[mipLevel]);

    if (colorMip.neuralMip < 0)
        return false;

    int inputOffset = 0;

    // Sample the latent grids
    if (!NtcSampleLatentGrid<Params::HR_FEATURES, true, Params::INPUT_CHANNELS / 4>(latentsBuffer, latentsOffset,
        NtcUnpackLatentEncodingConstants(desc.highResEncoding),
        NtcUnpackNeuralMipConstants(desc.highResNeuralMips[colorMip.neuralMip]),
        uv, inputOffset, networkInputs))
        return false;
    inputOffset += Params::SAMPLED_FEATURES_HR / 4;

    if (!NtcSampleLatentGrid<Params::LR_FEATURES, false, Params::INPUT_CHANNELS / 4>(latentsBuffer, latentsOffset,
        NtcUnpackLatentEncodingConstants(desc.lowResEncoding),
        NtcUnpackNeuralMipConstants(desc.lowResNeuralMips[colorMip.neuralMip]),
        uv, inputOffset, networkInputs))
        return false;
    inputOffset += Params::SAMPLED_FEATURES_LR / 4;

    // Encode the sample position
    NtcEncodeSamplePosition<Params::INPUT_CHANNELS / 4>(float2(texel) * colorMip.positionScale,
        colorMip.positionLod, inputOffset, networkInputs);

    return true;
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
// Use like NtcSampleTextureSet<NETWORK_VERSION>(Constants, LatentsBuffer, ...)
// Returns true if the mip level is valid; out-of-bounds texel positions are clamped.
NTC_TEMPLATE_FN_1(bool, NtcSampleTextureSet, int, VERSION)
    (NtcTextureSetConstants desc,
    ByteAddressBuffer latentsBuffer,
    uint latentsOffset, // Offset of the latents chunk in latentsBuffer if packing multiple textures together
    ByteAddressBuffer weightsBuffer,
    uint weightsOffset, // Offset of the weight chunk in weightsBuffer if packing multiple textures together
    int2 texel,
    int mipLevel,
    bool convertToLinearColorSpace,
    inout float outputs[NtcNetworkParams<VERSION>::OUTPUT_CHANNELS])
{
    typedef NtcNetworkParams<VERSION> Params;

    uint networkInputs[Params::INPUT_CHANNELS / 4];
    if (!NtcPrepareNetworkInputs<VERSION>(desc, latentsBuffer, latentsOffset, texel, mipLevel, networkInputs))
        return false;

    int scaleBiasOffset = weightsOffset + desc.networkScaleBiasOffset;

    // Evaluate the MLP layers:
    const int totalChannels = Params::HIDDEN_LAYER_CHANNELS * 3 + Params::OUTPUT_CHANNELS;

    // Input layer
    uint hiddenOutput1[Params::HIDDEN_LAYER_CHANNELS / 4];
    NtcEvaluateLayerINT8<Params::INPUT_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, false>
        (weightsBuffer, weightsOffset + desc.networkWeightOffsets.x, scaleBiasOffset,
        totalChannels, true, networkInputs, hiddenOutput1);

    // Hidden layer 1
    uint hiddenOutput2[Params::HIDDEN_LAYER_CHANNELS / 4];
    NtcEvaluateLayerINT8<Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, false>
        (weightsBuffer, weightsOffset + desc.networkWeightOffsets.y, scaleBiasOffset,
        totalChannels, true, hiddenOutput1, hiddenOutput2);

    // Hidden layer 2
    NtcEvaluateLayerINT8<Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, false>
        (weightsBuffer, weightsOffset + desc.networkWeightOffsets.z, scaleBiasOffset,
        totalChannels, true, hiddenOutput2, hiddenOutput1);

    // Output layer
#if NTC_USE_FLOAT16
    uint networkOutputs[Params::OUTPUT_CHANNELS / 2];
#else
    uint networkOutputs[Params::OUTPUT_CHANNELS];
#endif
    NtcEvaluateLayerINT8<Params::HIDDEN_LAYER_CHANNELS, Params::OUTPUT_CHANNELS, true>
        (weightsBuffer, weightsOffset + desc.networkWeightOffsets.w, scaleBiasOffset,
        totalChannels, false, hiddenOutput1, networkOutputs);

#if NTC_USE_FLOAT16
    [unroll]
    for (int ch = 0; ch < Params::OUTPUT_CHANNELS/2; ++ch)
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
#else
    [unroll]
    for (int ch = 0; ch < Params::OUTPUT_CHANNELS; ++ch)
    {
        outputs[ch] = asfloat(networkOutputs[ch]);

        if (convertToLinearColorSpace)
        {
            outputs[ch] = NtcConvertChannelToLinearColorSpace(desc, ch, outputs[ch]);
        }
    }
#endif
    
    return true;
}

#endif