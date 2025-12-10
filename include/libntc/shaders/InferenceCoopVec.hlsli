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

#ifndef NTC_INFERENCE_COOPVEC_SLANGH
#define NTC_INFERENCE_COOPVEC_SLANGH

#if !__SLANG__
#define CoopVec vector
#endif

#include "Inference.hlsli"

bool NtcSampleLatentGrid_FP16(
    Texture2DArray latentTexture,
    SamplerState latentSampler,
    float2 uv,
    int neuralLod,
    int featureOffset,
    inout CoopVec<float16_t, NTC_MLP_INPUT_CHANNELS> outputArray)
{
    int width, height, arraySize;
    latentTexture.GetDimensions(width, height, arraySize);

    [unroll]
    for (int layerIndex = 0; layerIndex < NTC_MLP_FEATURES / NTC_FEATURES_PER_LAYER; ++layerIndex)
    {
        const bool mask = (layerIndex == 0) || (layerIndex < arraySize);

        float4 sampledValue = latentTexture.SampleLevel(latentSampler, float3(uv, layerIndex), neuralLod);
        sampledValue = sampledValue.bgra; // The texture format is BGRA4, unswizzle that
        sampledValue = mask ? sampledValue * 2.f - 1.f : 0.f;
        
        outputArray[featureOffset + layerIndex * NTC_FEATURES_PER_LAYER + 0] = float16_t(sampledValue.x);
        outputArray[featureOffset + layerIndex * NTC_FEATURES_PER_LAYER + 1] = float16_t(sampledValue.y);
        outputArray[featureOffset + layerIndex * NTC_FEATURES_PER_LAYER + 2] = float16_t(sampledValue.z);
        outputArray[featureOffset + layerIndex * NTC_FEATURES_PER_LAYER + 3] = float16_t(sampledValue.w);
    }

    return true;
}

void NtcEncodeSamplePosition_FP16(
    float2 posf,
    float lod,
    int offset,
    inout CoopVec<float16_t, NTC_MLP_INPUT_CHANNELS> outputArray)
{
    int idx = offset;
    
    [unroll]
    for (int wave = 0; wave < NTC_MLP_POS_ENC_WAVES; ++wave)
    {
        float4 enc = NtcEvaluatePositionalEncoding(posf);

        outputArray[idx + 0] = float16_t(enc.x);
        outputArray[idx + 1] = float16_t(enc.y);
        outputArray[idx + 2] = float16_t(enc.z);
        outputArray[idx + 3] = float16_t(enc.w);

        idx += 4;
        posf *= 2.f;
    }
    
    outputArray[idx + 0] = float16_t(lod);
    outputArray[idx + 1] = float16_t(lod);
}

bool NtcPrepareNetworkInputsInternal_FP16(
    Texture2DArray latentTexture,
    SamplerState latentSampler,
    int2 texel,
    float2 uv,
    const NtcColorMipConstants colorMip,
    out CoopVec<float16_t, NTC_MLP_INPUT_CHANNELS> networkInputs)
{
    // Zero init the vector
    [unroll]
    for (int i = 0; i < NTC_MLP_INPUT_CHANNELS; ++i)
        networkInputs[i] = 0;

    if (colorMip.neuralMip < 0)
        return false;

    // Sample the latent grids
    if (!NtcSampleLatentGrid_FP16(latentTexture, latentSampler,
        uv, colorMip.neuralMip, 0, networkInputs))
        return false;

    if (!NtcSampleLatentGrid_FP16(latentTexture, latentSampler,
        uv, colorMip.neuralMip + 1, NTC_MLP_FEATURES, networkInputs))
        return false;

    // Encode the sample position
    NtcEncodeSamplePosition_FP16(float2(texel) * colorMip.positionScale,
        colorMip.positionLod, NTC_MLP_FEATURES * 2, networkInputs);

    return true;
}

bool NtcPrepareNetworkInputs_FP16(
    NtcTextureSetConstants desc,
    Texture2DArray latentTexture,
    SamplerState latentSampler,
    int2 texel,
    int mipLevel,
    inout CoopVec<float16_t, NTC_MLP_INPUT_CHANNELS> networkInputs)
{
    const int2 imageSize = NtcGetTextureDimensions(desc, mipLevel);
    const float2 uv = (float2(texel) + 0.5) / imageSize;

    const NtcColorMipConstants colorMip = NtcUnpackColorMipConstants(desc.colorMips[mipLevel]);

    return NtcPrepareNetworkInputsInternal_FP16(latentTexture, latentSampler,
        texel, uv, colorMip, networkInputs);
}


#if __SLANG__
void NtcHGELUClamp_Forward_CoopVec<T: __BuiltinFloatingPointType, let SIZE: int>(inout CoopVec<T, SIZE> x, bool scaleAndBias)
#else
template<typename T, int SIZE>
void NtcHGELUClamp_Forward_CoopVec(inout CoopVec<T, SIZE> x, bool scaleAndBias)
#endif
{
    const NtcHGELUParams params = NtcGetHGELUParams();

#if __SLANG__
    let v3  = CoopVec<T, SIZE>(T(params.maxval));
    let v0  = CoopVec<T, SIZE>(T(0.f));
    let v1  = CoopVec<T, SIZE>(T(1.f));
    let vi3 = CoopVec<T, SIZE>(T(1/3.f));
    let v05 = CoopVec<T, SIZE>(T(0.5f));

    x = min(x, v3) * clamp(vi3 * x + v05, v0, v1);

    if (scaleAndBias)
    {
        let istep = CoopVec<T, SIZE>(T(params.invStep));
        let obias = CoopVec<T, SIZE>(T(params.bias));
        x = x * istep + obias;
    }
#else
    CoopVec<T, SIZE> tmp, v3;
    tmp = x * T(1.0 / 3.0) + T(0.5);
    tmp = clamp(tmp, T(0.0), T(1.0));
    x = tmp * min(x, T(params.maxval));

    if (scaleAndBias)
    {
        x = x * T(params.invStep) + T(params.bias);
    }
#endif
}

#if __SLANG__
    inline void NtcEvaluateLayerMatMulAdd_CoopVec_Int8
        <T_IN: __BuiltinArithmeticType, let IN: int, let OUT: int>
#else
    template<typename T_IN, int IN, int OUT>
    void NtcEvaluateLayerMatMulAdd_CoopVec_Int8
#endif
    (ByteAddressBuffer weightBuffer,
    int weightOffset,
    int biasOffset,
    in CoopVec<T_IN, IN> inputArray,
    out CoopVec<float, OUT> outputArray)
{
#if __SLANG__

    const CoopVecComponentType inputType = CoopVecComponentType::SignedInt8;
    const CoopVecComponentType weightType = CoopVecComponentType::SignedInt8;
    const CoopVecMatrixLayout weightLayout = CoopVecMatrixLayout::InferencingOptimal;
    const CoopVecComponentType biasType = CoopVecComponentType::SignedInt32;
    const uint stride = 0;

    CoopVec<int, OUT> accum;
    accum = coopVecMatMulAdd<int, OUT, IN, T_IN>(
        inputArray,
        inputType,
        weightBuffer,
        weightOffset,
        weightType,
        weightBuffer,
        biasOffset,
        biasType,
        weightLayout,
        false,
        stride
    );

    outputArray = CoopVec<float, OUT>(accum);

#else

    // See https://microsoft.github.io/hlsl-specs/proposals/0031-hlsl-vector-matrix-operations
    // for the enum value definitions
    
    const uint inputType = 20; // SINT8
    const uint weightType = 20; // SINT8
    const uint weightLayout = 2; // MUL_OPTIMAL
    const uint biasType = 4; // SINT32
    const uint stride = 0;
    
    CoopVec<int, OUT> accum;
    __builtin_MatVecMulAdd(
        accum,
        false,
        inputArray,
        false,
        inputType,
        weightBuffer,
        weightOffset,
        weightType,
        OUT,
        IN,
        weightLayout,
        false,
        stride,
        weightBuffer,
        biasOffset,
        biasType
    );
    
    outputArray = accum;
#endif
}

NTC_TEMPLATE_FN_2(void, NtcEvaluateLayerMatMulAdd_CoopVec_FP8, int, IN, int, OUT)
    (ByteAddressBuffer weightBuffer,
    int weightOffset,
    int biasOffset,
    in CoopVec<float16_t, IN> inputArray,
    out CoopVec<float16_t, OUT> outputArray)
{
#if __SLANG__

    const CoopVecComponentType inputType = CoopVecComponentType::FloatE4M3;
    const CoopVecComponentType weightType = CoopVecComponentType::FloatE4M3;
    const CoopVecComponentType biasType = CoopVecComponentType::Float16;
    const CoopVecMatrixLayout weightLayout = CoopVecMatrixLayout::InferencingOptimal;
    const uint stride = 0;

    outputArray = coopVecMatMulAdd<float16_t, OUT, IN, float16_t>(
        inputArray,
        inputType,
        weightBuffer,
        weightOffset,
        weightType,
        weightBuffer,
        biasOffset,
        biasType,
        weightLayout,
        false,
        stride
    );

#else

    const uint inputType = 21; // F8_E4M3
    const uint weightType = 21; // F8_E4M3
    const uint weightLayout = 2; // MATRIX_LAYOUT_MUL_OPTIMAL
    const uint biasType = 8; // FLOAT16
    const uint stride = 0;

    __builtin_MatVecMulAdd(
        outputArray,
        false,
        inputArray,
        false,
        inputType,
        weightBuffer,
        weightOffset,
        weightType,
        OUT,
        IN,
        weightLayout,
        false,
        stride,
        weightBuffer,
        biasOffset,
        biasType
    );

#endif
}

NTC_TEMPLATE_FN_3(void, NtcEvaluateLayer_CoopVec_FP8, int, IN, int, OUT, bool, ACT)
    (ByteAddressBuffer weightBuffer,
    int weightOffset,
    int biasOffset,
    bool scaleActivation,
    in CoopVec<float16_t, IN> inputArray,
    out CoopVec<float16_t, OUT> outputArray)
{
    NtcEvaluateLayerMatMulAdd_CoopVec_FP8<IN, OUT>
        (weightBuffer, weightOffset, biasOffset, inputArray, outputArray);
   
    if (ACT)
    {
        NtcHGELUClamp_Forward_CoopVec<float16_t, OUT>(outputArray, scaleActivation);
    }
}

#if __SLANG__
    inline void NtcEvaluateOutputLayer_CoopVec_FP8
        <let IN: int, let OUT: int>
#else
    template<int IN, int OUT>
    void NtcEvaluateOutputLayer_CoopVec_FP8
#endif
    (ByteAddressBuffer weightBuffer,
    int weightOffset,
    int biasOffset,
    int scaleOffset,
    in CoopVec<float16_t, IN> inputArray,
    out CoopVec<float, OUT> outputArray)
{
    // Convert the inputs from float16_t to float, necessary for correct output at this time
#if __SLANG__
    CoopVec<float, IN> inputArrayFloat = CoopVec<float, IN>(inputArray);
#else
    CoopVec<float, IN> inputArrayFloat;
    inputArrayFloat = inputArray;
#endif

    NtcEvaluateLayerMatMulAdd_CoopVec_Int8<float, IN, OUT>
        (weightBuffer, weightOffset, biasOffset, inputArrayFloat, outputArray);
    
    // Enforce the scale alignment to help the compiler optimize the code
    scaleOffset &= ~15;
    
#if __SLANG__
    let scale = CoopVec<float, OUT>.load(weightBuffer, scaleOffset);
#else
    CoopVec<float, OUT> scale = weightBuffer.Load<CoopVec<float, OUT> >(scaleOffset);
#endif

    outputArray = outputArray * scale;
}

void NtcEvaluateMLP_CoopVec_FP8(
    ByteAddressBuffer weightsBuffer,
    uint weightsOffset,
    int4 networkWeightOffsets,
    int4 networkBiasOffsets,
    int4 networkScaleOffsets,
    CoopVec<float16_t, NTC_MLP_INPUT_CHANNELS> networkInputs,
    out CoopVec<float, NTC_MLP_OUTPUT_CHANNELS> networkOutputs)
{
    // Input layer
    CoopVec<float16_t, NTC_MLP_HIDDEN0_CHANNELS> hiddenOutput0;
    NtcEvaluateLayer_CoopVec_FP8<NTC_MLP_INPUT_CHANNELS, NTC_MLP_HIDDEN0_CHANNELS, true>(
        weightsBuffer,
        weightsOffset + networkWeightOffsets.x,
        weightsOffset + networkBiasOffsets.x,
        false, networkInputs, hiddenOutput0);
    
    // Hidden layer 1
    CoopVec<float16_t, NTC_MLP_HIDDEN1_CHANNELS> hiddenOutput1;
    NtcEvaluateLayer_CoopVec_FP8<NTC_MLP_HIDDEN0_CHANNELS, NTC_MLP_HIDDEN1_CHANNELS, true>(
        weightsBuffer,
        weightsOffset + networkWeightOffsets.y,
        weightsOffset + networkBiasOffsets.y,
        /* scaleActivation = */ (NTC_MLP_LAYERS == 3), hiddenOutput0, hiddenOutput1);
    
#if NTC_MLP_LAYERS == 4
    // Hidden layer 2
    CoopVec<float16_t, NTC_MLP_HIDDEN2_CHANNELS> hiddenOutput2;
    NtcEvaluateLayer_CoopVec_FP8<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_HIDDEN2_CHANNELS, true>(
        weightsBuffer,
        weightsOffset + networkWeightOffsets.z,
        weightsOffset + networkBiasOffsets.z,
        true, hiddenOutput1, hiddenOutput2);

    // Output layer
    NtcEvaluateOutputLayer_CoopVec_FP8<NTC_MLP_HIDDEN2_CHANNELS, NTC_MLP_OUTPUT_CHANNELS>(
        weightsBuffer,
        weightsOffset + networkWeightOffsets.w,
        weightsOffset + networkBiasOffsets.w,
        weightsOffset + networkScaleOffsets.w,
        hiddenOutput2, networkOutputs);
#else
    // Output layer
    NtcEvaluateOutputLayer_CoopVec_FP8<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_OUTPUT_CHANNELS>(
        weightsBuffer,
        weightsOffset + networkWeightOffsets.z,
        weightsOffset + networkBiasOffsets.z,
        weightsOffset + networkScaleOffsets.z,
        hiddenOutput1, networkOutputs);
#endif
}

// NtcSampleTextureSet_CoopVec_FP8 - version of NtcSampleTextureSet that uses Cooperative Vectors with FP8 (E4M3) math.
// Use like SampleTextureSet_CoopVec_FP8(Constants, LatentsBuffer, ...)
// Returns true if the mip level is valid; out-of-bounds texel positions are clamped.
bool NtcSampleTextureSet_CoopVec_FP8(
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
    CoopVec<float16_t, NTC_MLP_INPUT_CHANNELS> networkInputs;
    if (!NtcPrepareNetworkInputs_FP16(desc, latentTexture, latentSampler, texel, mipLevel, networkInputs))
        return false;

    CoopVec<float, NTC_MLP_OUTPUT_CHANNELS> networkOutputs;
    NtcEvaluateMLP_CoopVec_FP8(
        weightsBuffer,
        weightsOffset,
        desc.networkWeightOffsets,
        desc.networkBiasOffsets,
        desc.networkScaleOffsets,
        networkInputs,
        networkOutputs
    );

    [unroll]
    for (int ch = 0; ch < NTC_MLP_OUTPUT_CHANNELS; ++ch)
    {
        outputs[ch] = networkOutputs[ch];
        
        if (convertToLinearColorSpace)
        {
            outputs[ch] = NtcConvertChannelToLinearColorSpace(desc, ch, outputs[ch]);
        }
    }

    return true;
}

#endif
