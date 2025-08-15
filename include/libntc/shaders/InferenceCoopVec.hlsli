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

#if __SLANG__
void NtcCoopVecStoreHalf4<let SIZE: int>(inout CoopVec<float16_t, SIZE> vec, int offset, float16_t4 values)
#else
template<int SIZE>
void NtcCoopVecStoreHalf4(inout CoopVec<float16_t, SIZE> vec, int offset, float16_t4 values)
#endif
{
    vec[offset + 0] = values.x;
    vec[offset + 1] = values.y;
    vec[offset + 2] = values.z;
    vec[offset + 3] = values.w;
}

float16_t4 NtcLoadFourInputQuantizedLatents_FP16(
    ByteAddressBuffer buffer,
    uint bufferOffset,
    NtcLatentEncodingConstants encoding,
    NtcNeuralMipConstants neuralMip,
    int addr)
{
    uint4 bits = NtcLoadFourRawLatents(buffer, bufferOffset, encoding, neuralMip, addr);

    // TODO: pass the parameters as float16 in the CB
    return float16_t4(bits.xyzw) * float16_t(asfloat(encoding.quantizedScale)) + float16_t(asfloat(encoding.quantizedBias));
}

NTC_TEMPLATE_FN_3(bool, NtcSampleLatentGrid_FP16, int, NUM_FEATURES, bool, ALL_CORNERS, int, OUTPUT_SIZE)
    (ByteAddressBuffer buffer,
    uint bufferOffset,
    NtcLatentEncodingConstants encoding,
    NtcNeuralMipConstants neuralMip,
    float2 uv,
    int outputOffset,
    inout CoopVec<float16_t, OUTPUT_SIZE> outputArray)
{
    if (neuralMip.sliceWidth == 0 || neuralMip.sliceHeight == 0)
        return false;

    int2 topLeftPos;
    float4 weights;
    NtcSetupLatentBilinearFilter(neuralMip, uv, topLeftPos, weights);
    float16_t4 iweights = float16_t4(weights);

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

        float16_t4 x00 = NtcLoadFourInputQuantizedLatents_FP16(buffer, bufferOffset, encoding, neuralMip, a00) * iweights.x; a00 += 4;
        float16_t4 x01 = NtcLoadFourInputQuantizedLatents_FP16(buffer, bufferOffset, encoding, neuralMip, a01) * iweights.y; a01 += 4;
        float16_t4 x10 = NtcLoadFourInputQuantizedLatents_FP16(buffer, bufferOffset, encoding, neuralMip, a10) * iweights.z; a10 += 4;
        float16_t4 x11 = NtcLoadFourInputQuantizedLatents_FP16(buffer, bufferOffset, encoding, neuralMip, a11) * iweights.w; a11 += 4;

        if (ALL_CORNERS)
        {
            // Copy the latents for the 4 pixels into the network inputs.
            NtcCoopVecStoreHalf4(outputArray, outputOffset + i * 4 + NUM_FEATURES * 0, x00);
            NtcCoopVecStoreHalf4(outputArray, outputOffset + i * 4 + NUM_FEATURES * 1, x01);
            NtcCoopVecStoreHalf4(outputArray, outputOffset + i * 4 + NUM_FEATURES * 2, x10);
            NtcCoopVecStoreHalf4(outputArray, outputOffset + i * 4 + NUM_FEATURES * 3, x11);
        }
        else
        {
            // Blend the features of the 4 pixels.
            float16_t4 d = x00 + x01 + x10 + x11;
            NtcCoopVecStoreHalf4(outputArray, outputOffset + i * 4, d);
        }
    }

    return true;
}

NTC_TEMPLATE_FN_1(void, NtcEncodeSamplePosition_FP16, int, OUTPUT_SIZE)
    (float2 posf, float lod, int offset, inout CoopVec<float16_t, OUTPUT_SIZE> outputArray)
{
    int idx = offset;
    int scale = NTC_MLP_POS_ENC_SCALE;
    float iscale = 1.f / scale;
    
    [unroll]
    for (; scale > 1; scale >>= 1)
    {
        float4 enc = NtcEvaluatePositionalEncoding(posf, iscale);
        NtcCoopVecStoreHalf4(outputArray, idx, float16_t4(enc));

        idx+=4;
        iscale *= 2;
    }
    
    outputArray[idx+0] = float16_t(lod);
    outputArray[idx+1] = float16_t(lod);
}

NTC_TEMPLATE_FN_1(bool, NtcPrepareNetworkInputs_FP16, int, VERSION)
    (NtcTextureSetConstants desc,
    ByteAddressBuffer latentsBuffer,
    uint latentsOffset,
    int2 texel,
    int mipLevel,
    inout CoopVec<float16_t, NtcNetworkParams<VERSION>::INPUT_CHANNELS> networkInputs)
{
    typedef NtcNetworkParams<VERSION> Params;

    const int2 imageSize = NtcGetTextureDimensions(desc, mipLevel);
    const float2 uv = (float2(texel) + 0.5) / imageSize;

    const NtcColorMipConstants colorMip = NtcUnpackColorMipConstants(desc.colorMips[mipLevel]);

    if (colorMip.neuralMip < 0)
        return false;

    int inputOffset = 0;

    // Sample the latent grids
    if (!NtcSampleLatentGrid_FP16<Params::HR_FEATURES, true, Params::INPUT_CHANNELS>(latentsBuffer, latentsOffset,
        NtcUnpackLatentEncodingConstants(desc.highResEncoding),
        NtcUnpackNeuralMipConstants(desc.highResNeuralMips[colorMip.neuralMip]),
        uv, inputOffset, networkInputs))
        return false;
    inputOffset += Params::SAMPLED_FEATURES_HR;

    if (!NtcSampleLatentGrid_FP16<Params::LR_FEATURES, false, Params::INPUT_CHANNELS>(latentsBuffer, latentsOffset,
        NtcUnpackLatentEncodingConstants(desc.lowResEncoding),
        NtcUnpackNeuralMipConstants(desc.lowResNeuralMips[colorMip.neuralMip]),
        uv, inputOffset, networkInputs))
        return false;
    inputOffset += Params::SAMPLED_FEATURES_LR;

    // Encode the sample position
    NtcEncodeSamplePosition_FP16<Params::INPUT_CHANNELS>(float2(texel) * colorMip.positionScale,
        colorMip.positionLod, inputOffset, networkInputs);

    return true;
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
        <T_IN: __BuiltinArithmeticType, let T_IN_NUM: int, let IN_IS_PACKED: bool, let IN: int, let OUT: int>
#else
    template<typename T_IN, int T_IN_NUM, bool IN_IS_PACKED, int IN, int OUT>
    void NtcEvaluateLayerMatMulAdd_CoopVec_Int8
#endif
    (ByteAddressBuffer weightBuffer,
    int weightOffset,
    int biasOffset,
    in CoopVec<T_IN, T_IN_NUM> inputArray,
    out CoopVec<float, OUT> outputArray)
{
#if __SLANG__

    const CoopVecComponentType inputType = IN_IS_PACKED
        ? CoopVecComponentType::SignedInt8Packed
        : CoopVecComponentType::SignedInt8;
    const CoopVecComponentType weightType = CoopVecComponentType::SignedInt8;
    const CoopVecMatrixLayout weightLayout = CoopVecMatrixLayout::InferencingOptimal;
    const CoopVecComponentType biasType = CoopVecComponentType::SignedInt32;
    const uint stride = 0;

    CoopVec<int, OUT> accum;
    accum = coopVecMatMulAddPacked<int, OUT, T_IN_NUM, T_IN>(
        inputArray,
        inputType,
        IN,
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

    // See https://microsoft.github.io/hlsl-specs/proposals/0031-hlsl-vector-matrix-operations.html
    // for the enum value definitions
    
    const uint inputType = IN_IS_PACKED ? 17 /* SINT8_T4_PACKED */ : 20 /* SINT8 */;
    const uint weightType = 20; // SINT8
    const uint weightLayout = 2; // MUL_OPTIMAL
    const uint biasType = 4; // SINT32
    const uint stride = 0;
    
    CoopVec<int, OUT> accum;
    __builtin_MatVecMulAdd(
        accum,
        false,
        inputArray,
        IN_IS_PACKED,
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

NTC_TEMPLATE_FN_2(void, NtcEvaluateLayerMatMul_CoopVec_FP8, int, IN, int, OUT)
    (ByteAddressBuffer weightBuffer,
    int weightOffset,
    in CoopVec<float16_t, IN> inputArray,
    out CoopVec<float16_t, OUT> outputArray)
{
#if __SLANG__

    const CoopVecComponentType inputType = CoopVecComponentType::FloatE4M3;
    const CoopVecComponentType weightType = CoopVecComponentType::FloatE4M3;
    const CoopVecMatrixLayout weightLayout = CoopVecMatrixLayout::InferencingOptimal;
    const uint stride = 0;

    outputArray = coopVecMatMul<float16_t, OUT, IN, float16_t>(
        inputArray,
        inputType,
        weightBuffer,
        weightOffset,
        weightType,
        weightLayout,
        false,
        stride
    );

#else

    const uint inputType = 21; // F8_E4M3
    const uint weightType = 21; // F8_E4M3
    const uint weightLayout = 2; // MATRIX_LAYOUT_MUL_OPTIMAL
    const uint stride = 0;

    __builtin_MatVecMul(
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
        stride
    );

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

#if __SLANG__
    inline void NtcEvaluateLayer_CoopVec_Int8
        <T_IN: __BuiltinArithmeticType, let T_IN_NUM: int, let IN_IS_PACKED: bool, let IN: int, let OUT: int, let ACT: bool>
#else
    template<typename T_IN, int T_IN_NUM, bool IN_IS_PACKED, int IN, int OUT, bool ACT>
    void NtcEvaluateLayer_CoopVec_Int8
#endif
    (ByteAddressBuffer weightBuffer,
    int weightOffset,
    uint scaleBiasOffset,
    int totalChannels,
    in CoopVec<T_IN, T_IN_NUM> inputArray,
    out CoopVec<float, OUT> outputArray)
{
    // See the comment block in the beginning of TextureSet.cpp for the weight layouts

    const uint biasOffset = scaleBiasOffset + totalChannels * sizeof(float);
    
    NtcEvaluateLayerMatMulAdd_CoopVec_Int8<T_IN, T_IN_NUM, IN_IS_PACKED, IN, OUT>
        (weightBuffer, weightOffset, biasOffset, inputArray, outputArray);
    
    // Enforce the scale alignment to help the compiler optimize the code
    scaleBiasOffset &= ~15;
    
#if __SLANG__
    let scale = CoopVec<float, OUT>.load(weightBuffer, scaleBiasOffset);
#else
    CoopVec<float, OUT> scale = weightBuffer.Load<CoopVec<float, OUT> >(scaleBiasOffset);
#endif
    outputArray = outputArray * scale;

    if (ACT)
    {
        NtcHGELUClamp_Forward_CoopVec<float, OUT>(outputArray, true);
    }
}

NTC_TEMPLATE_FN_3(void, NtcEvaluateLayer_CoopVec_FP8, int, IN, int, OUT, bool, ACT)
    (ByteAddressBuffer weightBuffer,
    int weightOffset,
    uint scaleBiasOffset,
    bool scaleActivation,
    in CoopVec<float16_t, IN> inputArray,
    out CoopVec<float16_t, OUT> outputArray)
{
    // See the comment block in the beginning of TextureSet.cpp for the weight layouts

    NtcEvaluateLayerMatMulAdd_CoopVec_FP8<IN, OUT>
        (weightBuffer, weightOffset, scaleBiasOffset, inputArray, outputArray);
   
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
    uint scaleBiasOffset,
    in CoopVec<float16_t, IN> inputArray,
    out CoopVec<float, OUT> outputArray)
{
    // See the comment block in the beginning of TextureSet.cpp for the weight layouts

    // Convert the inputs from float16_t to float, necessary for correct output at this time
#if __SLANG__
    CoopVec<float, IN> inputArrayFloat = CoopVec<float, IN>(inputArray);
#else
    CoopVec<float, IN> inputArrayFloat;
    inputArrayFloat = inputArray;
#endif

    const int biasOffset = scaleBiasOffset + OUT * sizeof(float);

    NtcEvaluateLayerMatMulAdd_CoopVec_Int8<float, IN, false, IN, OUT>
        (weightBuffer, weightOffset, biasOffset, inputArrayFloat, outputArray);
    
    // Enforce the scale alignment to help the compiler optimize the code
    scaleBiasOffset &= ~15;
    
#if __SLANG__
    let scale = CoopVec<float, OUT>.load(weightBuffer, scaleBiasOffset);
#else
    CoopVec<float, OUT> scale = weightBuffer.Load<CoopVec<float, OUT> >(scaleBiasOffset);
#endif

    outputArray = outputArray * scale;
}

// NtcSampleTextureSet_CoopVec_Int8 - version of NtcSampleTextureSet that uses Cooperative Vectors with Int8 math.
// Use like SampleTextureSet_CoopVec_Int8<NETWORK_VERSION>(Constants, LatentsBuffer, ...)
// Returns true if the mip level is valid; out-of-bounds texel positions are clamped.
NTC_TEMPLATE_FN_1(bool, NtcSampleTextureSet_CoopVec_Int8, int, VERSION)
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

    CoopVec<uint32_t, Params::INPUT_CHANNELS / 4> networkInputsVec;
    [unroll]
    for (int i = 0; i < Params::INPUT_CHANNELS / 4; ++i)
    {
        networkInputsVec[i] = networkInputs[i];
    }

    // Input layer
    CoopVec<float, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput1;
    NtcEvaluateLayer_CoopVec_Int8<uint32_t, Params::INPUT_CHANNELS/4, true, Params::INPUT_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (weightsBuffer, weightsOffset + desc.networkWeightOffsets.x, scaleBiasOffset, totalChannels, networkInputsVec, hiddenOutput1);
    // Advance scaleBiasOffset to point at the next layer - it's here as a workaround for a Slang bug
    // that prevents it from compiling EvaluateLayer_CoopVec_Int8 with scaleBiasOffset as 'inout' parameter.
    scaleBiasOffset += Params::HIDDEN_LAYER_CHANNELS * sizeof(float);
    
    // Hidden layer 1
    CoopVec<float, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput2;
    NtcEvaluateLayer_CoopVec_Int8<float, Params::HIDDEN_LAYER_CHANNELS, false, Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (weightsBuffer, weightsOffset + desc.networkWeightOffsets.y, scaleBiasOffset, totalChannels, hiddenOutput1, hiddenOutput2);
    scaleBiasOffset += Params::HIDDEN_LAYER_CHANNELS * sizeof(float);
    
    // Hidden layer 2
    CoopVec<float, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput3;
    NtcEvaluateLayer_CoopVec_Int8<float, Params::HIDDEN_LAYER_CHANNELS, false, Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (weightsBuffer, weightsOffset + desc.networkWeightOffsets.z, scaleBiasOffset, totalChannels, hiddenOutput2, hiddenOutput3);
    scaleBiasOffset += Params::HIDDEN_LAYER_CHANNELS * sizeof(float);
    
    // Output layer
    CoopVec<float, Params::OUTPUT_CHANNELS> networkOutputs;
    NtcEvaluateLayer_CoopVec_Int8<float, Params::HIDDEN_LAYER_CHANNELS, false, Params::HIDDEN_LAYER_CHANNELS, Params::OUTPUT_CHANNELS, false>
    (weightsBuffer, weightsOffset + desc.networkWeightOffsets.w, scaleBiasOffset, totalChannels, hiddenOutput3, networkOutputs);

    [unroll]
    for (int ch = 0; ch < Params::OUTPUT_CHANNELS; ++ch)
    {
        outputs[ch] = networkOutputs[ch];
        
        if (convertToLinearColorSpace)
        {
            outputs[ch] = NtcConvertChannelToLinearColorSpace(desc, ch, outputs[ch]);
        }
    }

    return true;
}

// NtcSampleTextureSet_CoopVec_FP8 - version of NtcSampleTextureSet that uses Cooperative Vectors with FP8 (E4M3) math.
// Use like SampleTextureSet_CoopVec_FP8<NETWORK_VERSION>(Constants, LatentsBuffer, ...)
// Returns true if the mip level is valid; out-of-bounds texel positions are clamped.
NTC_TEMPLATE_FN_1(bool, NtcSampleTextureSet_CoopVec_FP8, int, VERSION)
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

    CoopVec<float16_t, Params::INPUT_CHANNELS> networkInputs;
    if (!NtcPrepareNetworkInputs_FP16<VERSION>(desc, latentsBuffer, latentsOffset, texel, mipLevel, networkInputs))
        return false;

    int scaleBiasOffset = weightsOffset + desc.networkScaleBiasOffset;

    // Evaluate the MLP layers:
    const int totalChannels = Params::HIDDEN_LAYER_CHANNELS * 3 + Params::OUTPUT_CHANNELS;

    // Input layer
    CoopVec<float16_t, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput1;
    NtcEvaluateLayer_CoopVec_FP8<Params::INPUT_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (weightsBuffer, weightsOffset + desc.networkWeightOffsets.x, scaleBiasOffset, false, networkInputs, hiddenOutput1);
    // Advance scaleBiasOffset to point at the next layer - it's here as a workaround for a Slang bug
    // that prevents it from compiling EvaluateLayer_CoopVec with scaleBiasOffset as 'inout' parameter.
    scaleBiasOffset += Params::HIDDEN_LAYER_CHANNELS * sizeof(float16_t);
    
    // Hidden layer 1
    CoopVec<float16_t, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput2;
    NtcEvaluateLayer_CoopVec_FP8<Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (weightsBuffer, weightsOffset + desc.networkWeightOffsets.y, scaleBiasOffset, false, hiddenOutput1, hiddenOutput2);
    scaleBiasOffset += Params::HIDDEN_LAYER_CHANNELS * sizeof(float16_t);
    
    // Hidden layer 2
    CoopVec<float16_t, Params::HIDDEN_LAYER_CHANNELS> hiddenOutput3;
    NtcEvaluateLayer_CoopVec_FP8<Params::HIDDEN_LAYER_CHANNELS, Params::HIDDEN_LAYER_CHANNELS, true>
    (weightsBuffer, weightsOffset + desc.networkWeightOffsets.z, scaleBiasOffset, true, hiddenOutput2, hiddenOutput3);
    scaleBiasOffset += Params::HIDDEN_LAYER_CHANNELS * sizeof(float16_t);
    
    // Output layer
    CoopVec<float, Params::OUTPUT_CHANNELS> networkOutputs;
    NtcEvaluateOutputLayer_CoopVec_FP8<Params::HIDDEN_LAYER_CHANNELS, Params::OUTPUT_CHANNELS>
    (weightsBuffer, weightsOffset + desc.networkWeightOffsets.w, scaleBiasOffset, hiddenOutput3, networkOutputs);

    [unroll]
    for (int ch = 0; ch < Params::OUTPUT_CHANNELS; ++ch)
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