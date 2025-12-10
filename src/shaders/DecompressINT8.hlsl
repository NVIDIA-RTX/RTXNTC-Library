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
   
// Use manual allocation and addressing for shared memory to share the same memory
// between matrix/scale/bias preloading and output shuffling because these actions
// do not overlap.

// First phase - matrix and scale/bias preloading
static const int MATRIX_B_BASE_ADDR = 0;
static const int MATRIX_B_MEM_SIZE = MAX_OUTPUT_SIZE * (MAX_INPUT_SIZE / 4);
static const int SCALE_BIAS_SIZE = MAX_OUTPUT_SIZE;
static const int BIAS_BASE_ADDR = MATRIX_B_MEM_SIZE;
static const int SCALE_BASE_ADDR = BIAS_BASE_ADDR + SCALE_BIAS_SIZE;

// Second phase - output shuffling
static const int OUTPUT_BASE_ADDR = 0;
static const int OUTPUT_UINTS = NTC_MLP_OUTPUT_CHANNELS/2;
static const int OUTPUT_SIZE = DECOMPRESS_CS_BLOCK_WIDTH * DECOMPRESS_CS_BLOCK_HEIGHT * (OUTPUT_UINTS+1);

// Calculate the total shared memory size and allocate it
static const int SHARED_MEMORY_SIZE = max(OUTPUT_SIZE, MATRIX_B_MEM_SIZE + SCALE_BIAS_SIZE * 2);
groupshared uint s_SharedMem[SHARED_MEMORY_SIZE];

// Shared memory address calculation functions

int GetMatrixBAddress(int col, int row)
{
    return MATRIX_B_BASE_ADDR + col * (MAX_INPUT_SIZE / 4) + row;
}

int GetBiasAddress(int index)
{
    return BIAS_BASE_ADDR + index;
}

int GetScaleAddress(int index)
{
    return SCALE_BASE_ADDR + index;
}

int GetOutputAddress(int ch, int2 threadIdx)
{
    return (threadIdx.y * DECOMPRESS_CS_BLOCK_WIDTH + threadIdx.x) * (OUTPUT_UINTS+1) + ch;
}

template<int IN, int OUT, bool OUT_FLOAT>
void EvaluateLayerINT8_SharedMem(
    int weightOffset,
    int biasOffset,
    int scaleOffset,
    bool activation,
    uint inputArray[IN / 4],
    out uint outputArray[OUT_FLOAT ? OUT / 2 : OUT / 4],
    int2 threadIndex)
{
    GroupMemoryBarrierWithGroupSync();

    // Preload the bias values into shared memory.
    // Note: this 'if' assumes that there are enough threads in the group to load all bias values in one pass.
    // If that ever changes, use a loop like one used for the weights below.
    const int linearThreadIndex = threadIndex.x + threadIndex.y * DECOMPRESS_CS_BLOCK_WIDTH;
    if (linearThreadIndex < OUT)
    {
        float scale = t_WeightBuffer.Load<float>(scaleOffset + linearThreadIndex * sizeof(float));
        int bias = t_WeightBuffer.Load<int>(biasOffset + linearThreadIndex * sizeof(int));

        s_SharedMem[GetScaleAddress(linearThreadIndex)] = asuint(scale);
        s_SharedMem[GetBiasAddress(linearThreadIndex)] = asuint(bias);
    }

    // Preload the weights into shared memory.
    // The weights form a matrix with IN rows and OUT columns, stored in a column-major layout
    // (i.e. elements of a column are continuous.)
    int preloadIndex = linearThreadIndex;
    while (preloadIndex < (IN * OUT) / 4)
    {
        int k = preloadIndex % (IN/4); // row
        int c = preloadIndex / (IN/4); // column
        
        s_SharedMem[GetMatrixBAddress(c, k)] = t_WeightBuffer.Load(weightOffset + preloadIndex * 4);

        preloadIndex += DECOMPRESS_CS_BLOCK_WIDTH * DECOMPRESS_CS_BLOCK_HEIGHT;
    }

    GroupMemoryBarrierWithGroupSync();

    // Note: not unrolling the outer loop.
    // If we do, DXC/SPIR-V crashes.
    // DXC/DXIL compiles the unrolled loop successfully, but then creating a pipeline with it takes seconds,
    // and the resulting code works slower than a regular loop.
    for (uint c = 0; c < OUT; c += 4)
    {
        int acc0 = asint(s_SharedMem[GetBiasAddress(c + 0)]);
        int acc1 = asint(s_SharedMem[GetBiasAddress(c + 1)]);
        int acc2 = asint(s_SharedMem[GetBiasAddress(c + 2)]);
        int acc3 = asint(s_SharedMem[GetBiasAddress(c + 3)]);
        
        [unroll]
        for (uint k = 0; k < IN / 4; k++)
        {
            const int matrixAddr0 = GetMatrixBAddress(c + 0, k);
            const int matrixAddr1 = GetMatrixBAddress(c + 1, k);
            const int matrixAddr2 = GetMatrixBAddress(c + 2, k);
            const int matrixAddr3 = GetMatrixBAddress(c + 3, k);

            acc0 = dot4add_i8packed(inputArray[k], s_SharedMem[matrixAddr0], acc0);
            acc1 = dot4add_i8packed(inputArray[k], s_SharedMem[matrixAddr1], acc1);
            acc2 = dot4add_i8packed(inputArray[k], s_SharedMem[matrixAddr2], acc2);
            acc3 = dot4add_i8packed(inputArray[k], s_SharedMem[matrixAddr3], acc3);
        }

        float4 results = float4(acc0, acc1, acc2, acc3);

        float4 scales;
        scales.x = asfloat(s_SharedMem[GetScaleAddress(c + 0)]);
        scales.y = asfloat(s_SharedMem[GetScaleAddress(c + 1)]);
        scales.z = asfloat(s_SharedMem[GetScaleAddress(c + 2)]);
        scales.w = asfloat(s_SharedMem[GetScaleAddress(c + 3)]);

        results *= scales;

        float16_t4 hresults = float16_t4(results);
        
        if (activation)
        {
            hresults = NtcHGELUClamp_ForwardHalf(hresults);
        }

        if (OUT_FLOAT)
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
        true, networkInputs, hiddenOutput0, threadIndex);

    // Hidden layer 1
    uint hiddenOutput1[NTC_MLP_HIDDEN1_CHANNELS / 4];
    EvaluateLayerINT8_SharedMem<NTC_MLP_HIDDEN0_CHANNELS, NTC_MLP_HIDDEN1_CHANNELS, false>(
        g_Const.networkWeightOffsets.y,
        g_Const.networkBiasOffsets.y,
        g_Const.networkScaleOffsets.y,
        true, hiddenOutput0, hiddenOutput1, threadIndex);

#if NTC_MLP_LAYERS == 4
    // Hidden layer 2
    uint hiddenOutput2[NTC_MLP_HIDDEN2_CHANNELS / 4];
    EvaluateLayerINT8_SharedMem<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_HIDDEN2_CHANNELS, false>(
        g_Const.networkWeightOffsets.z,
        g_Const.networkBiasOffsets.z,
        g_Const.networkScaleOffsets.z,
        true, hiddenOutput1, hiddenOutput2, threadIndex);

    // Output layer
    uint networkOutputs[OUTPUT_UINTS];
    EvaluateLayerINT8_SharedMem<NTC_MLP_HIDDEN2_CHANNELS, NTC_MLP_OUTPUT_CHANNELS, true>(
        g_Const.networkWeightOffsets.w,
        g_Const.networkBiasOffsets.w,
        g_Const.networkScaleOffsets.w,
        false, hiddenOutput2, networkOutputs, threadIndex);
#else
    // Output layer
    uint networkOutputs[OUTPUT_UINTS];
    EvaluateLayerINT8_SharedMem<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_OUTPUT_CHANNELS, true>(
        g_Const.networkWeightOffsets.z,
        g_Const.networkBiasOffsets.z,
        g_Const.networkScaleOffsets.z,
        false, hiddenOutput1, networkOutputs, threadIndex);
#endif

    HashBasedRNG rng = HashBasedRNG::Create(pixelPosition.x + pixelPosition.y * g_Const.imageWidth, 0);

    // Store the outputs into shared memory for efficient indexed access later.
    // Note: there is no need for a barrier after this store because each thread only reads the data
    // it's written - nothing from other threads.
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (int ch = 0; ch < OUTPUT_UINTS; ++ch)
    {
        s_SharedMem[GetOutputAddress(ch, threadIndex)] = networkOutputs[ch];
    }

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
            uint twoCh = s_SharedMem[GetOutputAddress(srcChannel/2, threadIndex)];
            if (srcChannel & 1)
                twoCh >>= 16;
            texelValue[ch] = asfloat16(uint16_t(twoCh));
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

        // Write out the texel to the UAV
        u_Outputs[outputDesc.textureIndex][dstPosition] = texelValue;
    }
}

[numthreads(DECOMPRESS_CS_BLOCK_WIDTH, DECOMPRESS_CS_BLOCK_HEIGHT, 1)]
void main(uint2 globalIndex : SV_DispatchThreadID, uint2 threadIndex : SV_GroupThreadID)
{
    DecompressPixel(globalIndex, threadIndex);
}
