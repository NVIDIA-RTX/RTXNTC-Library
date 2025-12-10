/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "RegressionKernels.h"

namespace ntc::cuda
{

__global__ void InferenceKernel(InferenceKernelParams const params)
{
    using namespace cooperative_groups;

    grid_group grid = this_grid();
    thread_block threadBlock = this_thread_block();

    const auto tile32 = tiled_partition<tin::WarpSize>(threadBlock);
    const int threadInWarp = tile32.thread_rank();
    const int warpIndex = tile32.meta_group_rank();

    int baseX = threadBlock.group_index().x * TILE_SIZE_X;
    int baseY = threadBlock.group_index().y * TILE_SIZE_Y;

    int x = baseX + threadInWarp;
    int y = baseY + threadBlock.thread_index().y;

    FeatureGrid highResFeatureGrid(params.numFeatures, params.highResLatentWidth, params.highResLatentHeight, params.latentStride);
    FeatureGrid lowResFeatureGrid(params.numFeatures, params.lowResLatentWidth, params.lowResLatentHeight, params.latentStride);
    using Network = MLP<Activation, tin::ReducerUpdateMode::ATOMIC_ADD, WARPS_PER_TBLOCK * tin::WarpSize, float>;

    // shared memory for loss reduction
    __shared__ float lossReductionShared[tin::Reducer<float, WARPS_PER_TBLOCK>::sharedmem_size()];
        
    float lossAccumulator[NTC_MLP_OUTPUT_CHANNELS];
    for (int i = 0; i < NTC_MLP_OUTPUT_CHANNELS; ++i)
        lossAccumulator[i] = 0.f;

    for (int iteration = 0; iteration < Y_ITERS; iteration++)
    {
        // Set network input
        tin::HArray<NTC_MLP_INPUT_CHANNELS> networkInputsArray(0.f);

        // Copy input
        float u = (float(x) + PIXEL_CENTER_OFFSET) / float(params.referenceWidth);
        float v = (float(y) + PIXEL_CENTER_OFFSET) / float(params.referenceHeight);

        highResFeatureGrid.Sample(u, v, params.highResLatents, networkInputsArray, 0);
        lowResFeatureGrid.Sample(u, v, params.lowResLatents, networkInputsArray, NTC_MLP_FEATURES);

        EncodeSamplePosition(float(x) * params.positionScale, float(y) * params.positionScale,
            params.positionLod, NTC_MLP_FEATURES * 2, networkInputsArray);

        tin::HVector<NTC_MLP_INPUT_CHANNELS> networkInputsVector(networkInputsArray);
        
        // Run network
        // See the comment block in the beginning of WeightLayout.cpp for the weight layouts
        const tin::Quantization quantization = params.useFP8Quantization ? tin::Quantization::FP8 : tin::Quantization::Int8;
        Network mlp(params.mlpWeights, params.mlpWeights + MlpDesc::GetTotalWeightCount(), quantization, tin::Quantization::Int8);

        auto networkOutputsVector = mlp.forward(networkInputsVector);
        
        tin::HArray<NTC_MLP_OUTPUT_CHANNELS> networkOutputArray(networkOutputsVector);

        // Check whether this texel is inside the reference image
        const bool insideReferenceImage = x >= 0 && x < params.referenceWidth && y >= 0 && y < params.referenceHeight;
        // When attemping to sample outside bounds we simply use the first element in the array
        const uint64_t pixelBaseAddress = insideReferenceImage ? GetPixelBaseAddress(x, y, params.referenceWidth, params.numChannels) : 0;
        
        if (params.validChannelMask != 0)
        {
            bool isMaskedOut = false;
            if (params.maskChannelIndex >= 0 && params.discardMaskedOutPixels)
            {
                const half maskValue = params.referenceImage[GetChannelAddress(pixelBaseAddress, params.maskChannelIndex, params.referenceWidth)];
                isMaskedOut = maskValue == half(0);
            }

#pragma unroll
            for (int i = 0; i < NTC_MLP_OUTPUT_CHANNELS / 2; i++)
            {
                half2 outputs = networkOutputArray.get_packed_element(i);
                if (IsHalfSpecial(outputs.x)) outputs.x = 0;
                if (IsHalfSpecial(outputs.y)) outputs.y = 0;

                outputs.x = half(float(outputs.x) * g_ChannelInfo[i * 2 + 0].optimalToLinearScale + g_ChannelInfo[i * 2 + 0].optimalToLinearBias);
                outputs.y = half(float(outputs.y) * g_ChannelInfo[i * 2 + 1].optimalToLinearScale + g_ChannelInfo[i * 2 + 1].optimalToLinearBias);
                
                if (params.maskChannelIndex == i * 2 + 0) outputs.x = max(0.f, min(1.f, outputs.x));
                if (params.maskChannelIndex == i * 2 + 1) outputs.y = max(0.f, min(1.f, outputs.y));

                const half2 reference = (i * 2 < params.numChannels)
                    ? *(const half2*)(params.referenceImage + GetChannelAddress(pixelBaseAddress, i * 2, params.referenceWidth))
                    : half2{0, 0};

                float dx = (float(outputs.x) - float(reference.x)) * float(GetBit(params.validChannelMask, i * 2));
                float dy = (float(outputs.y) - float(reference.y)) * float(GetBit(params.validChannelMask, i * 2 + 1));

                // For a masked out pixel, zero the loss function on all channels except the mask channel
                if (isMaskedOut && params.maskChannelIndex != i * 2 + 0) dx = 0;
                if (isMaskedOut && params.maskChannelIndex != i * 2 + 1) dy = 0;

                if (insideReferenceImage && (i * 2 < params.numChannels))
                {
                    *(half2*)(params.outputImage + GetChannelAddress(pixelBaseAddress, i * 2, params.referenceWidth)) = outputs;
                    
                    lossAccumulator[i * 2 + 0] += tin::Reducer<float, WARPS_PER_TBLOCK>::sum(lossReductionShared, dx * dx);
                    lossAccumulator[i * 2 + 1] += tin::Reducer<float, WARPS_PER_TBLOCK>::sum(lossReductionShared, dy * dy);
                }
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NTC_MLP_OUTPUT_CHANNELS / 2; i++)
            {
                half2 outputs = networkOutputArray.get_packed_element(i);
                if (IsHalfSpecial(outputs.x)) outputs.x = 0;
                if (IsHalfSpecial(outputs.y)) outputs.y = 0;

                outputs.x = half(float(outputs.x) * g_ChannelInfo[i * 2 + 0].optimalToLinearScale + g_ChannelInfo[i * 2 + 0].optimalToLinearBias);
                outputs.y = half(float(outputs.y) * g_ChannelInfo[i * 2 + 1].optimalToLinearScale + g_ChannelInfo[i * 2 + 1].optimalToLinearBias);

                if (insideReferenceImage && (i * 2 < params.numChannels))
                {
                    *(half2*)(params.outputImage + GetChannelAddress(pixelBaseAddress, i * 2, params.referenceWidth)) = outputs;
                }
            }
        }
        
        // Move on to the next iteration / pixel
        y += TB_SIZE_Y;
    }

    if (threadInWarp == 0 && warpIndex == 0)
    {
        const int validThreadsInGrid = std::min(grid.dim_blocks().x * TILE_SIZE_X, (unsigned)params.referenceWidth) * 
                                       std::min(grid.dim_blocks().y * TILE_SIZE_Y, (unsigned)params.referenceHeight);

        const float lossNormalization = 1.f / float(validThreadsInGrid);

        int blockLinearIndex = threadBlock.group_index().y * grid.dim_blocks().x + threadBlock.group_index().x;

        // Write out per-channel loss
        for (int i = 0; i < NTC_MLP_OUTPUT_CHANNELS; i++)
        {
            params.outputLoss[i * params.lossItemsPerChannel + blockLinearIndex] = lossAccumulator[i] * lossNormalization;
        }
    }
}

void Inference(InferenceKernelParams const& params)
{
    auto threadBlockSize = dim3(TB_SIZE_X, TB_SIZE_Y, 1);
    auto gridSize = DivRoundUp(dim3(params.referenceWidth, params.referenceHeight, 1), threadBlockSize);

    InferenceKernel<<<gridSize, threadBlockSize>>>(params);
}

} // namespace ntc::cuda
