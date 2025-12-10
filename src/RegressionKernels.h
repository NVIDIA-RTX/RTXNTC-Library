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

#pragma once

#include "RegressionCommon.h"
#include "FeatureGridDevice.h"
#include "CudaMLP.h"
#include "CudaUtils.h"
#include <libntc/ntc.h>
#include "tin/tin_reducer.h"

namespace ntc::cuda
{

static constexpr float PIXEL_CENTER_OFFSET = 0.5f;

using Activation = tin::ActHGELUClamp;

static const int LATENT_ARRAY_SIZE = NTC_MLP_FEATURES / NTC_FEATURES_PER_LAYER;

extern __constant__ MipInfo g_MipInfo[NTC_MAX_MIPS];
extern __constant__ ChannelInfo g_ChannelInfo[NTC_MAX_CHANNELS];

// Computes the address (offset into the textureData array for one mip) for a given pixel in the texture data.
// See the comment to PitchLinearImageSlice structure for the texture data layout explanation.
inline __device__ uint64_t GetPixelBaseAddress(int x, int y, int width, int numChannels)
{
    return uint64_t(y) * uint64_t(width) * uint64_t(numChannels) + uint64_t(x) * 2;
}

// Computes the address (offset into the textureData array for one mip) for a channel in the pixel.
inline __device__ uint64_t GetChannelAddress(uint64_t pixelBaseAddress, int channel, int width)
{
    return pixelBaseAddress + uint64_t(channel & ~1) * uint64_t(width) + uint64_t(channel & 1);
}

// Shifts the 0.0 and 1.0 values slightly outside of the 0-1 range to make sure that
// after lossy compression, decompression, and clamping the output values will still be 0.0 and 1.0.
inline __device__ half ExpandMaskChannel(half value)
{
    half const expansion = half(0.125f);
    if (value <= half(0.f))
        return half(-expansion);
    if (value >= half(1.f))
        return half(1.f) + expansion;
    return value;
}

inline __device__ void EncodeSamplePosition(float xf, float yf, float lod, int offset, tin::HArray<NTC_MLP_INPUT_CHANNELS>& m_i)
{
    int idx = offset / 2;
    
#pragma unroll
    for (int wave = 0; wave < NTC_MLP_POS_ENC_WAVES; ++wave)
    {
        float2 enc;
        enc.x = frac(xf) * 2 - 1;
        enc.y = frac(yf) * 2 - 1;

        m_i.set_packed_element(__float22half2_rn(enc), idx);
        idx++;

        enc.x = frac(xf + 0.25f) * 2 - 1;
        enc.y = frac(yf + 0.25f) * 2 - 1;

        m_i.set_packed_element(__float22half2_rn(enc), idx);
        idx++;

        xf *= 2.f;
        yf *= 2.f;
    }

    half2 lodh = __float2half2_rn(lod);
    m_i.set_packed_element(lodh, idx);
}

// This is the main NTC training kernel.
// It is implemented as a template to allow easy switching between stable and fast training modes, which differ
// in the formats used for gradients (float32 vs float16) and in the way network weight gradients are accumulated.
// The actual kernel instantiations are defined in RegressionFast.cu and RegressionStable.cu - to allow parallel compilation.
template<bool STABLE_GRADIENTS>
__device__ void RegressionKernel(RegressionKernelParams const params)
{
    using GRID_GRAD_TYPE = std::conditional_t<STABLE_GRADIENTS, float, half>;
    using NW_GRAD_TYPE = std::conditional_t<STABLE_GRADIENTS, float, half>;
    constexpr bool NW_GRAD_ATOMICS = !STABLE_GRADIENTS;
    constexpr tin::ReducerUpdateMode REDUCE_MODE = NW_GRAD_ATOMICS ? tin::ReducerUpdateMode::ATOMIC_ADD : tin::ReducerUpdateMode::STORE;

    using namespace cooperative_groups;
    grid_group grid = this_grid();
    thread_block threadBlock = this_thread_block();
    const int threadInGroup = threadBlock.thread_rank();
    const auto tile32 = tiled_partition<tin::WarpSize>(threadBlock);
    const int threadInWarp = tile32.thread_rank();
    
    HashBasedRNG rng(threadBlock.group_index().x * TB_SIZE_Y + threadBlock.thread_index().y, params.randomSeed);

    // Select a reference mip level to sample
    const float randomForMipSelection = rng.NextFloat();

    int mip;
    if constexpr (true) // selection between parallel and reference versions of the mip selection code
    {
        // Parallel CDF inversion using warp intrinsics
        if (threadInWarp < params.numMips)
        {
            bool less = randomForMipSelection < g_MipInfo[threadInWarp].cdf;
            uint32_t lessMask = __ballot_sync(__activemask(), less);
            mip = __ffs(lessMask) - 1;
            mip = std::max(0, std::min(params.numMips - 1, mip));
        }
        mip = __shfl_sync(~0u, mip, 0);
    }
    else
    {
        // Brute-force linear CDF inversion
        for (mip = 0; mip < params.numMips; ++mip)
        {
            if (randomForMipSelection < g_MipInfo[mip].cdf)
                break;
        }
    }

    // Derive the addressing parameters for this mip level
    MipInfo const& mipInfo = g_MipInfo[mip];
    const int referenceWidth = std::max(params.referenceWidth >> mip, 1);
    const int referenceHeight = std::max(params.referenceHeight >> mip, 1);

    // Shift the base pointers to this mip level
    const half* referenceImage = params.referenceImage + mipInfo.referenceTextureOffset * params.numChannels;
    const half* highResLatents = params.latents + mipInfo.highResLatentOffset;
    const half* lowResLatents = params.latents + mipInfo.lowResLatentOffset;
    GRID_GRAD_TYPE* highResLatentGradients = (GRID_GRAD_TYPE*)params.latentGradients + mipInfo.highResLatentOffset;
    GRID_GRAD_TYPE* lowResLatentGradients = (GRID_GRAD_TYPE*)params.latentGradients + mipInfo.lowResLatentOffset;
    NW_GRAD_TYPE* networkGradientsTyped = (NW_GRAD_TYPE*)params.networkGradients;
    
    FeatureGrid highResFeatureGrid(params.numFeatures, mipInfo.highResLatentWidth, mipInfo.highResLatentHeight, params.latentStride);
    FeatureGrid lowResFeatureGrid(params.numFeatures, mipInfo.lowResLatentWidth, mipInfo.lowResLatentHeight, params.latentStride);
    using Network = MLP<Activation, REDUCE_MODE, WARPS_PER_TBLOCK * tin::WarpSize, NW_GRAD_TYPE>;

    // Run network
    // 
    // shared memory for weight reduction
    __align__(16)
    __shared__ half weightReductionShared[Network::GetSharedMemorySize()];

    // shared memory for loss reduction
    __shared__ float lossReductionShared[tin::Reducer<float, WARPS_PER_TBLOCK>::sharedmem_size()];

    float lossAccumulator = 0;
    const int pixelsPerBatch = grid.dim_blocks().x * TILE_SIZE_X * TILE_SIZE_Y;
    const float lossNormalization = 1.f / float(pixelsPerBatch);

    // See the comment block in the beginning of WeightLayout.cpp for the weight layouts
    const tin::Quantization quantization = params.useFP8Quantization ? tin::Quantization::FP8 : tin::Quantization::Int8;
    Network mlp(params.networkWeights, params.networkWeights + MlpDesc::GetTotalWeightCount(), quantization, tin::Quantization::Int8, weightReductionShared,
        networkGradientsTyped, networkGradientsTyped + MlpDesc::GetTotalWeightCount());
    
    for (int iteration = 0; iteration < Y_ITERS; iteration++)
    {
        const float xOffset = rng.NextFloat();
        const float yOffset = rng.NextFloat();
        
        // Generate the sample position for this thread, so that the warp samples a random 32x1 line in the image
        int x = int(floorf(xOffset * float(referenceWidth))) + threadInWarp;
        int y = int(floorf(yOffset * float(referenceHeight)));

        // Wrap the sampling position to make sure it's inside the reference image
        y += x / referenceWidth;
        x = x % referenceWidth;
        y = y % referenceHeight;

        // Set network input
        tin::HArray<NTC_MLP_INPUT_CHANNELS> networkInputsArray(0.f);

        float u = (float(x) + PIXEL_CENTER_OFFSET) / float(referenceWidth);
        float v = (float(y) + PIXEL_CENTER_OFFSET) / float(referenceHeight);

        highResFeatureGrid.Sample(u, v, highResLatents, networkInputsArray, 0);
        lowResFeatureGrid.Sample(u, v, lowResLatents, networkInputsArray, NTC_MLP_FEATURES);

        EncodeSamplePosition(
            float(x) * mipInfo.positionScale,
            float(y) * mipInfo.positionScale,
            mipInfo.positionLod,
            NTC_MLP_FEATURES * 2, networkInputsArray);
        
        tin::HVector<NTC_MLP_INPUT_CHANNELS> networkInputsVector(networkInputsArray);

        // Run network
        auto networkOutputsVector = mlp.forward(networkInputsVector);

        tin::HArray<NTC_MLP_OUTPUT_CHANNELS> networkOutputsArray(networkOutputsVector);

        // Compute loss gradient and store l2 loss
        tin::HArray<NTC_MLP_OUTPUT_CHANNELS> lossGradientsArray;
        
        float localLoss = 0;

        const uint64_t pixelBaseAddress = GetPixelBaseAddress(x, y, referenceWidth, params.numChannels);
        
        // If mask channel is enabled, determine if this pixel is masked out and thus irrelevant
        bool isMaskedOut = false;
        if (params.maskChannelIndex >= 0 && params.discardMaskedOutPixels)
        {
            const half maskValue = referenceImage[GetChannelAddress(pixelBaseAddress, params.maskChannelIndex, referenceWidth)];
            isMaskedOut = maskValue == half(0);
        }

        // Compute loss and loss gradient
#pragma unroll
        for (int i = 0; i < NTC_MLP_OUTPUT_CHANNELS / 2; i++)
        {
            half2 outputs = networkOutputsArray.get_packed_element(i);

            // de-normalize network output. All network activations before this point are therefore inherently normalized.
            outputs.x = half(float(outputs.x) * g_ChannelInfo[i * 2 + 0].optimalToLinearScale + g_ChannelInfo[i * 2 + 0].optimalToLinearBias);
            outputs.y = half(float(outputs.y) * g_ChannelInfo[i * 2 + 1].optimalToLinearScale + g_ChannelInfo[i * 2 + 1].optimalToLinearBias);

            half2 reference = (i * 2 < params.numChannels) ? *(const half2*)(referenceImage + GetChannelAddress(pixelBaseAddress, i * 2, referenceWidth)) : half2{0, 0};

            // Expand the alpha mask channel's values to make 0 and 1 more accurate.
            if (params.maskChannelIndex == i * 2 + 0) reference.x = ExpandMaskChannel(reference.x);
            if (params.maskChannelIndex == i * 2 + 1) reference.y = ExpandMaskChannel(reference.y);

            float difference0 = (float(outputs.x) - float(reference.x)) * float(GetBit(params.validChannelMask, i * 2));
            float difference1 = (float(outputs.y) - float(reference.y)) * float(GetBit(params.validChannelMask, i * 2 + 1));

            // For a masked out pixel, zero the loss function on all channels except the mask channel
            if (isMaskedOut && params.maskChannelIndex != i * 2 + 0) difference0 = 0;
            if (isMaskedOut && params.maskChannelIndex != i * 2 + 1) difference1 = 0;

            float scaledNormalization = lossNormalization * params.lossScale;

            localLoss += (difference0 * difference0 + difference1 * difference1);
            float lossGradient0 = 2.f * difference0 * scaledNormalization * g_ChannelInfo[i * 2 + 0].lossFunctionScale;
            float lossGradient1 = 2.f * difference1 * scaledNormalization * g_ChannelInfo[i * 2 + 1].lossFunctionScale;

            // copy loss gradient into a TIN matrix for backprop
            half2 loss_d_h2 = __floats2half2_rn(lossGradient0, lossGradient1);
            lossGradientsArray.set_packed_element(loss_d_h2, i);
        }

        // Reduce loss in the thread group for reporting
        if (params.loss != nullptr)
        {
            if (IsFloatSpecial(localLoss))
                localLoss = 0;

            lossAccumulator += tin::Reducer<float, WARPS_PER_TBLOCK>::sum(lossReductionShared, localLoss);
        }

        tin::HVector<NTC_MLP_OUTPUT_CHANNELS> lossGradientsVector(lossGradientsArray);

        // Backward pass with loss gradients
        uint32_t gradientOffset = NW_GRAD_ATOMICS ? 0 : (Y_ITERS * threadBlock.group_index().x + iteration) 
            * (MlpDesc::GetTotalWeightCount() + MlpDesc::GetTotalOutputCount());

        auto backwardOutputsVector = mlp.backward(lossGradientsVector, gradientOffset);
        tin::HArray<NTC_MLP_INPUT_CHANNELS> backwardOutputsArray(backwardOutputsVector);

        // Store latent gradients from backward pass 
        highResFeatureGrid.template SampleBackward(
            u, v,
            backwardOutputsArray, 0,
            highResLatentGradients,
            params.gradientMask,
            mipInfo.highResMaskOffset);
            
        lowResFeatureGrid.template SampleBackward(
            u, v,
            backwardOutputsArray, NTC_MLP_FEATURES,
            lowResLatentGradients,
            params.gradientMask,
            mipInfo.lowResMaskOffset);
    }

    // Store the per-group loss into a buffer
    if (params.loss != nullptr && threadInGroup == 0)
    {
        params.loss[threadBlock.group_index().x] = lossAccumulator * lossNormalization;
    }
}

} // namespace ntc::cuda
