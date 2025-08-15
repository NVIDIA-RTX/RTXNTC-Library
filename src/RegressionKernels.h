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
#include "tin/tin_mlp.h"
#include "tin/tin_reducer.h"
#include "FeatureGridDevice.h"
#include "CudaUtils.h"
#include <libntc/ntc.h>

// NVCC says "expression has no effect" on the calls to FeatureGrid::Sample and SampleBackward.
// It also says "variable "u" was set but never used".
// Since the code actually works, suppress the messages.
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 174
#pragma nv_diag_suppress 550
#endif

namespace ntc::cuda
{

using Activation = tin::ActHGELUClamp;

template<int NETWORK_VERSION>
struct NTC_PARAMS
{   
    static const int INPUT_CHANNELS = 
        (NETWORK_VERSION == NTC_NETWORK_SMALL) ? NTC_MLP_INPUT_CHANNELS_SMALL :
        (NETWORK_VERSION == NTC_NETWORK_MEDIUM) ? NTC_MLP_INPUT_CHANNELS_MEDIUM :
        (NETWORK_VERSION == NTC_NETWORK_LARGE) ? NTC_MLP_INPUT_CHANNELS_LARGE :
        (NETWORK_VERSION == NTC_NETWORK_XLARGE) ? NTC_MLP_INPUT_CHANNELS_XLARGE :
        0; // Unsupported value

    static const int HR_FEATURES = 
        (NETWORK_VERSION == NTC_NETWORK_SMALL) ? NTC_MLP_HR_FEATURES_SMALL :
        (NETWORK_VERSION == NTC_NETWORK_MEDIUM) ? NTC_MLP_HR_FEATURES_MEDIUM :
        (NETWORK_VERSION == NTC_NETWORK_LARGE) ? NTC_MLP_HR_FEATURES_LARGE :
        (NETWORK_VERSION == NTC_NETWORK_XLARGE) ? NTC_MLP_HR_FEATURES_XLARGE :
        0; // Unsupported value

    static const int LR_FEATURES = NTC_MLP_LR_FEATURES;

    static const int SAMPLED_FEATURES_HR = HR_FEATURES * 4;
    static const int SAMPLED_FEATURES_LR = LR_FEATURES;
    static const int SAMPLED_FEATURES_TOTAL = SAMPLED_FEATURES_HR + SAMPLED_FEATURES_LR;

    static const int HIDDEN_LAYER_CHANNELS = NTC_MLP_HIDDEN_CHANNELS;

    static const int OUTPUT_CHANNELS = NTC_MLP_OUTPUT_CHANNELS;
};

#define REGRESSION_KERNEL_IMPL(NAME, NETWORK_VERSION, STABLE_GRAD) \
    namespace ntc::cuda { \
    __global__ void RegressionKernel_##NAME##_stable##STABLE_GRAD(RegressionKernelParams params) \
    {   RegressionKernel<NETWORK_VERSION, STABLE_GRAD>(params); } }


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

template<int CH>
__device__ void EncodeSamplePosition(float xf, float yf, float lod, int offset, tin::HArray<CH>& m_i) {

    int idx = offset;
    
#pragma unroll
    for (int scale = NTC_MLP_POS_ENC_SCALE; scale > 1; scale >>= 1) {
        float2 enc;
        enc.x = frac(xf / scale) * 4;
        enc.x = abs(enc.x - 2) - 1;
        enc.y = frac(yf / scale) * 4;
        enc.y = abs(enc.y - 2) - 1;

        m_i.set_packed_element(__float22half2_rn(enc), idx);
        idx++;

        enc.x = frac(xf / scale + 0.25f) * 4;
        enc.x = abs(enc.x - 2) - 1;
        enc.y = frac(yf / scale + 0.25f) * 4;
        enc.y = abs(enc.y - 2) - 1;

        m_i.set_packed_element(__float22half2_rn(enc), idx);
        idx++;
    }

    half2 lodh = __float2half2_rn(lod);
    m_i.set_packed_element(lodh, idx);
}

template<int NETWORK_VERSION, bool STABLE_GRADIENTS>
__device__ void RegressionKernel(RegressionKernelParams const params)
{
    using GRID_GRAD_TYPE = std::conditional_t<STABLE_GRADIENTS, float, half>;
    using NW_GRAD_TYPE = std::conditional_t<STABLE_GRADIENTS, float, half>;
    constexpr bool NW_GRAD_ATOMICS = !STABLE_GRADIENTS;
    constexpr tin::ReducerUpdateMode REDUCE_MODE = NW_GRAD_ATOMICS ? tin::ReducerUpdateMode::ATOMIC_ADD : tin::ReducerUpdateMode::STORE;

    using namespace cooperative_groups;
    grid_group grid = this_grid();
    thread_block threadBlock = this_thread_block();
    const auto tile32 = tiled_partition<tin::WarpSize>(threadBlock);
    const int threadInWarp = tile32.thread_rank();
    const int warpIndex = tile32.meta_group_rank();
    
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
    const half* highResLatents = params.latents + mipInfo.latentsOffsetHighRes;
    const half* lowResLatents = params.latents + mipInfo.latentsOffsetLowRes;
    GRID_GRAD_TYPE* highResLatentGradients = (GRID_GRAD_TYPE*)params.latentGradients + mipInfo.latentsOffsetHighRes;
    GRID_GRAD_TYPE* lowResLatentGradients = (GRID_GRAD_TYPE*)params.latentGradients + mipInfo.latentsOffsetLowRes;
    NW_GRAD_TYPE* networkGradientsTyped = (NW_GRAD_TYPE*)params.networkGradients;
    
    using PARAMS = NTC_PARAMS<NETWORK_VERSION>;
    FeatureGrid<PARAMS::HR_FEATURES, true> highResFeatureGrid(params.highResFeatures, mipInfo.highResLatentWidth, mipInfo.highResLatentHeight);
    FeatureGrid<PARAMS::LR_FEATURES, false> lowResFeatureGrid(params.lowResFeatures, mipInfo.lowResLatentWidth, mipInfo.lowResLatentHeight);
    using Network = tin::HMLP<NTC_MLP_LAYERS-2, PARAMS::INPUT_CHANNELS, PARAMS::HIDDEN_LAYER_CHANNELS, PARAMS::OUTPUT_CHANNELS,
        Activation, tin::ActNone, REDUCE_MODE, WARPS_PER_TBLOCK * tin::WarpSize, NW_GRAD_TYPE>;

    // Run network
    // 
    // shared memory for weight reduction
    __align__(16)
    __shared__ half weightReductionShared[Network::smem_size()];

    // shared memory for loss reduction
    __shared__ float lossReductionShared[tin::Reducer<float, WARPS_PER_TBLOCK>::sharedmem_size()];

    float lossAccumulator = 0;
    const int pixelsPerBatch = grid.dim_blocks().x * TILE_SIZE_X * TILE_SIZE_Y;
    const float lossNormalization = 1.f / float(pixelsPerBatch);

    // See the comment block in the beginning of TextureSet.cpp for the weight layouts
    const tin::Quantization quantization = params.useFP8Quantization ? tin::Quantization::FP8 : tin::Quantization::Int8;
    Network mlp(params.networkWeights, params.networkWeights + Network::num_weights(), quantization, tin::Quantization::Int8, weightReductionShared,
        networkGradientsTyped, networkGradientsTyped + Network::num_weights());
    
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
        tin::HArray<PARAMS::INPUT_CHANNELS> networkInputsArray(0.f);

        float pixelCenterX = (x + 0.5f);
        float pixelCenterY = (y + 0.5f);
        float u = pixelCenterX / float(referenceWidth);
        float v = pixelCenterY / float(referenceHeight);

        highResFeatureGrid.Sample<PARAMS::INPUT_CHANNELS>(u, v, 0, highResLatents, networkInputsArray);
        lowResFeatureGrid.Sample<PARAMS::INPUT_CHANNELS>(u, v, PARAMS::SAMPLED_FEATURES_HR / 2, lowResLatents, networkInputsArray);

        EncodeSamplePosition<PARAMS::INPUT_CHANNELS>(
            float(x) * mipInfo.positionScale,
            float(y) * mipInfo.positionScale,
            mipInfo.positionLod,
            PARAMS::SAMPLED_FEATURES_TOTAL / 2, networkInputsArray);

        tin::HVector<PARAMS::INPUT_CHANNELS> networkInputsVector(networkInputsArray);

        // Run network
        auto networkOutputsVector = mlp.forward(networkInputsVector);

        tin::HArray<PARAMS::OUTPUT_CHANNELS> networkOutputsArray(networkOutputsVector);

        // Compute loss gradient and store l2 loss
        tin::HArray<PARAMS::OUTPUT_CHANNELS> lossGradientsArray;
        
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
        for (int i = 0; i < PARAMS::OUTPUT_CHANNELS / 2; i++)
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
            float lossGradient0 = 2.f * difference0 * scaledNormalization;
            float lossGradient1 = 2.f * difference1 * scaledNormalization;

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

        tin::HVector<PARAMS::OUTPUT_CHANNELS> lossGradientsVector(lossGradientsArray);

        // Backward pass with loss gradients
        uint32_t grad_offset = NW_GRAD_ATOMICS ? 0 : (Y_ITERS * threadBlock.group_index().x + iteration) * Network::num_params();

        auto backwardOutputsVector = mlp.backward(lossGradientsVector, grad_offset, grad_offset);
        tin::HArray<PARAMS::INPUT_CHANNELS> backwardOutputsArray(backwardOutputsVector);

        // Store latent gradients from backward pass 
        highResFeatureGrid.SampleBackward<PARAMS::INPUT_CHANNELS>(u, v, 0,
            backwardOutputsArray, highResLatentGradients, mipInfo.highResGradientMask);
        lowResFeatureGrid.SampleBackward<PARAMS::INPUT_CHANNELS>(u, v, PARAMS::SAMPLED_FEATURES_HR / 2,
            backwardOutputsArray, lowResLatentGradients, mipInfo.lowResGradientMask);
    }

    // Store the per-group loss into a buffer
    if (params.loss != nullptr && threadInWarp == 0 && warpIndex == 0)
    {
        params.loss[threadBlock.group_index().x] = lossAccumulator * lossNormalization;
    }
}

#define INFERENCE_KERNEL_IMPL(NAME, NETWORK_VERSION) \
    namespace ntc::cuda { \
    __global__ void InferenceKernel_##NAME(InferenceKernelParams params) \
    {   InferenceKernel<NETWORK_VERSION>(params); } }

template<int NETWORK_VERSION>
__device__ void InferenceKernel(InferenceKernelParams const params)
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

    using PARAMS = NTC_PARAMS<NETWORK_VERSION>;
    FeatureGrid<PARAMS::HR_FEATURES, true> highResFeatureGrid(params.highResFeatures, params.highResLatentWidth, params.highResLatentHeight);
    FeatureGrid<PARAMS::LR_FEATURES, false> lowResFeatureGrid(params.lowResFeatures, params.lowResLatentWidth, params.lowResLatentHeight);
    using Network = tin::HMLP<NTC_MLP_LAYERS-2, PARAMS::INPUT_CHANNELS, PARAMS::HIDDEN_LAYER_CHANNELS, PARAMS::OUTPUT_CHANNELS,
        Activation, tin::ActNone, tin::ReducerUpdateMode::ATOMIC_ADD, WARPS_PER_TBLOCK * tin::WarpSize, float>;

    // shared memory for loss reduction
    __shared__ float lossReductionShared[tin::Reducer<float, WARPS_PER_TBLOCK>::sharedmem_size()];
        
    float lossAccumulator[PARAMS::OUTPUT_CHANNELS];
    for (int i = 0; i < PARAMS::OUTPUT_CHANNELS; ++i)
        lossAccumulator[i] = 0.f;

    for (int iteration = 0; iteration < Y_ITERS; iteration++)
    {
        // Set network input
        tin::HArray<PARAMS::INPUT_CHANNELS> networkInputsArray(0.f);

        // Copy input
        float pixelCenterX = (x + 0.5f);
        float pixelCenterY = (y + 0.5f);
        float u = pixelCenterX / float(params.referenceWidth);
        float v = pixelCenterY / float(params.referenceHeight);

        highResFeatureGrid.Sample<PARAMS::INPUT_CHANNELS>(u, v, /* offset = */ 0, params.highResLatents, networkInputsArray);
        lowResFeatureGrid.Sample<PARAMS::INPUT_CHANNELS>(u, v, /* offset = */ PARAMS::SAMPLED_FEATURES_HR / 2, params.lowResLatents, networkInputsArray);
        
        EncodeSamplePosition<PARAMS::INPUT_CHANNELS>(float(x) * params.positionScale, float(y) * params.positionScale,
            params.positionLod, PARAMS::SAMPLED_FEATURES_TOTAL / 2, networkInputsArray);

        tin::HVector<PARAMS::INPUT_CHANNELS> networkInputsVector(networkInputsArray);
        
        // Run network
        // See the comment block in the beginning of TextureSet.cpp for the weight layouts
        const tin::Quantization quantization = params.useFP8Quantization ? tin::Quantization::FP8 : tin::Quantization::Int8;
        Network mlp(params.mlpWeights, params.mlpWeights + Network::num_weights(), quantization, tin::Quantization::Int8);

        auto networkOutputsVector = mlp.forward(networkInputsVector);
        
        tin::HArray<PARAMS::OUTPUT_CHANNELS> networkOutputArray(networkOutputsVector);

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
            for (int i = 0; i < PARAMS::OUTPUT_CHANNELS / 2; i++)
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
            for (int i = 0; i < PARAMS::OUTPUT_CHANNELS / 2; i++)
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
        for (int i = 0; i < PARAMS::OUTPUT_CHANNELS; i++)
        {
            params.outputLoss[i * params.lossItemsPerChannel + blockLinearIndex] = lossAccumulator[i] * lossNormalization;
        }
    }
}

} // namespace ntc::cuda