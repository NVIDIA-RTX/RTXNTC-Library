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

#include "Optimizer.h"
#include "CudaUtils.h"
#include "LatentQuantization.h"
#include "tin/tin_reducer.h"
#include <cooperative_groups.h>

namespace ntc::cuda
{

const int OPT_WG_SIZE = 128;
const bool SANITIZE_FLOATS = true;

struct AdamConstants
{
    float invLossScale;
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float stepSize;
    float invSqrtBiasCorrection2;
    float step;
    float rmin;
    float rmax;
};

// Pre-computes various constants that are used by AdamKernel
AdamConstants PrepareAdamConstants(
    int   quantizationBits,
    float lossScale,
    float currentStep,
    float learningRate,
    float beta1,
    float beta2,
    float epsilon)
{
    AdamConstants constants{};

    constants.learningRate = learningRate;
    constants.beta1 = beta1;
    constants.beta2 = beta2;
    constants.epsilon = epsilon;

    constants.invLossScale = 1.f / lossScale;
    const float biasCorrection1 = 1.f - powf(beta1, currentStep);
    const float biasCorrection2 = 1.f - powf(beta2, currentStep);
    const float invBiasCorrection1 = 1.f / biasCorrection1;
    constants.stepSize = learningRate * invBiasCorrection1;
    constants.invSqrtBiasCorrection2 = 1.f / sqrtf(biasCorrection2);
    
    if (quantizationBits > 0)
    {
        QuantizationParameters const quantizationParams = GetLatentQuantization(quantizationBits);
        
        constants.step = quantizationParams.step;
        constants.rmin = round(quantizationParams.qmin) * constants.step;
        constants.rmax = round(quantizationParams.qmax) * constants.step;
    }

    return constants;
}

template<class TD>
__device__ void AdamOptimizerCore(
    half* __restrict__ baseWeights,
    half* __restrict__ quantizedWeights,
    TD* __restrict__ gradients,
    float* __restrict__ moments1,
    float* __restrict__ moments2,

    int weightIndex,
    uint32_t randomSeed,
    AdamConstants constants
)
{
    float gradient = float(gradients[weightIndex]) * constants.invLossScale;
    if (gradient == 0.f)
        return;

    const float inputWeight = baseWeights[weightIndex];

    const float gradientSquared = gradient * gradient;

    const float moment1 = moments1[weightIndex] = constants.beta1 * moments1[weightIndex] + (1.f - constants.beta1) * gradient;
    const float moment2 = moments2[weightIndex] = constants.beta2 * moments2[weightIndex] + (1.f - constants.beta2) * gradientSquared;
    
    const float denom = sqrtf(moment2) * constants.invSqrtBiasCorrection2 + constants.epsilon;
    
    float newWeight = inputWeight - (moment1 / denom) * constants.stepSize;
    half newWeightQuantized = half(newWeight);

    if (constants.step != 0.f)
    {
        newWeight = std::min(std::max(newWeight, constants.rmin), constants.rmax);
        
        HashBasedRNG rng(randomSeed, weightIndex);
        const float noise = rng.NextFloat() - 0.5f;

        float weight = newWeight + noise * constants.step;

        newWeightQuantized = half(weight);
    }

    if constexpr (SANITIZE_FLOATS)
    {
        if (IsFloatSpecial(newWeight)) newWeight = 0;
        if (IsHalfSpecial(newWeightQuantized)) newWeightQuantized = 0;
    }

    gradients[weightIndex] = 0;
    baseWeights[weightIndex] = newWeight;
    if (quantizedWeights)
        quantizedWeights[weightIndex] = newWeightQuantized;
}

template<class TD>
__global__ void NetworkAdamKernel(
    int       dispatchSize,
    half* __restrict__ baseWeights,
    half* __restrict__ quantizedWeights,
    TD* __restrict__ gradients,
    float* __restrict__ moments1,
    float* __restrict__ moments2,

    uint32_t randomSeed,
    AdamConstants constants)
{
    using namespace cooperative_groups;

    grid_group grid = this_grid();
    int i = grid.thread_rank();
    if (i < dispatchSize)
    {
        AdamOptimizerCore<TD>(baseWeights, quantizedWeights, gradients, moments1, moments2, i, randomSeed, constants);
    }
}

template<class TD>
__global__ void LatentAdamKernel(
    int numLatentPixels,
    int numFeatures,
    half* __restrict__ baseWeights,
    half* __restrict__ quantizedWeights,
    TD* __restrict__ gradients,
    float* __restrict__ moments1,
    float* __restrict__ moments2,
    uint32_t const* __restrict__ gradientMask,

    uint32_t randomSeed,
    AdamConstants constants)
{
    using namespace cooperative_groups;

    grid_group grid = this_grid();
    int i = grid.thread_rank();

    if (i >= numLatentPixels)
        return;

    uint32_t const mask = gradientMask[i >> 5];
    if ((mask & (1u << (i & 31))) == 0)
        return;
        
    for (int feature = 0; feature < numFeatures; feature += 2)
    {
        AdamOptimizerCore<TD>(baseWeights, quantizedWeights, gradients,
            moments1, moments2, i * 2, randomSeed, constants);

        AdamOptimizerCore<TD>(baseWeights, quantizedWeights, gradients,
            moments1, moments2, i * 2 + 1, randomSeed, constants);

        i += numLatentPixels;
    }
}

void OptimizeNetwork(
    int       dispatchSize,
    bool      useFloatGradients,
    half*     __restrict__ baseWeights,
    half*     __restrict__ quantizedWeights,
    void*     __restrict__ gradients,
    float*    __restrict__ moments1,
    float*    __restrict__ moments2,
    
    float     lossScale,
    float     currentStep,
    uint32_t  randomSeed,
    float     learningRate,
    float     beta1,
    float     beta2,
    float     epsilon)
{
    int threadBlockSize = OPT_WG_SIZE;
    int gridSize = (dispatchSize + threadBlockSize - 1) / threadBlockSize;

    AdamConstants constants = PrepareAdamConstants(0, lossScale, currentStep, learningRate, beta1, beta2, epsilon);
    
    if (useFloatGradients)
        NetworkAdamKernel<float> <<< gridSize, threadBlockSize >>> (dispatchSize, baseWeights, quantizedWeights,
            (float*)gradients, moments1, moments2, randomSeed, constants);
    else
        NetworkAdamKernel<half> <<< gridSize, threadBlockSize >>> (dispatchSize, baseWeights, quantizedWeights,
            (half*)gradients, moments1, moments2, randomSeed, constants);
}

template<class TD>
__global__ void ReduceNetworkGradKernel(
    int       numGrads,
    int       numSlices,
    TD* __restrict__ gradients)
{
    using namespace cooperative_groups;

    grid_group grid = this_grid();
    int i = grid.thread_rank();
    if (i < numGrads)
    {
        float acc = 0;
        for (int k = 0; k < numSlices; k++)
        {
            acc += float(gradients[k * numGrads + i]);
        }
        
        gradients[i] = acc;
    }
}

void ReduceNetworkGrad(
    int       numGrads,
    int       numSlices,
    bool      useFloatGradients,
    void* __restrict__ gradients)
{
    int threadBlockSize = 32;
    int gridSize = (numGrads + threadBlockSize - 1) / threadBlockSize;
    if (useFloatGradients)
        ReduceNetworkGradKernel<float> <<< gridSize, threadBlockSize >>> (numGrads, numSlices, (float*)gradients);
    else
        ReduceNetworkGradKernel<half> <<< gridSize, threadBlockSize >>> (numGrads, numSlices, (half*)gradients);
}

void OptimizeLatentGrid(
    int         numLatents,
    int         numFeatures,
    int         quantizationBits,
    bool        useFloatGradients,
    half*       __restrict__ baseWeights,
    half*       __restrict__ quantizedWeights,
    void*       __restrict__ gradients,
    float*      __restrict__ moments1,
    float*      __restrict__ moments2,
    uint32_t const* __restrict__ gradientMask,

    float     lossScale,
    float     currentStep,
    uint32_t  randomSeed,
    float     learningRate,
    float     beta1,
    float     beta2,
    float     epsilon)
{
    int numLatentPixels = numLatents / numFeatures;
    int dispatchSize = numLatentPixels;
    int threadBlockSize = OPT_WG_SIZE;
    int gridSize = (dispatchSize + threadBlockSize - 1) / threadBlockSize;

    AdamConstants constants = PrepareAdamConstants(quantizationBits, lossScale, currentStep, learningRate, beta1, beta2, epsilon);
    
    if (useFloatGradients)
        LatentAdamKernel<float> <<< gridSize, threadBlockSize >>> (numLatentPixels, numFeatures,
            baseWeights, quantizedWeights, (float*)gradients, moments1, moments2, gradientMask, randomSeed, constants);
    else
        LatentAdamKernel<half> <<< gridSize, threadBlockSize >>> (numLatentPixels, numFeatures,
            baseWeights, quantizedWeights, (half*)gradients, moments1, moments2, gradientMask, randomSeed, constants);
}

__global__ void FreezeQuantizationKernel(
    int       dispatchSize,
    const half* __restrict__ baseWeights,
    half*  __restrict__ quantizedWeights,
    int quantizationBits)
{
    using namespace cooperative_groups;

    grid_group grid = this_grid();
    int i = grid.thread_rank();

    if (quantizationBits > 0)
    {
        if (i < dispatchSize)
        {
            QuantizationParameters const quantizationParams = GetLatentQuantization(quantizationBits);

            const float inputWeight = baseWeights[i];
            
            float weight = inputWeight * quantizationParams.scale;
            weight = std::min(std::max(weight, quantizationParams.qmin), quantizationParams.qmax);
            weight = round(weight) * quantizationParams.step;
            quantizedWeights[i] = half(weight);
        }
    }
}

void FreezeQuantization(
    int       dispatchSize,
    int       quantizationBits,
    half* __restrict__ baseWeights,
    half*  __restrict__ quantizedWeights)
{
    int threadBlockSize = OPT_WG_SIZE;
    int gridSize = (dispatchSize + threadBlockSize - 1) / threadBlockSize;
    
    FreezeQuantizationKernel <<< gridSize, threadBlockSize >>> (dispatchSize, baseWeights, quantizedWeights, quantizationBits);
}

__global__ void LossReductionKernel(
    int inputSize,
    float const* __restrict__ input,
    float* __restrict__ output)
{
    using namespace cooperative_groups;

    grid_group grid = this_grid();
    uint32_t const threadIdx = grid.thread_rank();

    static_assert(LOSS_ITEMS_PER_THREAD == 4, "LOSS_ITEMS_PER_THREAD is supposed to be 4 to use float4 loads");
    float4 const* input4 = reinterpret_cast<float4 const*>(input);
    uint32_t const baseIdx = threadIdx * LOSS_ITEMS_PER_THREAD;

    float acc = 0;
    if (baseIdx < inputSize)
    {
        float4 items = input4[threadIdx];

        acc = items.x;
        if (baseIdx + 1 < inputSize) acc += items.y;
        if (baseIdx + 2 < inputSize) acc += items.z;
        if (baseIdx + 3 < inputSize) acc += items.w;
    }

    typedef tin::Reducer<float, LOSS_GROUP_SIZE / tin::WarpSize> Reducer;
    __shared__ float reductionMem[Reducer::sharedmem_size()];
    acc = Reducer::sum(reductionMem, acc);

    output[grid.block_rank()] = acc;
}

cudaError_t ReduceLoss(size_t size, float* __restrict__ loss, DeviceAndHostArray<float>& scratch, float& outReducedLoss)
{
    int threadBlockSize = LOSS_GROUP_SIZE;
    size_t gridSize = (size + LOSS_ITEMS_PER_GROUP - 1) / LOSS_ITEMS_PER_GROUP;

    if (size_t(gridSize) > scratch.Length())
        return cudaErrorInvalidValue; // This should not happen, but checking just in case

    // Reduce the long input array to a short array on the GPU
    LossReductionKernel <<< gridSize, threadBlockSize >>> (size, loss, scratch.DevicePtr());

    // Copy the short array to the CPU
    cudaError_t err = scratch.CopyToHost(gridSize);
    if (err != cudaSuccess)
        return err;
    
    // Reduce the short array on the CPU using double to avoid precision loss from serial reduction
    double acc = 0.0;
    for (int idx = 0; idx < gridSize; ++idx)
        acc += double(scratch.HostPtr()[idx]);
    
    outReducedLoss = float(acc);

    return cudaSuccess;
}

} // namespace ntc::cuda