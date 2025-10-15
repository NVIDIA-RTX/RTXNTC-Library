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
#include "FeatureGridMath.h"
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
};

// Pre-computes various constants that are used by AdamKernel
AdamConstants PrepareAdamConstants(
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

    return constants;
}

template<class TD>
__device__ void AdamOptimizerCore(
    half& inOutBaseWeight,
    half& inOutQuantizedWeight,
    TD& inOutGradient,
    float& inOutMoment1,
    float& inOutMoment2,

    HashBasedRNG& rng,
    AdamConstants constants,
    float quantizationStep
)
{
    float gradient = float(inOutGradient) * constants.invLossScale;
    if (gradient == 0.f)
        return;
    
    const float inputWeight = inOutBaseWeight;

    const float gradientSquared = gradient * gradient;

    const float moment1 = inOutMoment1 = constants.beta1 * inOutMoment1 + (1.f - constants.beta1) * gradient;
    const float moment2 = inOutMoment2 = constants.beta2 * inOutMoment2 + (1.f - constants.beta2) * gradientSquared;
    
    const float denom = sqrtf(moment2) * constants.invSqrtBiasCorrection2 + constants.epsilon;
    
    float newWeight = inputWeight - (moment1 / denom) * constants.stepSize;
    
    half newWeightQuantized = half(newWeight);

    if (quantizationStep != 0.f)
    {
        newWeight = std::min(std::max(newWeight, 0.f), 1.f);
        
        const float noise = rng.NextFloat() - 0.5f;

        float weight = newWeight + noise * quantizationStep;
        weight = std::min(std::max(weight, 0.f), 1.f);

        newWeightQuantized = half(weight);
    }

    if constexpr (SANITIZE_FLOATS)
    {
        if (IsFloatSpecial(newWeight)) newWeight = 0;
        if (IsHalfSpecial(newWeightQuantized)) newWeightQuantized = 0;
    }

    inOutBaseWeight = newWeight;
    inOutQuantizedWeight = newWeightQuantized;
    inOutGradient = TD(0);
}

template<class TD>
__global__ void NetworkAdamKernel(
    uint32_t dispatchSize,
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
    uint32_t i = grid.thread_rank();
    if (i < dispatchSize)
    {
        HashBasedRNG rng(randomSeed, i);

        AdamOptimizerCore<TD>(baseWeights[i], quantizedWeights[i], gradients[i], moments1[i], moments2[i],
            rng, constants, 0.f);
    }
}

template<class TD, class TD2>
__global__ void LatentAdamKernel(
    uint32_t numPixels,
    size_t latentStride,
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
    uint32_t pixelIndex = grid.thread_index().x;
    uint32_t featureGroupIndex = grid.thread_index().y;
    
    if (pixelIndex >= numPixels)
        return;

    uint32_t const mask = gradientMask[pixelIndex >> 5];
    if ((mask & (1u << (pixelIndex & 31))) == 0)
        return;

    size_t latentIndex = size_t(pixelIndex) * FeatureGridMath::FeaturesPerGroup + featureGroupIndex * latentStride;
    float const quantizationStep = 1.f / 15.f;

    TD2 gradientPair = *reinterpret_cast<TD2*>(gradients + latentIndex);
    if (gradientPair.x == TD(0) && gradientPair.y == TD(0))
        return;

    half2 baseWeightPair = *reinterpret_cast<half2*>(baseWeights + latentIndex);
    half2 quantizedWeightPair = *reinterpret_cast<half2*>(quantizedWeights + latentIndex);
    float2 moment1Pair = *reinterpret_cast<float2*>(moments1 + latentIndex);
    float2 moment2Pair = *reinterpret_cast<float2*>(moments2 + latentIndex);

    HashBasedRNG rng(randomSeed, latentIndex);

    AdamOptimizerCore<TD>(baseWeightPair.x, quantizedWeightPair.x, gradientPair.x,
        moment1Pair.x, moment2Pair.x, rng, constants, quantizationStep);

    AdamOptimizerCore<TD>(baseWeightPair.y, quantizedWeightPair.y, gradientPair.y,
        moment1Pair.y, moment2Pair.y, rng, constants, quantizationStep);

    *reinterpret_cast<half2*>(baseWeights + latentIndex) = baseWeightPair;
    *reinterpret_cast<half2*>(quantizedWeights + latentIndex) = quantizedWeightPair;
    *reinterpret_cast<TD2*>(gradients + latentIndex) = gradientPair;
    *reinterpret_cast<float2*>(moments1 + latentIndex) = moment1Pair;
    *reinterpret_cast<float2*>(moments2 + latentIndex) = moment2Pair;
}

void OptimizeNetwork(
    uint32_t  dispatchSize,
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
    uint32_t threadBlockSize = OPT_WG_SIZE;
    uint32_t gridSize = DivRoundUp(dispatchSize, threadBlockSize);

    AdamConstants constants = PrepareAdamConstants(lossScale, currentStep, learningRate, beta1, beta2, epsilon);
    
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
    uint32_t    numPixels,
    uint32_t    numFeatures,
    size_t      latentStride,
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
    dim3 threadBlockSize = dim3(OPT_WG_SIZE, 1, 1);
    dim3 gridSize = DivRoundUp(dim3(numPixels, numFeatures / FeatureGridMath::FeaturesPerGroup, 1), threadBlockSize);

    AdamConstants constants = PrepareAdamConstants(lossScale, currentStep, learningRate, beta1, beta2, epsilon);
    
    if (useFloatGradients)
        LatentAdamKernel<float, float2> <<< gridSize, threadBlockSize >>> (numPixels, latentStride,
            baseWeights, quantizedWeights, (float*)gradients, moments1, moments2, gradientMask, randomSeed, constants);
    else
        LatentAdamKernel<half, half2> <<< gridSize, threadBlockSize >>> (numPixels, latentStride,
            baseWeights, quantizedWeights, (half*)gradients, moments1, moments2, gradientMask, randomSeed, constants);
}

__global__ void FreezeQuantizationKernel(
    uint32_t numPixels,
    uint32_t numFeatures,
    const half* __restrict__ baseWeights,
    half* __restrict__ quantizedWeights)
{
    using namespace cooperative_groups;

    grid_group grid = this_grid();
    uint32_t globalThreadIndex = grid.thread_rank();

    if (globalThreadIndex >= numPixels)
        return;
        
    size_t latentIndex = size_t(globalThreadIndex) * size_t(numFeatures);

    for (int featureIndex = 0; featureIndex < numFeatures; ++featureIndex)
    {
        float const quantizationStep = 1.f / 15.f;

        float weight = baseWeights[latentIndex];
        weight = std::min(std::max(weight, 0.f), 1.f);
        weight = roundf(weight / quantizationStep) * quantizationStep;
        quantizedWeights[latentIndex] = half(weight);
        
        ++latentIndex;
    }
}

void FreezeQuantization(
    uint32_t numPixels,
    uint32_t numFeatures,
    half* __restrict__ baseWeights,
    half*  __restrict__ quantizedWeights)
{
    uint32_t threadBlockSize = OPT_WG_SIZE;
    uint32_t gridSize = (numPixels + threadBlockSize - 1) / threadBlockSize;
    
    FreezeQuantizationKernel <<< gridSize, threadBlockSize >>> (numPixels, numFeatures, baseWeights, quantizedWeights);
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
    size_t gridSize = DivRoundUp(size, LOSS_ITEMS_PER_GROUP);

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
    for (size_t idx = 0; idx < gridSize; ++idx)
        acc += double(scratch.HostPtr()[idx]);
    
    outReducedLoss = float(acc);

    return cudaSuccess;
}

} // namespace ntc::cuda