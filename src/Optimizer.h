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

#include <cuda_fp16.h>
#include "CudaArray.h"

namespace ntc::cuda
{

void OptimizeNetwork(
    uint32_t dispatchSize,
    bool   useFloatGradients,
    half* __restrict__ baseWeights,
    half* __restrict__ quantizedWeights,
    void* __restrict__ gradients,
    float* __restrict__ moments1,
    float* __restrict__ moments2,

    float  lossScale,
    float  currentStep,
    uint32_t randomSeed,
    float  learningRate = 1E-3f,
    float  beta1 = 0.9f,
    float  beta2 = 0.999,
    float  epsilon = 1E-8f);

void ReduceNetworkGrad(
    int       numGrads,
    int       numSlices,
    bool      useFloatGradients,
    void* __restrict__ gradients);


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

    float  lossScale,
    float  currentStep,
    uint32_t randomSeed,
    float  learningRate = 1E-3f,
    float  beta1 = 0.9f,
    float  beta2 = 0.999f,
    float  epsilon = 1E-8f);

constexpr int LOSS_GROUP_SIZE = 256;
constexpr int LOSS_ITEMS_PER_THREAD = 4;
constexpr int LOSS_ITEMS_PER_GROUP = LOSS_GROUP_SIZE * LOSS_ITEMS_PER_THREAD;

// Computes a sum of the items in 'loss' array on the GPU, outputs it into 'outReducedLoss'
cudaError_t ReduceLoss(size_t size, float* __restrict__ loss, DeviceAndHostArray<float>& scratch, float& outReducedLoss);

} // namespace ntc::cuda
