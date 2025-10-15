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

#include "tin/tin_common.h"
#include <libntc/shaders/InferenceConstants.h>
#include "ChannelInfo.h"

const int WARPS_PER_TBLOCK = 4; // This is the threadblock size in warps
const int Y_ITERS = 1; // Number of iterations in y to reduce weights in shared memory before writing to memory

const int TB_SIZE_X = tin::WarpSize;
const int TB_SIZE_Y = WARPS_PER_TBLOCK;

const int TILE_SIZE_X = TB_SIZE_X;
const int TILE_SIZE_Y = (TB_SIZE_Y * Y_ITERS);

const int LOCAL_PIXELS = TILE_SIZE_X * TILE_SIZE_Y;

struct RegressionKernelParams
{
    int   referenceWidth;
    int   referenceHeight;
    int   numChannels;
    int   numMips;
    int   numNeuralMips;
    int   numFeatures;
    int   maskChannelIndex;
    size_t latentStride;
    bool  discardMaskedOutPixels;
    bool  useFP8Quantization;
    uint32_t validChannelMask;
    uint32_t randomSeed;
    float lossScale;
    float experimentalKnob;
    const half*     __restrict__ referenceImage;
    const half*     __restrict__ latents;
    const half*     __restrict__ networkWeights;
          void*     __restrict__ latentGradients;
          void*     __restrict__ networkGradients;
          float*    __restrict__ loss;
          uint32_t* __restrict__ gradientMask;
};

struct InferenceKernelParams
{
    int   referenceWidth;
    int   referenceHeight;
    int   numChannels;
    int   maskChannelIndex;
    size_t latentStride;
    bool  discardMaskedOutPixels;
    bool  useFP8Quantization;
    uint32_t validChannelMask;
    int   highResLatentWidth;
    int   highResLatentHeight;
    int   lowResLatentWidth;
    int   lowResLatentHeight;
    int   numFeatures;
    int   lossItemsPerChannel;
    float positionScale;
    float positionLod;
    float experimentalKnob;
    const half*  __restrict__ highResLatents;
    const half*  __restrict__ lowResLatents;
    const half*  __restrict__ mlpWeights;
    const half*               referenceImage;
    half*                     outputImage;
    float*       __restrict__ outputLoss;
};

struct MipInfo
{
    uint64_t referenceTextureOffset;
    uint64_t highResLatentOffset;
    uint64_t lowResLatentOffset;
    uint64_t highResMaskOffset;
    uint64_t lowResMaskOffset;

    float positionScale;
    float positionLod;
    int neuralLod;
    float cdf;

    int highResLatentWidth;
    int highResLatentHeight;
    int lowResLatentWidth;
    int lowResLatentHeight;
};
