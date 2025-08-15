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

#include "FeatureGridMath.h"
#include "MathUtils.h"
#include <libntc/shaders/InferenceConstants.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>

namespace ntc
{

size_t FeatureGridMath::CalculateQuantizedLatentsSize(int imageWidth, int imageHeight, int imageMips, int highResGridScale,
    int highResFeatures, int lowResFeatures, int highResQuantBits, int lowResQuantBits)
{
    int numNeuralMipLevels = CalculateNumNeuralMipLevels(imageWidth, imageHeight, highResGridScale);

    // Clamp the theoretical mip count with the one that will be used for this texture
    const int lastNeuralMip = LodToNeuralLod(imageMips - 1, highResGridScale, NTC_MAX_NEURAL_MIPS);
    numNeuralMipLevels = std::min(numNeuralMipLevels, lastNeuralMip + 1);

    size_t quantizedLatentUintCount = 0;

    for (int mip = 0; mip < numNeuralMipLevels; mip++)
    {
        const size_t highResLatentCountInThisMip = CalculateNumLatentsInNeuralMip(Grid::HighRes, imageWidth, imageHeight, highResGridScale, mip);
        quantizedLatentUintCount += GetQuantizedLatentSizeUints(highResLatentCountInThisMip * highResFeatures, highResQuantBits);

        const size_t lowResLatentCountInThisMip = CalculateNumLatentsInNeuralMip(Grid::LowRes, imageWidth, imageHeight, highResGridScale, mip);
        quantizedLatentUintCount += GetQuantizedLatentSizeUints(lowResLatentCountInThisMip * lowResFeatures, lowResQuantBits);
    }

    return quantizedLatentUintCount * sizeof(uint32_t);
}

size_t FeatureGridMath::GetQuantizedLatentSizeUints(size_t num_weights, int quant_bits)
{
    assert(quant_bits < 32);
    const uint32_t elements_per_uint = 32 / uint32_t(quant_bits);
    return (num_weights + elements_per_uint - 1) / elements_per_uint;
}

int FeatureGridMath::LodToNeuralLod(int lod, int highResGridScale, int neuralLods)
{
    return std::min(neuralLods - 1, std::max(0, lod - Log2i(highResGridScale)) / NeuralMipRatio);
}

int FeatureGridMath::GetGridDimension(Grid grid, int imageDimension, int neuralLod, int highResScale)
{
    int scale = (grid == Grid::HighRes) ? highResScale : (highResScale * 2);
    return std::max((imageDimension / scale) >> (neuralLod * NeuralMipRatio), 1);
}

void FeatureGridMath::GetPositionLodAndScale(int neuralLod, int mipLevel, float& outPositionLod, float& outPositionScale)
{
    float const lodDiff = float(mipLevel - neuralLod * NeuralMipRatio); 
    outPositionLod = lodDiff / 3.f;
    outPositionScale = powf(2.f, lodDiff);
}

int FeatureGridMath::CalculateNumNeuralMipLevels(int imageWidth, int imageHeight, int highResGridScale)
{
    const int minImageSize = std::min(imageWidth, imageHeight);
    const int minGridSize = int(float(minImageSize) / float(highResGridScale * 2));
    return std::max(1, int((1.f + floor(std::log2f(float(minGridSize)))) / float(NeuralMipRatio)));
}

size_t FeatureGridMath::CalculateNumLatentsInNeuralMip(Grid grid, int imageWidth, int imageHeight, int highResGridScale, int mip)
{
    int mipWidth = GetGridDimension(grid, imageWidth, mip, highResGridScale);
    int mipHeight = GetGridDimension(grid, imageHeight, mip, highResGridScale);
    return size_t(mipWidth) * size_t(mipHeight);
}

}