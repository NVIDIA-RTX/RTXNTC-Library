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

size_t FeatureGridMath::CalculateQuantizedLatentsSize(int imageWidth, int imageHeight, int imageMips, int gridScale,
    int numFeatures)
{
    int numNeuralMipLevels = CalculateNumNeuralMipLevels(imageWidth, imageHeight, gridScale);

    // Clamp the theoretical mip count with the one that will be used for this texture
    int const lastNeuralMip = LodToNeuralLod(imageMips - 1, gridScale, NTC_MAX_NEURAL_MIPS) + 1; // +1 for the neural mip grouping
    numNeuralMipLevels = std::min(numNeuralMipLevels, lastNeuralMip + 1);

    size_t totalBlocks = 0;

    for (int mip = 0; mip < numNeuralMipLevels; mip++)
    {
        int const mipWidth = GetGridDimension(imageWidth, mip, gridScale);
        int const mipHeight = GetGridDimension(imageHeight, mip, gridScale);
        totalBlocks += size_t(mipWidth) * size_t(mipHeight);
    }

    int const numLayers = GetNumLayers(numFeatures);

    return totalBlocks * BytesPerLatentPixel * numLayers;
}

int FeatureGridMath::LodToNeuralLod(int lod, int gridScale, int neuralLods)
{
    // Number of first color mips that share the same latent mip level.
    // Dictionary indexed by gridScale:           1  2  3  4  5  6
    static int const sharedMipsForGridScale[] = { 2, 3, 3, 3, 4, 4 };
    assert(gridScale >= 1 && gridScale <= 6);
    int sharedMips = sharedMipsForGridScale[gridScale - 1];

    // Convert the color mip into neural mip, accounting for the shared mips and 2-mip latent grouping.
    int neuralLod = (lod - sharedMips + 2) & ~1;
    neuralLod = std::max(0, std::min(neuralLod, neuralLods - 2));
    return neuralLod;
}

int FeatureGridMath::GetGridDimension(int imageDimension, int neuralLod, int gridScale)
{
    return std::max((imageDimension / gridScale) >> neuralLod, 1);
}

int FeatureGridMath::GetNumLayers(int numFeatures)
{
    return (numFeatures + FeaturesPerLayer - 1) / FeaturesPerLayer;
}

int FeatureGridMath::CalculateNumNeuralMipLevels(int imageWidth, int imageHeight, int gridScale)
{
    const int minImageSize = std::min(imageWidth, imageHeight);
    const int minGridSize = int(float(minImageSize) / float(gridScale));
    return std::max(2, int((1.f + floor(std::log2f(float(minGridSize)))))) & ~1;
}

}