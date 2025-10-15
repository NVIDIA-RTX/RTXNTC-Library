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

#pragma once

#include <cstddef>
#include <libntc/shaders/InferenceConstants.h>

namespace ntc
{

class FeatureGridMath
{
public:
    static constexpr int BytesPerLatentPixel = 2;
    static constexpr int FeaturesPerLayer = NTC_FEATURES_PER_LAYER;
    static constexpr int FeaturesPerGroup = 2;

    static size_t CalculateQuantizedLatentsSize(int imageWidth, int imageHeight, int imageMips, int gridScale,
        int numFeatures);

    static int LodToNeuralLod(int lod, int gridScale, int neuralLods);

    static int GetGridDimension(int imageDimension, int neuralLod, int gridScale);

    static int GetNumLayers(int numFeatures);

protected:
    static int CalculateNumNeuralMipLevels(int imageWidth, int imageHeight, int gridScale);
};

}