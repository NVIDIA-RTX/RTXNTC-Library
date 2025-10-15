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

#include <libntc/ntc.h>
#include <libntc/shaders/InferenceConstants.h>
#include <cstddef>

namespace ntc
{

// The MlpDesc structure describes the geometry of the MLP used to decode neural textures.
// There used to be a few versions of the MLP in the library, now there is only one.
struct MlpDesc
{
    int numFeatures;
    int inputChannels;

    int GetInputChannels() const { return inputChannels; }
    int GetHiddenChannels() const;
    int GetOutputChannels() const;
    int GetHiddenLayers() const;

    // Returns the total number of weights in all layers.
    int GetWeightCount() const;

    // Returns the total number of outputs from all layers.
    int GetLayerOutputCount() const;

    // Returns the number of inputs for a specific layer by index.
    int GetLayerInputChannels(int layer) const;

    // Returns the number of outputs for a specific layer by index.
    int GetLayerOutputChannels(int layer) const;

    static MlpDesc const& Get();
};

enum class DataType
{
    None,
    Int8,
    Int32,
    FP8,
    FP16,
    FP32
};

struct Span
{
    size_t offset = 0;
    size_t size = 0;
    DataType type = DataType::None;
};

struct WeightLayout
{
    Span weights[NTC_MLP_LAYERS]{};
    Span combinedWeights;
    Span scales[NTC_MLP_LAYERS]{};
    Span biases[NTC_MLP_LAYERS]{};
    Span combinedScaleBias;
    size_t bufferSize = 0;
};

}