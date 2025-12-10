/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "WeightLayout.h"
#include "MlpDesc.h"
#include "CoopVecWeightConverter.h"
#include <cassert>

/* Note on the data formats and layouts used in various places...

=== Training and CUDA decompression ===

Training and CUDA decompression use FP16 weights and bias vectors. They are stored in the `m_mlpWeightsBase`
and `m_mlpWeightsQuantized` arrays. The "Base" array contains non-quantized weights that are used by the optimizer:
it takes the gradients from the regression kernel and applies them to the base weights. The "Quantized" weights are
used by the regression kernel. They are produced by the optimizer and at that point are not yet quantized. They become
quantized on the next step using the `QuantizeNetwork` calls in the final part of the training session.

Both of these arrays contain two copies of weights: the first set at offset 0 that get FP8 quantization, and 
the second set at offset `m_numNetworkParams` that get the Int8 quantization.

There are two parts in each set of weights: the actual layer weights, and the bias vectors. The layer weights are
stored in the obscure "MMA" layout that is compatible with the tensor core matrix multiplication operations. Their size
is equal to the number of elements in all layer matrices, i.e. there are no holes in the layout. Weights for layer 0
are immediately followed by weights for layer 1, and so on until layer 3. After the matrix weights, all bias vectors
are stored consecutively: dense bias vector for layer 0, followed by bias for layer 1, and so on until layer 3.

The FP16 weights and bias vectors are converted into the Int8 or FP8 format and row-major layout by the
`QuantizeNetwork` function after training is complete. These Int8 or FP8 weights are stored in the NTC files.
Before CUDA decompression, these low-precision weights are converted back into FP16 using the
`ConvertNetworkFromQuantizedToFp16` function.

=== Int8 inference ===

The Int8 inference weights are available in one generic layout that works for DP4a based inference on any GPU.

The weights package contains three components: the matrix weights, the scale vectors, and the bias vectors. The matrix
weights for all layers are stored in Int8 format and densely packed one after another. Then the scale vectors for all
layers are stored in Float32 format and are densely packed one after another. Finally, the bias vectors for all layers
are stored in Int32 format and also densely packed.

=== FP8 inference ===

Technically, it's hybrid FP8 and Int8 inference: layers 0-2 are using FP8 weights, and the output layer uses Int8
weights. This is done to improve the output precision, which is lacking with FP8 because that format cannot even
represent all integers in the range 0-255.

FP8 weights come in two flavors, the generic one (with row-major matrices) and the CoopVec specific one. The generic
weights are only used for storage, and not used by any GPU for inference because there are currently no scalar FP8
operations in GPUs.  The CoopVec weights are derived from the generic weights when the texture set is loaded from disk.
See `TextureSetMetadata::LoadWeightsFromStream`, `TextureSetMetadata::ConvertWeightsForCoopVec` and
`CoopVecWeightConverter.cpp`.

The generic weights contain the weight matrices, bias and scale vectors. They are laid out in the following order:
- Weight matrices for layers 0-2 using the FP8 type, in row-major layout
- Weight matrices for layer 3 using the Int8 type, in row-major layout
- Bias vectors for layers 0-2 using the FP16 type
- Scale vector for layer 3 using the FP32 type
- Bias vector for layer 3 using the Int32 type

The CoopVec weights follow the same general layout. The matrix weights are converted to the CoopVec-compatible
layout on load, which is normally larger than the dense row-major layout.

=== Important code locations using these layouts ===

                                                 | FP16 | GI8* | GFP8 | CVFP8 |
-------------------------------------------------+------+------+------+-------+
Layout definition                                |      |      |      |       |
    WeightLayout.cpp (this file)                 |  X   |  X   |  X   |   X   |
CUDA training and inference                      |      |      |      |       |
    RegressionKernels.h                          |  X   |      |      |       |
Weight quantization and conversion               |      |      |      |       |
    Quantizer.cu                                 |  X   |  X   |  X   |       |
Serialization                                    |      |      |      |       |
    TextureSet::SaveToStream                     |      |  X   |  X   |       |
Deserialization                                  |      |      |      |       |
    TextureSetMetadata::LoadWeightsFromStream    |      |  X   |  X   |       |
CoopVec layout conversion                        |      |      |      |       |
    TextureSetMetadata::ConvertWeightsForCoopVec |      |  X   |  X   |   X   |
    CoopVecWeightConverter.cpp                   |      |  X   |  X   |   X   |
GAPI decompression                               |      |      |      |       |
    DecompressINT8.hlsl                          |      |  X   |      |       |
    DecompressCoopVecFP8.slang                   |      |      |      |   X   |
GAPI inference                                   |      |      |      |       |
    Inference.hlsli                              |      |  X   |      |       |
    InferenceCoopVec.hlsli                       |      |      |      |   X   |

[*] GI8 = GenericInt8, GFP8 = GenericFP8, CVFP8 = CoopVecFP8

These layout descriptions and names match the InferenceWeightType enum declared in ntc.h

*/

namespace ntc
{

static DataType GetDataTypeForWeights(InferenceWeightType weightType)
{
    switch (weightType)
    {
    case InferenceWeightType::GenericInt8:
        return DataType::Int8;
    case InferenceWeightType::GenericFP8:
    case InferenceWeightType::CoopVecFP8:
        return DataType::FP8;
    default:
        return DataType::None;
    }
}

size_t GetDataTypeSize(DataType type)
{
    switch (type)
    {
    case DataType::None:
        return 0;
    case DataType::Int8:
        return sizeof(uint8_t);
    case DataType::Int32:
        return sizeof(int32_t);
    case DataType::FP8:
        return sizeof(uint8_t);
    case DataType::FP16:
        return sizeof(uint16_t);
    case DataType::FP32:
        return sizeof(float);
    default:
        assert(!"Unsupported data type");
        return 0;
    }
}

bool MakeQuantizedWeightLayout(GraphicsResources const* resources,
    InferenceWeightType weightType, WeightLayout& outLayout)
{
    for (int layer = 0; layer < NTC_MLP_LAYERS - 1; ++layer)
    {
        outLayout.weights[layer].type = GetDataTypeForWeights(weightType);
    }
    outLayout.weights[NTC_MLP_LAYERS - 1].type = (outLayout.weights[0].type == DataType::FP8)
        ? DataType::Int8
        : outLayout.weights[0].type;

    if (!CoopVecWeightConverter::IsCoopVecWeightType(weightType))
    {
        for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
        {
            outLayout.weights[layer].size =
                MlpDesc::GetLayerInputChannels(layer) *
                MlpDesc::GetLayerOutputChannels(layer) *
                GetDataTypeSize(outLayout.weights[layer].type);
        }
    }
    else
    {
        // Handle all coopvec layouts below

        if (!CoopVecWeightConverter::IsConversionSupported(resources, weightType))
            return false;
            
        // Compute converted weight sizes for all layers
        for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
        {
            if (!CoopVecWeightConverter::GetConvertedWeightMatrixSize(resources,
                MlpDesc::GetLayerInputChannels(layer),
                MlpDesc::GetLayerOutputChannels(layer),
                outLayout.weights[layer].type,
                outLayout.weights[layer].size))
            {
                return false;
            }
        }
    }

    // Calculate the offsets for all layers' weights and total weight size
    for (int layer = 1; layer < NTC_MLP_LAYERS; ++layer)
    {
        outLayout.weights[layer].offset = outLayout.weights[layer - 1].offset + outLayout.weights[layer - 1].size;
    }
    outLayout.combinedWeights.offset = 0;
    outLayout.combinedWeights.size = outLayout.weights[NTC_MLP_LAYERS - 1].offset + outLayout.weights[NTC_MLP_LAYERS - 1].size;

    // Calculate the sizes for the scale and bias vectors
    size_t totalScaleSize = 0;
    size_t totalBiasSize = 0;
    for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
    {
        DataType scaleType = DataType::None;
        DataType biasType = DataType::None;

        switch (weightType)
        {
        case InferenceWeightType::GenericInt8:
            scaleType = DataType::FP32;
            biasType = DataType::Int32;
            break;

        case InferenceWeightType::GenericFP8:
        case InferenceWeightType::CoopVecFP8:
            if (layer == NTC_MLP_LAYERS - 1)
            {
                scaleType = DataType::FP32;
                biasType = DataType::Int32;
            }
            else
            {
                // FP8 mode uses FP16 bias vectors, no scale
                biasType = DataType::FP16;
            }
            break;
        }

        outLayout.scales[layer].type = scaleType;
        outLayout.biases[layer].type = biasType;
        outLayout.scales[layer].size = GetDataTypeSize(scaleType) * MlpDesc::GetLayerOutputChannels(layer);
        outLayout.biases[layer].size = GetDataTypeSize(biasType) * MlpDesc::GetLayerOutputChannels(layer);
        totalScaleSize += outLayout.scales[layer].size;
        totalBiasSize += outLayout.biases[layer].size;
    }

    // Allocate the scale and bias vector block
    outLayout.combinedScaleBias.offset = outLayout.combinedWeights.offset + outLayout.combinedWeights.size;
    outLayout.combinedScaleBias.size = totalScaleSize + totalBiasSize;

    // Calculate the offsets for the scale and bias vectors following the rules for each format.
    // See the comment in the beginning of WeightLayout.cpp
    switch (weightType)
    {
    case InferenceWeightType::GenericInt8:
        // All scale vectors one after another, then all bias vectors
        outLayout.scales[0].offset = outLayout.combinedScaleBias.offset;
        outLayout.biases[0].offset = outLayout.combinedScaleBias.offset + totalScaleSize;
        for (int layer = 1; layer < NTC_MLP_LAYERS; ++layer)
        {
            outLayout.scales[layer].offset = outLayout.scales[layer - 1].offset + outLayout.scales[layer - 1].size;
            outLayout.biases[layer].offset = outLayout.biases[layer - 1].offset + outLayout.biases[layer - 1].size;
        }
        break;

    case InferenceWeightType::GenericFP8:
    case InferenceWeightType::CoopVecFP8:
        // Bias vectors for all layers except the last, then scale and bias for the last layer
        {
            size_t offset = outLayout.combinedScaleBias.offset;
            for (int layer = 0; layer < NTC_MLP_LAYERS - 1; ++layer) {
                outLayout.biases[layer].offset = offset;
                offset += outLayout.biases[layer].size;
            }
            outLayout.scales[NTC_MLP_LAYERS - 1].offset = offset;
            offset += outLayout.scales[NTC_MLP_LAYERS - 1].size;
            outLayout.biases[NTC_MLP_LAYERS - 1].offset = offset;
        }
        break;
    }

    outLayout.bufferSize = outLayout.combinedWeights.size + outLayout.combinedScaleBias.size;

    return true;
}

void MakeFP16WeightLayout(WeightLayout& outLayout)
{
    size_t currentOffset = 0;
    for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        int const layerInputs = MlpDesc::GetLayerInputChannels(layerIndex);
        int const layerOutputs = MlpDesc::GetLayerOutputChannels(layerIndex);

        Span& weightSpan = outLayout.weights[layerIndex];
        weightSpan.offset = currentOffset;
        weightSpan.size = size_t(layerInputs) * size_t(layerOutputs) * sizeof(uint16_t);
        weightSpan.type = DataType::FP16;
        currentOffset += weightSpan.size;
    }

    for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        int const layerOutputs = MlpDesc::GetLayerOutputChannels(layerIndex);

        Span& biasSpan = outLayout.biases[layerIndex];
        biasSpan.offset = currentOffset;
        biasSpan.size = size_t(layerOutputs) * sizeof(uint16_t);
        biasSpan.type = DataType::FP16;
        currentOffset += biasSpan.size;
    }
    outLayout.bufferSize = currentOffset;

    assert(outLayout.bufferSize == (MlpDesc::GetTotalWeightCount() + MlpDesc::GetTotalOutputCount()) * sizeof(uint16_t));

    Span const& firstLayerWeights = outLayout.weights[0];
    Span const& lastLayerWeights = outLayout.weights[NTC_MLP_LAYERS - 1];
    outLayout.combinedWeights.offset = firstLayerWeights.offset;
    outLayout.combinedWeights.size = lastLayerWeights.offset + lastLayerWeights.size - firstLayerWeights.offset;
    outLayout.combinedWeights.type = DataType::FP16;

    Span const& firstLayerBiases = outLayout.biases[0];
    Span const& lastLayerBiases = outLayout.biases[NTC_MLP_LAYERS - 1];
    outLayout.combinedScaleBias.offset = firstLayerBiases.offset;
    outLayout.combinedScaleBias.size = lastLayerBiases.offset + lastLayerBiases.size - firstLayerBiases.offset;
    outLayout.combinedScaleBias.type = DataType::FP16;
}

} // namespace ntc