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

#pragma once

#include <libntc/ntc.h>

namespace ntc
{

class GraphicsResources;

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

size_t GetDataTypeSize(DataType type);

bool MakeQuantizedWeightLayout(GraphicsResources const* resources,
    InferenceWeightType weightType, WeightLayout& outLayout);

void MakeFP16WeightLayout(WeightLayout& outLayout);

} // namespace ntc