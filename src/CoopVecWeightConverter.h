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

#include "WeightLayout.h"
#include <libntc/ntc.h>
#include <cstdint>
#include <cstdlib>
#include <libntc/shaders/InferenceConstants.h>

namespace ntc
{

struct MlpDesc;

class GraphicsResources;

class CoopVecWeightConverter
{
public:
    static bool IsConversionSupported(GraphicsResources const* resources, InferenceWeightType weightType);

    static void ConvertWeights(GraphicsResources const* resources,
        WeightLayout const& srcLayout, void* srcBuffer, uint64_t srcOffset,
        WeightLayout const& dstLayout, void* dstBuffer, uint64_t dstOffset,
        void* commandListOrBuffer);

    static bool IsCoopVecWeightType(InferenceWeightType weightType);
    static InferenceWeightType GetGenericWeightType(InferenceWeightType weightType);

    static bool GetConvertedWeightMatrixSize(GraphicsResources const* resources,
        int inputChannels, int outputChannels,
        DataType weightType, size_t& outSize);

#if NTC_WITH_DX12
    static void IsDX12CoopVecSupported(GraphicsResources const* resources, bool& outSupported);
#endif
#if NTC_WITH_VULKAN
    static void IsVkCoopVecSupported(GraphicsResources const* resources, bool& outSupported);
#endif
};

}