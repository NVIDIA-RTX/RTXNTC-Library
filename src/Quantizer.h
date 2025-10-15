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

namespace ntc
{

struct MlpDesc;

} // namespace ntc

namespace ntc::cuda
{

void QuantizeNetwork(
    MlpDesc const& mlpDesc,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ outputData,
    bool useFP8);

void ConvertNetworkFromQuantizedToFp16(
    MlpDesc const& mlpDesc,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ inputData,
    bool useFP8);

void ExportNetworkIntoRowMajorLayout(
    MlpDesc const& mlpDesc,
    half* __restrict__ halfWeights,
    half* __restrict__ outputData);

void ImportNetworkFromRowMajorLayout(
    MlpDesc const& mlpDesc,
    half* __restrict__ halfWeights,
    half* __restrict__ outputData);

void FreezeQuantization(
    uint32_t    numPixels,
    uint32_t    numFeatures,
    half* __restrict__ baseWeights,
    half*  __restrict__ quantizedWeights);

void PackLatents(
    int widthBlocks,
    int heightBlocks,
    int numLayers,
    size_t latentStride,
    const half* __restrict__ w_in,
    uint16_t* __restrict__ w_out);

void UnpackLatents(
    int widthBlocks,
    int heightBlocks,
    int numLayers,
    size_t latentStride,
    const uint16_t* __restrict__ w_in,
    half* __restrict__ w_out);

} // namespace ntc::cuda
