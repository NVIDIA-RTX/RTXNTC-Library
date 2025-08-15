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
#include "RegressionCommon.h"
#include <libntc/ntc.h>

namespace ntc
{
struct MlpDesc;
}

namespace ntc::cuda
{

void SetMipInfos(const MipInfo* data, int count);
void SetChannelInfos(const ChannelInfo* data, int count);

bool ValidateKernelSpec(
    int highResFeatures,
    int lowResFeatures);

void Regression(
    size_t pixelsPerBatch,
    bool stableTraining,
    MlpDesc const& kernelVersion,
    RegressionKernelParams const& params);

void Inference(
    MlpDesc const& kernelVersion,
    InferenceKernelParams const& params);

} // namespace ntc::cuda