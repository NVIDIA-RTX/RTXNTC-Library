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

 #include "MlpDesc.h"
 #include <libntc/ntc.h>
 
 namespace ntc
 {
    
 static MlpDesc g_MlpDesc = 
    { NTC_MLP_FEATURES,  NTC_MLP_INPUT_CHANNELS };

int MlpDesc::GetHiddenChannels() const
{
    return NTC_MLP_HIDDEN_CHANNELS;
}

int MlpDesc::GetOutputChannels() const
{
    return NTC_MLP_OUTPUT_CHANNELS;
}

int MlpDesc::GetHiddenLayers() const
{
    return NTC_MLP_LAYERS - 2;
}

int MlpDesc::GetWeightCount() const
{
    return inputChannels * NTC_MLP_HIDDEN_CHANNELS
        + NTC_MLP_HIDDEN_CHANNELS * NTC_MLP_HIDDEN_CHANNELS * GetHiddenLayers()
        + NTC_MLP_HIDDEN_CHANNELS * NTC_MLP_OUTPUT_CHANNELS;
}

int MlpDesc::GetLayerOutputCount() const
{
    return NTC_MLP_HIDDEN_CHANNELS * (GetHiddenLayers() + 1) + NTC_MLP_OUTPUT_CHANNELS;
}

int MlpDesc::GetLayerInputChannels(int layer) const
{
    // Input layer
    if (layer == 0)
        return inputChannels;

    // Hidden and output layers
    if (layer > 0 && layer < NTC_MLP_LAYERS)
        return NTC_MLP_HIDDEN_CHANNELS;

    return 0;
}

int MlpDesc::GetLayerOutputChannels(int layer) const
{
    // Input and hidden layers
    if (layer >= 0 && layer < NTC_MLP_LAYERS - 1)
        return NTC_MLP_HIDDEN_CHANNELS;

    // Output layer
    if (layer == NTC_MLP_LAYERS - 1)
        return NTC_MLP_OUTPUT_CHANNELS;

    return 0;
}

MlpDesc const& MlpDesc::Get()
{
    return g_MlpDesc;
}

} // namespace ntc