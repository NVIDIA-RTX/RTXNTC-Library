/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <libntc/shaders/InferenceConstants.h>

namespace ntc
{

// The MlpDesc structure describes the geometry of the MLP used to decode neural textures.
// There used to be a few versions of the MLP in the library, now there is only one.
struct MlpDesc
{
    // Returns the total number of weights in all layers.
    static constexpr int GetTotalWeightCount()
    {
#if NTC_MLP_LAYERS == 4
        return NTC_MLP_INPUT_CHANNELS * NTC_MLP_HIDDEN0_CHANNELS
            + NTC_MLP_HIDDEN0_CHANNELS * NTC_MLP_HIDDEN1_CHANNELS
            + NTC_MLP_HIDDEN1_CHANNELS * NTC_MLP_HIDDEN2_CHANNELS
            + NTC_MLP_HIDDEN2_CHANNELS * NTC_MLP_OUTPUT_CHANNELS;
#elif NTC_MLP_LAYERS == 3
        return NTC_MLP_INPUT_CHANNELS * NTC_MLP_HIDDEN0_CHANNELS
            + NTC_MLP_HIDDEN0_CHANNELS * NTC_MLP_HIDDEN1_CHANNELS
            + NTC_MLP_HIDDEN1_CHANNELS * NTC_MLP_OUTPUT_CHANNELS;
#else
        #error "Unsupported NTC_MLP_LAYERS value"
#endif
    }

    // Returns the total number of outputs from all layers.
    static constexpr int GetTotalOutputCount()
    {
#if NTC_MLP_LAYERS == 4
        return NTC_MLP_HIDDEN0_CHANNELS
            + NTC_MLP_HIDDEN1_CHANNELS
            + NTC_MLP_HIDDEN2_CHANNELS
            + NTC_MLP_OUTPUT_CHANNELS;
#elif NTC_MLP_LAYERS == 3
        return NTC_MLP_HIDDEN0_CHANNELS
            + NTC_MLP_HIDDEN1_CHANNELS
            + NTC_MLP_OUTPUT_CHANNELS;
#else
        #error "Unsupported NTC_MLP_LAYERS value"
#endif
    }

    // Returns the number of inputs for a specific layer by index.
    static constexpr int GetLayerInputChannels(int layer)
    {
        switch(layer)
        {
            case 0:
                return NTC_MLP_INPUT_CHANNELS;
            case 1:
                return NTC_MLP_HIDDEN0_CHANNELS;
            case 2:
                return NTC_MLP_HIDDEN1_CHANNELS;
#if NTC_MLP_LAYERS == 4
            case 3:
                return NTC_MLP_HIDDEN2_CHANNELS;
#endif
            default:
                return 0;
        }
    }

    // Returns the number of outputs for a specific layer by index.
    static constexpr int GetLayerOutputChannels(int layer)
    {
        switch(layer)
        {
            case 0:
                return NTC_MLP_HIDDEN0_CHANNELS;
            case 1:
                return NTC_MLP_HIDDEN1_CHANNELS;
#if NTC_MLP_LAYERS == 4
            case 2:
                return NTC_MLP_HIDDEN2_CHANNELS;
            case 3:
                return NTC_MLP_OUTPUT_CHANNELS;
#elif NTC_MLP_LAYERS == 3
            case 2:
                return NTC_MLP_OUTPUT_CHANNELS;
#endif
            default:
                return 0;
        }
    }
};

}