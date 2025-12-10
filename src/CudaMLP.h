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
#include <libntc/shaders/InferenceConstants.h>
#include "MlpDesc.h"
#include "tin/tin_mlp.h"

namespace ntc::cuda
{
using namespace tin;

template<typename HiddenAct,
            ReducerUpdateMode UPDATE_MODE = ReducerUpdateMode::STORE, // Accumulate gradients using stores or atomic adds
            uint32_t NUM_THREADS = WarpSize,                          // Number of threads for gradient reduction (sum)
            typename GradType=half>                                       
class MLP {
public:

    static constexpr uint32_t GetSharedMemorySize()
    {
        return std::max(GetWeightReductionMemorySize(), GetBiasReductionMemorySize());
    }

    TIN_DEVICE MLP(
        const half* weights,
        const half* bias = nullptr,
        Quantization quant = Quantization::None,
        Quantization last_hidden_quant = Quantization::None,
        half* red_mem = nullptr,
        GradType* weights_grad = nullptr,
        GradType* bias_grad = nullptr
    )
        : m_hiddenActivation(quant)
        , m_lastHiddenActivation(last_hidden_quant)
        , m_inputQuantization(quant)
    {
        for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; layerIndex++)
        {
            m_layers[layerIndex] = HLinear<UPDATE_MODE, NUM_THREADS, GradType>(weights, bias, red_mem, weights_grad, bias_grad);

            uint32_t inputChannels = MlpDesc::GetLayerInputChannels(layerIndex);
            uint32_t outputChannels = MlpDesc::GetLayerOutputChannels(layerIndex);
            
            weights      += inputChannels * outputChannels;
            weights_grad += inputChannels * outputChannels;
            if (bias)       bias += outputChannels;
            if (bias_grad)  bias_grad += outputChannels;
        }
    }
/*
    3-layer MLP structure:
    m_cachedInput
        m_layers[0] ( INPUT -> HIDDEN0 )
    m_cachedHidden0
        m_hiddenActivation
    activatedHidden0
        m_layers[1] ( HIDDEN0 -> HIDDEN1 )
    m_cachedHidden1
        m_lastHiddenActivation
    activatedHidden1
        m_layers[2] ( HIDDEN1 -> OUTPUT )
    m_cachedOutput

    4-layer MLP structure:
    m_cachedInput
        m_layers[0] ( INPUT -> HIDDEN0 )
    m_cachedHidden0
        m_hiddenActivation
    activatedHidden0
        m_layers[1] ( HIDDEN0 -> HIDDEN1 )
    m_cachedHidden1
        m_hiddenActivation
    activatedHidden1
        m_layers[2] ( HIDDEN1 -> HIDDEN2 )
    m_cachedHidden2
        m_lastHiddenActivation
    activatedHidden2
        m_layers[3] ( HIDDEN2 -> OUTPUT )
    m_cachedOutput
*/

    TIN_DEVICE HVector<NTC_MLP_OUTPUT_CHANNELS> forward(const HVector<NTC_MLP_INPUT_CHANNELS>& ip)
    {
        auto quantizedInput = act_forward(m_inputQuantization, ip);

        m_cachedInput = quantizedInput;
        m_cachedHidden0 = m_layers[0].template forward<NTC_MLP_INPUT_CHANNELS, NTC_MLP_HIDDEN0_CHANNELS>(quantizedInput);
        auto activatedHidden0 = act_forward(m_hiddenActivation, m_cachedHidden0);

        m_cachedHidden1 = m_layers[1].template forward<NTC_MLP_HIDDEN0_CHANNELS, NTC_MLP_HIDDEN1_CHANNELS>(activatedHidden0);
#if NTC_MLP_LAYERS >= 4
        auto activatedHidden1 = act_forward(m_hiddenActivation, m_cachedHidden1);

        m_cachedHidden2 = m_layers[2].template forward<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_HIDDEN2_CHANNELS>(activatedHidden1);
        auto activatedHidden2 = act_forward(m_lastHiddenActivation, m_cachedHidden2);

        m_cachedOutput = m_layers[3].template forward<NTC_MLP_HIDDEN2_CHANNELS, NTC_MLP_OUTPUT_CHANNELS>(activatedHidden2);
#else
        auto activatedHidden1 = act_forward(m_lastHiddenActivation, m_cachedHidden1);
        m_cachedOutput = m_layers[2].template forward<NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_OUTPUT_CHANNELS>(activatedHidden1);
#endif

        return m_cachedOutput;
    }

    TIN_DEVICE HVector<NTC_MLP_INPUT_CHANNELS> backward(const HVector<NTC_MLP_OUTPUT_CHANNELS>& outputGradient,
        uint32_t gradientOffset)
    {
#if NTC_MLP_LAYERS >= 4
        auto activatedHidden2 = act_forward(m_lastHiddenActivation, m_cachedHidden2);
        auto activatedHidden2Gradient = m_layers[3].backward(activatedHidden2, outputGradient, gradientOffset, gradientOffset);
        auto hidden2Gradient = act_backward(m_lastHiddenActivation, activatedHidden2Gradient, m_cachedHidden2);

        auto activatedHidden1 = act_forward(m_hiddenActivation, m_cachedHidden1);
        auto activatedHidden1Gradient = m_layers[2].backward(activatedHidden1, hidden2Gradient, gradientOffset, gradientOffset);
        auto hidden1Gradient = act_backward(m_hiddenActivation, activatedHidden1Gradient, m_cachedHidden1);
#else
        auto activatedHidden1 = act_forward(m_lastHiddenActivation, m_cachedHidden1);
        auto activatedHidden1Gradient = m_layers[2].backward(activatedHidden1, outputGradient, gradientOffset, gradientOffset);
        auto hidden1Gradient = act_backward(m_lastHiddenActivation, activatedHidden1Gradient, m_cachedHidden1);
#endif
        auto activatedHidden0 = act_forward(m_hiddenActivation, m_cachedHidden0);
        auto activatedHidden0Gradient = m_layers[1].backward(activatedHidden0, hidden1Gradient, gradientOffset, gradientOffset);
        auto hidden0Gradient = act_backward(m_hiddenActivation, activatedHidden0Gradient, m_cachedHidden0);

        auto inputGradient = m_layers[0].backward(m_cachedInput, hidden0Gradient, gradientOffset, gradientOffset);

        return inputGradient;
    }


protected:

    static constexpr uint32_t GetWeightReductionMemorySize()
    {
        using Red0 = OuterProductReducer<NUM_THREADS, NTC_MLP_INPUT_CHANNELS, NTC_MLP_HIDDEN0_CHANNELS>;
        using Red1 = OuterProductReducer<NUM_THREADS, NTC_MLP_HIDDEN0_CHANNELS, NTC_MLP_HIDDEN1_CHANNELS>;
        using Red2 = OuterProductReducer<NUM_THREADS, NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_HIDDEN2_CHANNELS>;
        using Red3 = OuterProductReducer<NUM_THREADS, NTC_MLP_HIDDEN2_CHANNELS, NTC_MLP_OUTPUT_CHANNELS>;

        return std::max(std::max(Red0::shared_mem_size(), Red1::shared_mem_size()),
                        std::max(Red2::shared_mem_size(), Red3::shared_mem_size()));
    }

    static constexpr uint32_t GetBiasReductionMemorySize()
    {
        using Red0 = SumReducer<NUM_THREADS, NTC_MLP_HIDDEN0_CHANNELS>;
        using Red1 = SumReducer<NUM_THREADS, NTC_MLP_HIDDEN1_CHANNELS>;
        using Red2 = SumReducer<NUM_THREADS, NTC_MLP_HIDDEN2_CHANNELS>;
        using Red3 = SumReducer<NUM_THREADS, NTC_MLP_OUTPUT_CHANNELS>;

        return std::max(std::max(Red0::shared_mem_size(), Red1::shared_mem_size()),
                        std::max(Red2::shared_mem_size(), Red3::shared_mem_size()));
    }

    InputQuant m_inputQuantization;
    HiddenAct m_hiddenActivation;
    HiddenAct m_lastHiddenActivation;

    half* m_weights;
    half* m_bias;

    HLinear<UPDATE_MODE, NUM_THREADS, GradType> m_layers[NTC_MLP_LAYERS];

    HVector<NTC_MLP_INPUT_CHANNELS> m_cachedInput;
    HVector<NTC_MLP_HIDDEN0_CHANNELS> m_cachedHidden0;
    HVector<NTC_MLP_HIDDEN1_CHANNELS> m_cachedHidden1;
#if NTC_MLP_LAYERS >= 4
    HVector<NTC_MLP_HIDDEN2_CHANNELS> m_cachedHidden2;
#endif
    HVector<NTC_MLP_OUTPUT_CHANNELS> m_cachedOutput;
};

} // namespace ntc::cuda
