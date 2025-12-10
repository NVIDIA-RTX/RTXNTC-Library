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
#include "tin_matrix.h"
#include "tin_activation.h"

namespace tin {

    struct InputQuant {
        TIN_DEVICE InputQuant(Quantization quant = Quantization::None) : quant(quant) {};

        TIN_DEVICE half forward(half x) { 
            half y = x;

            if (quant == Quantization::Int8)
            {
                y = y * half(scale);
                y = hrint(y);
                y = __hmax(__hmin(y, half(qmax)), half(qmin));
                y = y * half(1 / scale);
            }
            else if (quant == Quantization::FP8)
            {
                y = RoundHalfToFloatE4M3(y);
            }
            return y;
        };
        TIN_DEVICE half backward(half x_d, half x) { 
            return x_d; 
        };

        static constexpr float maxval = 1.f;
        static constexpr float scale = 128 - 0.5f;
        static constexpr float qmax  = 127;
        static constexpr float qmin  = -127;
        Quantization quant;
    };
  
    template<ReducerUpdateMode UPDATE_MODE, uint32_t NUM_THREADS, typename GradType = half>
    class HLinear
    {
    public:

        TIN_DEVICE HLinear()
            : m_weights(nullptr)
            , m_bias(nullptr)
            , m_weights_grad(nullptr)
            , m_bias_grad(nullptr)
            , m_red_mem(nullptr)
        { }

        TIN_DEVICE HLinear(
            const half* weights,
            const half* bias = nullptr,
            half* red_mem = nullptr,
            GradType* weights_grad = nullptr,
            GradType* bias_grad = nullptr) :
            m_weights(weights), m_bias(bias), m_weights_grad(weights_grad), m_bias_grad(bias_grad), m_red_mem(red_mem)
        { }

        template<uint32_t Z0, uint32_t Z1>
        TIN_DEVICE HVector<Z1> forward(const HVector<Z0>& ip)
        {
            HMatrixB<Z0, Z1> w;
            w.load_native(m_weights);

            HVector<Z1> bias;

            if (m_bias) {
                bias.load(m_bias, 0);
            }
            else {
                bias.clear();
            }

            HVector<Z1> out;
            out = mad(ip, w, bias);
            return out;
        }

        template<uint32_t Z0, uint32_t Z1>
        TIN_DEVICE HVector<Z0> backward(const HVector<Z0>& ip, const HVector<Z1>& op_grad, uint32_t wt_grad_offset = 0, uint32_t bias_grad_offset = 0)
        {    
            using RedOuter = OuterProductReducer<NUM_THREADS, Z0, Z1, UPDATE_MODE>;
            using RedSum   = SumReducer<NUM_THREADS, Z1, UPDATE_MODE>;
            HMatrixB<Z0, Z1> w;
            w.load_native(m_weights);

            auto w_t = change_major_axis(w.transpose());
            HVector<Z0> ip_grad = mul(op_grad, w_t);

            if (m_bias_grad) {
                RedSum::reduce_store(op_grad, m_red_mem, m_bias_grad + bias_grad_offset);
            }

            if (m_weights_grad) {
                RedOuter::reduce_store_native(ip, op_grad, m_red_mem, m_weights_grad + wt_grad_offset);
            }

            return ip_grad;
        }

    protected:
        const half* m_weights;
        const half* m_bias;
        GradType* m_weights_grad;
        GradType* m_bias_grad;
        half* m_red_mem;
    };
    
    template<uint32_t HIDDEN_LAYERS,
             uint32_t Z_IP,
             uint32_t Z_HI,
             uint32_t Z_OP,
             typename HiddenAct,
             typename OutputAct,
             ReducerUpdateMode UPDATE_MODE = ReducerUpdateMode::STORE,     // Accumulate gradients using stores or atomic adds
             uint32_t NUM_THREADS = WarpSize,                              // Number of threads for gradient reduction (sum)
             typename GradType=half>                                       
    class HMLP {
    public:

        static constexpr uint32_t smem_size() {
            return std::max(wt_red_size(), bias_red_size());
        }

        static constexpr uint32_t num_weights() {
            return Z_IP * Z_HI + Z_HI * Z_HI * HIDDEN_LAYERS + Z_HI * Z_OP;
        }

        static constexpr uint32_t num_bias() {
            return Z_HI + Z_HI * HIDDEN_LAYERS + Z_OP;
        }

        static constexpr uint32_t num_params() {
            return num_weights() + num_bias();
        }

        TIN_DEVICE HMLP(
             const half* weights,
             const half* bias = nullptr,
             Quantization quant = Quantization::None,
             Quantization last_hidden_quant = Quantization::None,
             half* red_mem = nullptr,
             GradType* weights_grad = nullptr,
             GradType* bias_grad = nullptr
             ) : m_h_act(quant), m_lh_act(last_hidden_quant), m_o_act(quant), m_ip_quant(quant) {

            m_ip_layer = HLinear<UPDATE_MODE, NUM_THREADS, GradType>(weights, bias, red_mem, weights_grad, bias_grad);

            weights      += Z_IP * Z_HI;
            weights_grad += Z_IP * Z_HI;
            if (bias)       bias += Z_HI;
            if (bias_grad)  bias_grad += Z_HI;

TIN_UNROLL
            for (uint32_t i = 0; i < HIDDEN_LAYERS; i++) {
                m_hidden_layers[i] = HLinear<UPDATE_MODE, NUM_THREADS, GradType>(weights, bias, red_mem, weights_grad, bias_grad);

                weights      += Z_HI * Z_HI;
                weights_grad += Z_HI * Z_HI;
                if (bias)       bias += Z_HI;
                if (bias_grad)  bias_grad += Z_HI;
            }
            m_op_layer = HLinear<UPDATE_MODE, NUM_THREADS, GradType>(weights, bias, red_mem, weights_grad, bias_grad);
        }

        TIN_DEVICE HVector<Z_OP> forward(const HVector<Z_IP>& ip) {
            auto ipq = act_forward(m_ip_quant, ip);

            m_ip_cached = ipq;
            m_hidden_cached[0] = m_ip_layer.template forward<Z_IP, Z_HI>(ipq);
            auto h = act_forward(m_h_act, m_hidden_cached[0]);
TIN_UNROLL
            for (uint32_t i = 0; i < HIDDEN_LAYERS; i++) {

                m_hidden_cached[i + 1] = m_hidden_layers[i].template forward<Z_HI, Z_HI>(h);
                if (i == HIDDEN_LAYERS - 1)
                    h = act_forward(m_lh_act, m_hidden_cached[i + 1]);
                else
                    h = act_forward(m_h_act, m_hidden_cached[i + 1]);
            }

            m_op_cached = m_op_layer.template forward<Z_HI, Z_OP>(h);
            auto op = act_forward(m_o_act, m_op_cached);
            return op;
        }

        TIN_DEVICE HVector<Z_IP> backward(const HVector<Z_OP>& grad, uint32_t wt_grad_offset=0, uint32_t bias_grad_offset=0) {

            auto op_grad  = act_backward(m_o_act, grad, m_op_cached);
            auto layer_ip = act_forward (m_h_act, m_hidden_cached[HIDDEN_LAYERS]);
            auto h_grad   = m_op_layer.backward(layer_ip, op_grad, wt_grad_offset, bias_grad_offset);

TIN_UNROLL
            for (int i = HIDDEN_LAYERS - 1; i >= 0; i--)
            {
                h_grad   = act_backward(m_h_act, h_grad, m_hidden_cached[i + 1]);
                layer_ip = act_forward(m_h_act, m_hidden_cached[i]);
                h_grad   = m_hidden_layers[i].backward(layer_ip, h_grad, wt_grad_offset, bias_grad_offset);
            }

            h_grad = act_backward(m_h_act, h_grad, m_hidden_cached[0]);
            auto ip_grad = m_ip_layer.backward(m_ip_cached, h_grad, wt_grad_offset, bias_grad_offset);

            return ip_grad;
        }


    protected:

        static constexpr uint32_t wt_red_size() {
            using Red0 = OuterProductReducer<NUM_THREADS, Z_IP, Z_HI>;
            using Red1 = OuterProductReducer<NUM_THREADS, Z_HI, Z_HI>;
            using Red2 = OuterProductReducer<NUM_THREADS, Z_HI, Z_OP>;

            return std::max(Red2::shared_mem_size(),
                   std::max(Red0::shared_mem_size(), Red1::shared_mem_size()));
        }

        static constexpr uint32_t bias_red_size() {
            using Red0 = SumReducer<NUM_THREADS, Z_HI>;
            using Red1 = SumReducer<NUM_THREADS, Z_OP>;

            return std::max(Red0::shared_mem_size(), Red1::shared_mem_size());
        }

        HiddenAct m_h_act;
        HiddenAct m_lh_act;
        OutputAct m_o_act;
        InputQuant m_ip_quant;

        half* m_weights;
        half* m_bias;

        HLinear<UPDATE_MODE, NUM_THREADS, GradType> m_ip_layer;
        HLinear<UPDATE_MODE, NUM_THREADS, GradType> m_hidden_layers[HIDDEN_LAYERS];
        HLinear<UPDATE_MODE, NUM_THREADS, GradType> m_op_layer;

        HVector<Z_IP> m_ip_cached;
        HVector<Z_HI> m_hidden_cached[HIDDEN_LAYERS + 1];
        HVector<Z_OP> m_op_cached;

    };

}
