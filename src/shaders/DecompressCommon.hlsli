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

#include "libntc/shaders/ColorSpaces.hlsli"
#include "libntc/shaders/DecompressConstants.h"
#include "libntc/shaders/Inference.hlsli"
#include "libntc/shaders/Bindings.h"
#include "HashBasedRNG.hlsli"
#include "BindingHelpers.hlsli"

#ifdef __cplusplus
static const NtcDecompressConstants g_Const;
#else
NTC_DECLARE_CBUFFER(ConstantBuffer<NtcDecompressConstants> g_Const, NTC_BINDING_DECOMPRESSION_CONSTANT_BUFFER, NTC_BINDING_DECOMPRESSION_INPUT_SPACE);
#endif

NTC_DECLARE_SRV(Texture2DArray t_Latents,         NTC_BINDING_DECOMPRESSION_LATENT_TEXTURE,  NTC_BINDING_DECOMPRESSION_INPUT_SPACE);
NTC_DECLARE_SRV(ByteAddressBuffer t_WeightBuffer, NTC_BINDING_DECOMPRESSION_WEIGHT_BUFFER,   NTC_BINDING_DECOMPRESSION_INPUT_SPACE);
NTC_DECLARE_SAMPLER(SamplerState s_LatentSampler, NTC_BINDING_DECOMPRESSION_LATENT_SAMPLER,  NTC_BINDING_DECOMPRESSION_INPUT_SPACE);
NTC_DECLARE_UAV(RWTexture2D<float4> u_Outputs[],  NTC_BINDING_DECOMPRESSION_OUTPUT_TEXTURES, NTC_BINDING_DECOMPRESSION_OUTPUT_SPACE);

#define NTC_MAX(a, b) ((a) > (b) ? (a) : (b))

static const int MAX_INPUT_SIZE = NTC_MAX(NTC_MAX(NTC_MLP_INPUT_CHANNELS, NTC_MLP_HIDDEN0_CHANNELS), NTC_MAX(NTC_MLP_HIDDEN1_CHANNELS, NTC_MLP_HIDDEN2_CHANNELS));
static const int MAX_OUTPUT_SIZE = NTC_MAX(NTC_MAX(NTC_MLP_HIDDEN0_CHANNELS, NTC_MLP_HIDDEN1_CHANNELS), NTC_MAX(NTC_MLP_HIDDEN2_CHANNELS, NTC_MLP_OUTPUT_CHANNELS));
