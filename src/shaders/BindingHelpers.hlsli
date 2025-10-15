/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NTC_VULKAN_HLSLI
#define NTC_VULKAN_HLSLI

#ifdef SPIRV
#define NTC_VK_BINDING(reg,dset) [[vk::binding(reg,dset)]]
#else
#define NTC_VK_BINDING(reg,dset) 
#endif

#define NTC_REGISTER_HELPER(TY,REG,SPACE) register(TY##REG, space##SPACE)

#define NTC_DECLARE_CBUFFER(decl, reg, space) \
    NTC_VK_BINDING(reg, space) decl : NTC_REGISTER_HELPER(b, reg, space)

#define NTC_DECLARE_SAMPLER(decl, reg, space) \
    NTC_VK_BINDING(reg, space) decl : NTC_REGISTER_HELPER(s, reg, space)

#define NTC_DECLARE_SRV(decl, reg, space) \
    NTC_VK_BINDING(reg, space) decl : NTC_REGISTER_HELPER(t, reg, space)

#define NTC_DECLARE_UAV(decl, reg, space) \
    NTC_VK_BINDING(reg, space) decl : NTC_REGISTER_HELPER(u, reg, space)

#endif // NTC_VULKAN_HLSLI