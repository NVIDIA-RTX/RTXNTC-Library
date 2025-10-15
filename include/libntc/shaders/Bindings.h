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

#ifndef NTC_BINDINGS_H
#define NTC_BINDINGS_H

// Bindings for the decompression shader, see MakeDecompressionComputePass in ntc.h
#define NTC_BINDING_DECOMPRESSION_INPUT_SPACE       0
#define NTC_BINDING_DECOMPRESSION_CONSTANT_BUFFER   0 // b0 or descriptorSet 0, binding 0
#define NTC_BINDING_DECOMPRESSION_LATENT_TEXTURE    1 // t1 or descriptorSet 0, binding 1
#define NTC_BINDING_DECOMPRESSION_WEIGHT_BUFFER     2 // t2 or descriptorSet 0, binding 2
#define NTC_BINDING_DECOMPRESSION_LATENT_SAMPLER    3 // s3 or descriptorSet 0, binding 3
#define NTC_BINDING_DECOMPRESSION_OUTPUT_SPACE      1
#define NTC_BINDING_DECOMPRESSION_OUTPUT_TEXTURES   0 // u0, u1... or descriptorSet 1, binding 0[...]

// Bindings for the block compression shader, see MakeBlockCompressionComputePass in ntc.h
#define NTC_BINDING_BC_CONSTANT_BUFFER              0 // b0 or descriptorSet 0, binding 0
#define NTC_BINDING_BC_INPUT_TEXTURE                1 // t1 or descriptorSet 0, binding 1
#define NTC_BINDING_BC_OUTPUT_TEXTURE               2 // u2 or descriptorSet 0, binding 2
#define NTC_BINDING_BC_ACCELERATION_BUFFER          3 // u3 or descriptorSet 0, binding 3

// Bindings for the image difference shader, see MakeImageDifferenceComputePass in ntc.h
#define NTC_BINDING_IMAGE_DIFF_CONSTANT_BUFFER      0 // b0 or descriptorSet 0, binding 0
#define NTC_BINDING_IMAGE_DIFF_INPUT_TEXTURE_A      1 // t1 or descriptorSet 0, binding 1
#define NTC_BINDING_IMAGE_DIFF_INPUT_TEXTURE_B      2 // t2 or descriptorSet 0, binding 2
#define NTC_BINDING_IMAGE_DIFF_OUTPUT_BUFFER        3 // u3 or descriptorSet 0, binding 3

#endif // NTC_BINDINGS_H