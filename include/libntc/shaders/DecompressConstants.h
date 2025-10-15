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

#ifndef DECOMPRESS_CONSTANTS_H
#define DECOMPRESS_CONSTANTS_H

#include "InferenceConstants.h"

// Note: the block dimensions must be small enough to make sure that the shared memory arrays
// used in the shader fit into the DirectX's 32 KB shared memory size limit.
#define DECOMPRESS_CS_BLOCK_WIDTH 16
#define DECOMPRESS_CS_BLOCK_HEIGHT 8

#define DECOMPRESS_CS_MAX_OUTPUTS 16
#define DECOMPRESS_CS_MAX_CHANNELS 16

struct NtcDecompressOutputDesc
{
    int firstChannel;
    int numChannels;
    int textureIndex;
    float ditherScale;

    int srcRgbColorSpace;
    int dstRgbColorSpace;
    int srcAlphaColorSpace;
    int dstAlphaColorSpace;
};

struct NtcDecompressConstants
{
    NtcDecompressOutputDesc outputs[DECOMPRESS_CS_MAX_OUTPUTS];
#ifdef __cplusplus
    NtcColorMipConstants colorMip;
    int networkWeightOffsets[4];
    int networkBiasOffsets[4];
    int networkScaleOffsets[4];
#else
    int4 colorMip;
    int4 networkWeightOffsets;
    int4 networkBiasOffsets;
    int4 networkScaleOffsets;
#endif

    int srcLeft;
    int srcTop;
    int srcRight;
    int srcBottom;

    int dstLeft;
    int dstTop;
    int imageWidth;
    int imageHeight;

    int numOutputs;
    int pad0;
    int pad1;
    int pad2;
};

#endif