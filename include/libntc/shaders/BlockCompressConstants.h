/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef BLOCK_COMPRESS_CONSTANTS_H
#define BLOCK_COMPRESS_CONSTANTS_H

#define BLOCK_COMPRESS_CS_GROUP_WIDTH 16
#define BLOCK_COMPRESS_CS_GROUP_HEIGHT 8

// BC7 in single-mode accelerated path uses a larger thread group size to make thread reordering more effective.
// BC7 in non-accelerated path is not optimized for runtime use and doesn't care as much about group size.
#define BLOCK_COMPRESS_BC7_CS_GROUP_WIDTH 32
#define BLOCK_COMPRESS_BC7_CS_GROUP_HEIGHT 16

#define BLOCK_COMPRESS_BC7_MP_BITS 9 // 3 bits for mode, 6 bits for partition

// We use 0 as the "no data" value, and remap mode 0 partition 0 to mode 6 partition 63 (which is otherwise invalid)
#define BLOCK_COMPRESS_BC7_MODE0_PART0_VALUE 0x1BF

struct NtcBlockCompressConstants
{
    int srcLeft;
    int srcTop;
    int dstOffsetX;
    int dstOffsetY;

    int widthInBlocks;
    int heightInBlocks;
    float alphaThreshold;
    unsigned int modeBufferByteOffset;

    unsigned int modeMapWidthInBlocks;
    unsigned int modeMapHeightInBlocks;
    int modeMapOffsetX;
    int modeMapOffsetY;
};

#endif
