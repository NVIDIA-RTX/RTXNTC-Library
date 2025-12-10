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

#include <libntc/ntc.h>
#include "StdTypes.h"

namespace ntc
{

class GraphicsResources;
class ThreadPool;

bool CompressGDeflate(void const* uncompressedData, size_t uncompressedSize,
    Vector<uint8_t>& outputData, ThreadPool* threadPool, IAllocator* allocator, uint32_t* pOutCrc32, int compressionLevel = 0);

Status DecompressGDeflate(void const* compressedData, size_t compressedSize,
    void* uncompressedData, size_t uncompressedSize, IAllocator* allocator, uint32_t expectedCrc32 = 0);

Status DecompressGDeflateOnVulkanGPU(GraphicsResources const* resources, void* commandBuffer,
    void const* pCompressedHeader, size_t compressedHeaderSize,
    uint64_t compressedGpuVA, uint64_t decompressedGpuVA);

} // namespace ntc