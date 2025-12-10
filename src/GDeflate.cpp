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

#include "GDeflate.h"
#include "Errors.h"
#include "GraphicsResources.h"
#include "ThreadPool.h"
#include <libdeflate.h>

namespace ntc
{

#define REQUIRE_NOT_NULL(arg) if (arg == nullptr) { SetErrorMessage(#arg " is NULL."); return Status::InvalidArgument; }
#define REQUIRE_NOT_ZERO(arg) if (arg == 0) { SetErrorMessage(#arg " is 0."); return Status::InvalidArgument; }

constexpr int kDefaultCompressionLevel = 9;
constexpr uint32_t kDefaultTileSize = 64 * 1024;

// The format and semantics of the DirectStorage compressed stream for GDeflate data are taken from the
// reference implementation of GDeflate: https://github.com/microsoft/DirectStorage/tree/main/GDeflate/GDeflate 

struct TileStream
{
    static constexpr uint8_t kGDeflateId = 4;
    static constexpr uint32_t kMaxTiles = (1 << 16) - 1;

    uint8_t id;
    uint8_t magic;

    uint16_t numTiles;

    uint32_t tileSizeIdx : 2; // this must be set to 1
    uint32_t lastTileSize : 18;
    uint32_t reserved1 : 12;
};

static size_t GetDirectStorageStreamHeaderSize(size_t numPages)
{
    return sizeof(TileStream) + sizeof(uint32_t) * numPages;
}

class CompressTileTask : public ThreadPoolTask
{
public:
    bool Run() override
    {
        libdeflate_gdeflate_compressor* compressor = libdeflate_alloc_gdeflate_compressor(compressionLevel);
        if (!compressor)
            return false;

        size_t compressedSize = libdeflate_gdeflate_compress(compressor, uncompressedData, uncompressedSize, page, 1);

        libdeflate_free_gdeflate_compressor(compressor);

        return compressedSize != 0;
    }

    libdeflate_gdeflate_out_page* page = nullptr;
    void const* uncompressedData = nullptr;
    size_t uncompressedSize = 0;
    int compressionLevel = 0;
};

bool CompressGDeflate(const void* uncompressedData, size_t uncompressedSize,
    Vector<uint8_t>& compressedData, ThreadPool* threadPool, IAllocator* allocator,
    uint32_t* pOutCrc32, int compressionLevel)
{
    if (compressionLevel <= 0)
        compressionLevel = kDefaultCompressionLevel;
    else
        compressionLevel = std::min(compressionLevel, 12);


    size_t numPages = uncompressedSize / kDefaultTileSize;
    size_t const lastPageSize = uncompressedSize - kDefaultTileSize * numPages;
    if (lastPageSize != 0)
        ++numPages;
    if (numPages > TileStream::kMaxTiles)
        return false;

    // Use a custom page size bound that's slightly larger than what libdeflate reports (65708).
    // Sometimes, using the libdeflate bound results in heap corruption when compressing latents
    // because they don't compress well and produce large outputs.
    //
    // size_t const pageSizeBound = libdeflate_gdeflate_compress_bound(compressor, kDefaultTileSize, nullptr);
    //
    size_t const pageSizeBound = kDefaultTileSize + 1024;
    
    size_t const compressedSizeBound = numPages * pageSizeBound;

    Vector<uint8_t> outputBuffer(allocator);
    outputBuffer.resize(compressedSizeBound);

    bool const useThreadPool = threadPool && numPages > 1;

    Vector<libdeflate_gdeflate_out_page> pages(allocator);
    pages.resize(numPages);
    size_t consumedSize = 0;
    for (size_t i = 0; i < numPages; ++i)
    {
        pages[i].data = outputBuffer.data() + i * pageSizeBound;
        pages[i].nbytes = pageSizeBound;

        if (useThreadPool)
        {
            std::shared_ptr<CompressTileTask> task = MakeShared<CompressTileTask>(allocator);
            task->page = &pages[i];
            task->compressionLevel = compressionLevel;
            task->uncompressedData = static_cast<uint8_t const*>(uncompressedData) + consumedSize;
            task->uncompressedSize = std::min(uncompressedSize - consumedSize, size_t(kDefaultTileSize));
            consumedSize += task->uncompressedSize;
            threadPool->AddTask(std::move(task));
        }
    }
    
    size_t compressedSize = 0;
    if (useThreadPool)
    {
        if (threadPool->WaitForTasks())
        {
            for (libdeflate_gdeflate_out_page const& page : pages)
                compressedSize += page.nbytes;
        }
    }
    else
    {
        libdeflate_gdeflate_compressor* compressor = libdeflate_alloc_gdeflate_compressor(compressionLevel);
        if (!compressor)
            return false;

        compressedSize = libdeflate_gdeflate_compress(compressor,
            uncompressedData, uncompressedSize,
            pages.data(), pages.size());

        libdeflate_free_gdeflate_compressor(compressor);
    }

    if (compressedSize == 0)
        return false;
    
    size_t const headerSize = GetDirectStorageStreamHeaderSize(numPages);
    compressedData.resize(headerSize + compressedSize);

    TileStream* header = reinterpret_cast<TileStream*>(compressedData.data());
    uint32_t* tileOffsets = reinterpret_cast<uint32_t*>(compressedData.data() + sizeof(TileStream));
    
    header->id = TileStream::kGDeflateId;
    header->magic = header->id ^ 0xff;
    header->numTiles = numPages;
    header->tileSizeIdx = 1;
    header->lastTileSize = lastPageSize;

    tileOffsets[0] = lastPageSize;
    size_t compressedOffset = 0;
    for (size_t i = 0; i < numPages; ++i)
    {
        if (i > 0)
            tileOffsets[i] = compressedOffset;
        compressedOffset += pages[i].nbytes;
    }

    size_t outputOffset = headerSize;
    for (size_t i = 0; i < numPages; ++i)
    {
        // Check for buffer overflow
        if (outputOffset + pages[i].nbytes > compressedData.size())
            return false;

        // Write page data
        memcpy(compressedData.data() + outputOffset, pages[i].data, pages[i].nbytes);
        outputOffset += pages[i].nbytes;
    }

    if (pOutCrc32)
    {
        *pOutCrc32 = libdeflate_crc32(0, uncompressedData, uncompressedSize);
    }

    return true;
}

Status DecompressGDeflate(void const* compressedData, size_t compressedSize,
    void* uncompressedData, size_t uncompressedSize, IAllocator* allocator, uint32_t expectedCrc32)
{
    // Validate the input parameters
    REQUIRE_NOT_NULL(compressedData);
    REQUIRE_NOT_NULL(uncompressedData);
    REQUIRE_NOT_ZERO(uncompressedSize);
    REQUIRE_NOT_NULL(allocator);

    if (compressedSize < sizeof(TileStream))
    {
        SetErrorMessage("Compressed buffer is too small (%zu bytes).", compressedSize);
        return Status::InvalidData;
    }
    
    TileStream const* header = static_cast<TileStream const*>(compressedData);
    if (header->id != TileStream::kGDeflateId || header->magic != (header->id ^ 0xff))
    {
        SetErrorMessage("Compressed data has an unrecognized header.");
        return Status::InvalidData;
    }

    size_t const headerSize = GetDirectStorageStreamHeaderSize(header->numTiles);
    if (compressedSize < headerSize)
    {
        SetErrorMessage("Compressed buffer is too small (%zu bytes).", compressedSize);
        return Status::InvalidData;
    }

    size_t storedUncompressedSize = (header->numTiles - 1) * kDefaultTileSize;
    if (header->lastTileSize != 0)
        storedUncompressedSize += header->lastTileSize;
    else
        storedUncompressedSize += kDefaultTileSize;

    if (storedUncompressedSize > uncompressedSize)
    {
        SetErrorMessage("Provided buffer (%zu bytes) is too small for the uncompressed data (%zu bytes).",
            uncompressedSize, storedUncompressedSize);
        return Status::InvalidArgument;
    }

    // Parse the input data into pages
    Vector<libdeflate_gdeflate_in_page> pages(allocator);
    pages.resize(header->numTiles);

    uint32_t const* tileOffsets = reinterpret_cast<uint32_t const*>(
        static_cast<uint8_t const*>(compressedData) + sizeof(TileStream));

    for (uint32_t i = 0; i < header->numTiles; ++i)
    {
        uint32_t const compressedOffset = (i == 0) ? 0 : tileOffsets[i];

        libdeflate_gdeflate_in_page& page = pages[i];
        page.data = static_cast<uint8_t const*>(compressedData) + compressedOffset + headerSize;

        if (i == header->numTiles - 1)
            page.nbytes = tileOffsets[0];
        else
            page.nbytes = tileOffsets[i + 1] - compressedOffset;
    }
    
    libdeflate_gdeflate_decompressor* decompressor = libdeflate_alloc_gdeflate_decompressor();
    if (!decompressor)
    {
        SetErrorMessage("Failed to allocate a GDeflate decompressor.");
        return Status::InternalError;
    }

    // Decompress the data
    size_t outputBytes = 0;
    libdeflate_result result = libdeflate_gdeflate_decompress(decompressor, pages.data(), pages.size(),
        uncompressedData, storedUncompressedSize, &outputBytes);

    libdeflate_free_gdeflate_decompressor(decompressor);

    if (result == LIBDEFLATE_BAD_DATA)
    {
        SetErrorMessage("GDeflate decompression failed: input data is corrupted or invalid.");
        return Status::InvalidData;
    }
    else if (result != LIBDEFLATE_SUCCESS)
    {
        SetErrorMessage("GDeflate decompression failed with error code %d.", int(result));
        return Status::InternalError;
    }
    
    if (outputBytes != storedUncompressedSize)
    {
        SetErrorMessage("GDeflate decompression output size mismatch: expected %zu bytes, got %zu bytes.",
            storedUncompressedSize, outputBytes);
        return Status::InternalError;
    }

    if (expectedCrc32 != 0)
    {
        uint32_t const actualCrc32 = libdeflate_crc32(0, uncompressedData, outputBytes);
        if (actualCrc32 != expectedCrc32)
        {
            SetErrorMessage("GDeflate decompression CRC32 mismatch: expected 0x%08X, got 0x%08X.",
                expectedCrc32, actualCrc32);
            return Status::InvalidData;
        }
    }

    ClearErrorMessage();
    return Status::Ok;
}

// Public function declared in ntc.h
size_t GetGDeflateHeaderSize(size_t uncompressedSize)
{
    size_t const numPages = (uncompressedSize + kDefaultTileSize - 1) / kDefaultTileSize;
    return GetDirectStorageStreamHeaderSize(numPages);
}

Status DecompressGDeflateOnVulkanGPU(GraphicsResources const* resources, void* commandBuffer,
    void const* pCompressedHeader, size_t compressedHeaderSize,
    uint64_t compressedGpuVA, uint64_t decompressedGpuVA)
{
#if NTC_WITH_VULKAN
    if (resources->GetGraphicsApi() != GraphicsAPI::Vulkan)
    {
        SetErrorMessage("DecompressGDeflateOnVulkanGPU requires a Vulkan device.");
        return Status::Unsupported;
    }

    if (!resources->pfn_vkCmdDecompressMemoryNV)
    {
        SetErrorMessage("vkCmdDecompressMemoryNV function is not available.");
        return Status::Unsupported;
    }

    REQUIRE_NOT_NULL(commandBuffer);
    REQUIRE_NOT_NULL(pCompressedHeader);
    REQUIRE_NOT_ZERO(compressedGpuVA);
    REQUIRE_NOT_ZERO(decompressedGpuVA);
    
    if (compressedHeaderSize < sizeof(TileStream))
    {
        SetErrorMessage("Compressed header is too small (%zu bytes).", compressedHeaderSize);
        return Status::InvalidData;
    }
    
    TileStream const* header = static_cast<TileStream const*>(pCompressedHeader);
    if (header->id != TileStream::kGDeflateId || header->magic != (header->id ^ 0xff))
    {
        SetErrorMessage("Compressed data has an unrecognized header.");
        return Status::InvalidData;
    }

    uint32_t const numTiles = header->numTiles;
    if (numTiles == 0)
    {
        // No tiles to decompress. Not sure if this should be an error or not...
        ClearErrorMessage();
        return Status::Ok;
    }

    size_t const headerSize = GetDirectStorageStreamHeaderSize(numTiles);
    if (compressedHeaderSize < headerSize)
    {
        SetErrorMessage("Compressed data header is too small (%zu bytes).", compressedHeaderSize);
        return Status::InvalidData;
    }

    Vector<VkDecompressMemoryRegionNV> regions(resources->GetAllocator());
    VkDecompressMemoryRegionNV* pRegions = nullptr;
    size_t const sizeOfRegions = numTiles * sizeof(VkDecompressMemoryRegionNV);
    if (sizeOfRegions < 8192)
    {
        // If there are relatively few pages, allocate the array on the stack to avoid a dynamic allocation
        pRegions = static_cast<VkDecompressMemoryRegionNV*>(alloca(sizeOfRegions));
    }
    else
    {
        regions.resize(numTiles);
        pRegions = regions.data();
    }

    uint32_t const* tileOffsets = reinterpret_cast<uint32_t const*>(
        static_cast<uint8_t const*>(pCompressedHeader) + sizeof(TileStream));

    size_t const uncompressedSize = numTiles * kDefaultTileSize + header->lastTileSize;
    uint32_t decompressedTileOffset = 0;
    for (uint32_t i = 0; i < numTiles; ++i)
    {
        VkDecompressMemoryRegionNV& region = pRegions[i];
        region.decompressionMethod = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_NV;

        uint32_t compressedTileOffset = (i == 0) ? 0 : tileOffsets[i];
        region.srcAddress = compressedGpuVA + compressedTileOffset;
        region.dstAddress = decompressedGpuVA + decompressedTileOffset;
        if (i == numTiles - 1)
        {
            region.compressedSize = tileOffsets[0];
            region.decompressedSize = uncompressedSize - decompressedTileOffset;
        }
        else
        {
            region.compressedSize = tileOffsets[i + 1] - compressedTileOffset;
            region.decompressedSize = kDefaultTileSize;
        }
        decompressedTileOffset += region.decompressedSize;
    }

    VkCommandBuffer vkCmdBuf = static_cast<VkCommandBuffer>(commandBuffer);
    resources->pfn_vkCmdDecompressMemoryNV(vkCmdBuf, numTiles, pRegions);

    ClearErrorMessage();
    return Status::Ok;

#else // !NTC_WITH_VULKAN

    SetErrorMessage("LibNTC was build without Vulkan support.");
    return Status::NotImplemented;
#endif
}

} // namespace ntc