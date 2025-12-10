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

#include "BinaryChunkBuilder.h"
#include "GDeflate.h"
#include "JsonFileFormat.h"
#include "MathUtils.h"
#include "StdTypes.h"
#include <cassert>
#include <chrono>
#include <thread>

using namespace std::chrono;

namespace ntc
{

constexpr uint32_t kMaxAutomaticThreads = 8;

BinaryChunkBuilder::BinaryChunkBuilder(IAllocator* allocator, LosslessCompressionSettings const& settings)
    : m_allocator(allocator)
    , m_views(allocator)
    , m_settings(settings)
    , m_threadPool(nullptr, allocator)
{
    if (settings.compressionThreads >= 0)
    {
        uint32_t const numThreads = (settings.compressionThreads > 0)
            ? uint32_t(settings.compressionThreads)
            : std::min(kMaxAutomaticThreads, std::thread::hardware_concurrency());

        m_threadPool = MakeUniqueWithAllocator<ThreadPool>(allocator, numThreads);
    }
}

// If the current output position in the stream is not a multiple of 4 bytes, write some zeros until it is.
// Note: alignment of MLP and latent data in the stream is important for GAPI decompression, so that 
// we can load the entire file into a RawAddressBuffer and safely read uint's from it on the GPU.
static bool PadStreamTo4Bytes(IStream* outputStream)
{
    uint64_t actualOffset = outputStream->Tell();
    uint32_t padding = 0;
    if (actualOffset & 3)
        return outputStream->Write(&padding, 4 - (actualOffset & 3));
    return true;
}

uint32_t BinaryChunkBuilder::AllocateViewAndRegisterData(void const* pData, size_t uncompressedSize, bool enableCompression)
{
    UniquePtr<ViewInfo> view = MakeUniqueWithAllocator<ViewInfo>(m_allocator);
    view->offset = m_binaryChunkSize;
    view->uncompressedSize = uncompressedSize;
    view->uncompressedData = pData;
    
    size_t storedSize = uncompressedSize;
    if (enableCompression && m_settings.algorithm == CompressionType::GDeflate && uncompressedSize > 0)
    {
        time_point const begin = steady_clock::now();
        
        bool success = CompressGDeflate(pData, uncompressedSize, view->compressedData,
            m_threadPool.get(), m_allocator, &view->uncompressedCrc32, m_settings.compressionLevel);

        time_point const end = steady_clock::now();
        float const compressionTimeMs = float(duration_cast<microseconds>(end - begin).count()) * 1e-3f;
        m_stats.compressionTimeMs += compressionTimeMs;

        double compressionRatio = double(view->compressedData.size()) / double(view->uncompressedSize);
        if (success && compressionRatio >= m_settings.compressionRatioThreshold)
            success = false;

        if (success)
        {
            storedSize = view->compressedData.size();

            ++m_stats.compressedBuffers;
            m_stats.sizeOfCompressedBuffers += storedSize;
            m_stats.originalSizeOfCompressedBuffers += uncompressedSize;
        }
        else
        {
            view->compressedData.clear();
            view->uncompressedCrc32 = 0;
        }
    }

    ++m_stats.totalBuffers;
    if (view->compressedData.empty())
        m_stats.sizeOfUncompressedBuffers += uncompressedSize;

    m_binaryChunkSize += RoundUp4(storedSize);

    uint32_t const viewIndex = uint32_t(m_views.size());
    m_views.push_back(std::move(view));
    return viewIndex;
}

void BinaryChunkBuilder::WriteViewInfosToDocument(json::Document& document)
{
    document.views.clear();
    document.views.reserve(m_views.size());
    for (size_t i = 0; i < m_views.size(); ++i)
    {
        ViewInfo const& src = *m_views[i];
        json::BufferView& dst = document.views.emplace_back(document.allocator);

        dst.offset = src.offset;
        if (src.compressedData.empty())
        {
            dst.storedSize = src.uncompressedSize;
        }
        else
        {
            dst.compression = m_settings.algorithm;
            dst.storedSize = src.compressedData.size();
            dst.uncompressedSize = src.uncompressedSize;
            dst.crc32 = src.uncompressedCrc32;
        }
    }
}

bool BinaryChunkBuilder::WriteAllViewsToStream(IStream* stream, uint64_t binaryChunkOffset)
{
    for (uint32_t viewIndex = 0; viewIndex < uint32_t(m_views.size()); ++viewIndex)
    {
        if (!WriteViewDataToStream(viewIndex, stream, binaryChunkOffset))
            return false;
    }
    return true;
}

bool BinaryChunkBuilder::WriteViewDataToStream(uint32_t viewIndex, IStream* stream, uint64_t binaryChunkOffset)
{
    if (size_t(viewIndex) >= m_views.size())
        return false;
        
    ViewInfo const& view = *m_views[viewIndex];
    if (view.uncompressedSize == 0)
        return true;
    
    if (!PadStreamTo4Bytes(stream))
        return false;

#ifndef NDEBUG
    uint64_t expectedOffset = view.offset + binaryChunkOffset;
    uint64_t actualOffset = stream->Tell();
    assert(expectedOffset == actualOffset);
#endif

    if (view.compressedData.empty())
        return stream->Write(view.uncompressedData, view.uncompressedSize);
    else
        return stream->Write(view.compressedData.data(), view.compressedData.size());    
}

}