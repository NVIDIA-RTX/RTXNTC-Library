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
#include "ThreadPool.h"

namespace ntc
{

namespace json
{
    class Document;
}

// Helper class that manages the buffer views and compresses data during texture set saving.
class BinaryChunkBuilder
{
public:
    BinaryChunkBuilder(IAllocator* allocator, LosslessCompressionSettings const& settings);
    
    // Allocates a view and stores the data pointer in this view.
    // If enableCompression is true and compression is enabled in the settings, also compresses data
    // and keeps the compressed buffer if the compression ratio is below the threshold specified in the settings.
    //
    // Note: pData must be valid until WriteViewDataToStream is called for the same view later!
    //       The builder does not cache uncompressed data.
    uint32_t AllocateViewAndRegisterData(void const* pData, size_t uncompressedSize, bool enableCompression);

    // Transfers the view information into the JSON document object.
    void WriteViewInfosToDocument(json::Document& document);

    bool WriteAllViewsToStream(IStream* stream, uint64_t binaryChunkOffset);

    uint64_t GetBinaryChunkSize() const
    {
        return m_binaryChunkSize;
    }

    LosslessCompressionStats const& GetStatistics() const
    {
        return m_stats;
    }

private:

    struct ViewInfo
    {
        uint64_t offset = 0;
        void const* uncompressedData = nullptr;
        size_t uncompressedSize = 0;
        uint32_t uncompressedCrc32 = 0;
        Vector<uint8_t> compressedData;

        ViewInfo(IAllocator* allocator)
            : compressedData(allocator)
        { }
    };

    IAllocator* m_allocator;
    Vector<UniquePtr<ViewInfo>> m_views;
    uint64_t m_binaryChunkSize = 0;
    LosslessCompressionSettings m_settings;
    LosslessCompressionStats m_stats;
    UniquePtr<ThreadPool> m_threadPool;
    
    // Writes the data into the stream, makes sure it's 4-byte aligned.
    bool WriteViewDataToStream(uint32_t viewIndex, IStream* stream, uint64_t binaryChunkOffset);
};

}