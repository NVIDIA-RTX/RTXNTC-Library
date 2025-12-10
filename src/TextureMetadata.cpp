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

#include "TextureMetadata.h"
#include "TextureSetMetadata.h"
#include "Context.h"
#include "GDeflate.h"
#include "Errors.h"
#include "JsonFileFormat.h"
#include <libntc/shaders/BlockCompressConstants.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace ntc
{

TextureMetadata::TextureMetadata(IAllocator* allocator, Context const* context, TextureSetMetadata* parent)
    : m_allocator(allocator)
    , m_context(context)
    , m_parent(parent)
    , m_name(allocator)
    , m_modeBuffers(allocator)
{
    m_modeBuffers.resize(parent->GetDesc().mips, ModeBufferInfo(allocator));
}

void TextureMetadata::SetName(const char* name)
{
    m_name = String(name, m_allocator);
}

const char* TextureMetadata::GetName() const
{
    return m_name.c_str();
}

Status TextureMetadata::SetChannels(int firstChannel, int numChannels)
{
    const TextureSetDesc textureSetDesc = m_parent->GetDesc();

    if (firstChannel < 0 || firstChannel >= textureSetDesc.channels)
    {
        SetErrorMessage("firstChannel (%d) must be between 0 and %d.", firstChannel, textureSetDesc.channels - 1);
        return Status::OutOfRange;
    }

    if (numChannels < 1 || numChannels + firstChannel > textureSetDesc.channels)
    {
        SetErrorMessage("For the provided firstChannel (%d), numChannels (%d) must be between 1 and %d.",
            firstChannel, numChannels, textureSetDesc.channels - firstChannel);
        return Status::OutOfRange;
    }

    m_firstChannel = firstChannel;
    m_numChannels = numChannels;
    return Status::Ok;
}

void TextureMetadata::GetChannels(int& outFirstChannel, int& outNumChannels) const
{
    outFirstChannel = m_firstChannel;
    outNumChannels = m_numChannels;
}

void TextureMetadata::SetRgbColorSpace(ColorSpace colorSpace)
{
    m_rgbColorSpace = colorSpace;
}

ColorSpace TextureMetadata::GetRgbColorSpace() const
{
    return m_rgbColorSpace;
}

void TextureMetadata::SetAlphaColorSpace(ColorSpace colorSpace)
{
    m_alphaColorSpace = colorSpace;
}

ColorSpace TextureMetadata::GetAlphaColorSpace() const
{
    return m_alphaColorSpace;
}

Status TextureMetadata::MakeAndStoreBC7ModeBuffer(int mipLevel, int widthInBlocks, int heightInBlocks,
        void const* blockData, size_t blockDataSize, size_t rowPitch)
{
    TextureSetDesc const& desc = m_parent->GetDesc();
    if (mipLevel < 0 || mipLevel >= desc.mips)
    {
        SetErrorMessage("mipLevel (%d) must be between 0 and %d.", mipLevel, desc.mips - 1);
        return Status::OutOfRange;
    }

    size_t const bytesPerBlock = 16;

    int const mipWidth = std::max(1, desc.width >> mipLevel);
    int const mipHeight = std::max(1, desc.height >> mipLevel);
    int const mipWidthInBlocks = (mipWidth + 3) / 4;
    int const mipHeightInBlocks = (mipHeight + 3) / 4;

    if (widthInBlocks != mipWidthInBlocks || heightInBlocks != mipHeightInBlocks)
    {
        SetErrorMessage("Provided widthInBlocks (%d) and heightInBlocks (%d) do not match the expected "
            "dimensions (%d x %d) for mip level %d.", widthInBlocks, heightInBlocks,
            mipWidthInBlocks, mipHeightInBlocks, mipLevel);
        return Status::InvalidArgument;
    }

    ModeBufferInfo& bufferInfo = m_modeBuffers[mipLevel];
    // Discard any previous data
    bufferInfo.Clear();
    
    size_t const modeBufferSize = GetBC7ModeBufferSize(mipWidthInBlocks, mipHeightInBlocks);
    bufferInfo.data.resize(modeBufferSize);

    Status status = MakeBC7ModeBuffer(widthInBlocks, heightInBlocks, blockData, blockDataSize, rowPitch,
        bufferInfo.data.data(), modeBufferSize);
    if (status != Status::Ok)
        return status;

    ClearErrorMessage();
    return Status::Ok;
}

void TextureMetadata::GetBC7ModeBuffer(int mipLevel, void const** outData, size_t* outSize) const
{
    if (outData) *outData = nullptr;
    if (outSize) *outSize = 0;

    TextureSetDesc const& desc = m_parent->GetDesc();
    if (mipLevel < 0 || mipLevel >= int(m_modeBuffers.size()))
        return;

    ModeBufferInfo const& bufferInfo = m_modeBuffers[mipLevel];
    if (bufferInfo.data.empty())
        return;

    if (outData) *outData = bufferInfo.data.data();
    if (outSize) *outSize = bufferInfo.data.size();
}

void TextureMetadata::SetBC7ModeBufferFootprint(int mipLevel, json::BufferView const& view, uint64_t binaryChunkOffset)
{
    if (mipLevel < 0 || mipLevel >= int(m_modeBuffers.size()))
        return;

    ModeBufferInfo& bufferInfo = m_modeBuffers[mipLevel];
    bufferInfo.footprint.rangeInStream.offset = view.offset + binaryChunkOffset;
    bufferInfo.footprint.rangeInStream.size = view.storedSize;
    bufferInfo.footprint.compressionType = view.compression.value_or(CompressionType::None);
    bufferInfo.footprint.uncompressedSize = view.uncompressedSize.value_or(view.storedSize);
    bufferInfo.footprint.uncompressedCrc32 = view.crc32.value_or(0);
}

Status TextureMetadata::LoadBC7ModeBuffers(IStream* inputStream)
{
    for (size_t index = 0; index < m_modeBuffers.size(); ++index)
    {
        ModeBufferInfo& bufferInfo = m_modeBuffers[index];
        BufferFootprint const& footprint = bufferInfo.footprint;

        if (footprint.rangeInStream.size == 0)
        {
            bufferInfo.data.clear();
            continue;
        }
        
        Vector<uint8_t> dataFromStream(m_allocator);
        dataFromStream.resize(footprint.rangeInStream.size);

        if (!inputStream->Seek(footprint.rangeInStream.offset) ||
            !inputStream->Read(dataFromStream.data(), dataFromStream.size()))
        {
            SetErrorMessage("Failed to read BC7 mode buffer data of size %llu for mip level %zu.",
                footprint.rangeInStream.size, index);
            return Status::IOError;
        }

        if (footprint.compressionType == CompressionType::GDeflate)
        {
            Vector<uint8_t> uncompressedData(m_allocator);
            uncompressedData.resize(footprint.uncompressedSize);

            Status status = DecompressGDeflate(
                dataFromStream.data(), dataFromStream.size(),
                uncompressedData.data(), uncompressedData.size(),
                m_allocator, footprint.uncompressedCrc32);

            if (status != Status::Ok)
                return status;
            
            bufferInfo.data = std::move(uncompressedData);
        }
        else
        {
            bufferInfo.data = std::move(dataFromStream);
        }
    }

    return Status::Ok;
}

BufferFootprint TextureMetadata::GetBC7ModeBufferFootprint(int mipLevel) const
{
    BufferFootprint result;
    if (mipLevel < 0 || mipLevel >= int(m_modeBuffers.size()))
        return result;

    ModeBufferInfo const& bufferInfo = m_modeBuffers[mipLevel];
    return bufferInfo.footprint;
}

ModeBufferInfo const* TextureMetadata::GetModeBufferInfo(int mipLevel) const
{
    if (mipLevel < 0 || mipLevel >= int(m_modeBuffers.size()))
        return nullptr;

    return &m_modeBuffers[mipLevel];
}

}