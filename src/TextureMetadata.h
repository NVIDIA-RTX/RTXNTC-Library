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

#include <libntc/ntc.h>
#include "StdTypes.h"

namespace ntc
{

class Context;
class TextureSetMetadata;

namespace json
{
    struct BufferView;
}

struct ModeBufferInfo
{
    Vector<uint8_t> data;

    BufferFootprint footprint;

    ModeBufferInfo(IAllocator* allocator)
        : data(allocator)
    { }

    void Clear()
    {
        data.clear();
        footprint = BufferFootprint();
    }
};

class TextureMetadata : public ITextureMetadata
{
public:
    TextureMetadata(IAllocator* allocator, Context const* context, TextureSetMetadata* parent);

    void SetName(const char* name) override;
    char const* GetName() const override;
    String const& GetNameString() const { return m_name; }

    Status SetChannels(int firstChannel, int numChannels) override;
    void GetChannels(int& outFirstChannel, int& outNumChannels) const override;
    int GetFirstChannel() const override { return m_firstChannel; }
    int GetNumChannels() const override { return m_numChannels; }

    void SetChannelFormat(ChannelFormat format) override { m_channelFormat = format; }
    ChannelFormat GetChannelFormat() const override { return m_channelFormat; }

    void SetBlockCompressedFormat(BlockCompressedFormat format) override { m_bcFormat = format; }
    BlockCompressedFormat GetBlockCompressedFormat() const override { return m_bcFormat; }

    void SetRgbColorSpace(ColorSpace colorSpace) override;
    ColorSpace GetRgbColorSpace() const override;
    
    void SetAlphaColorSpace(ColorSpace colorSpace) override;
    ColorSpace GetAlphaColorSpace() const override;

    Status MakeAndStoreBC7ModeBuffer(int mipLevel, int widthInBlocks, int heightInBlocks,
        void const* blockData, size_t blockDataSize, size_t rowPitch) override;

    void GetBC7ModeBuffer(int mipLevel, void const** outData, size_t* outSize) const override;

    BufferFootprint GetBC7ModeBufferFootprint(int mipLevel) const override;
    
    // Library internal methods
    
    void SetBC7ModeBufferFootprint(int mipLevel, json::BufferView const& view, uint64_t binaryChunkOffset);
    Status LoadBC7ModeBuffers(IStream* inputStream);
    ModeBufferInfo const* GetModeBufferInfo(int mipLevel) const;
    TextureSetMetadata* GetParent() const { return m_parent; }

private:

    IAllocator* m_allocator;
    Context const* m_context;
    TextureSetMetadata* m_parent;
    String m_name;
    int m_firstChannel = 0;
    int m_numChannels = 0;
    ChannelFormat m_channelFormat = ChannelFormat::UNORM8;
    BlockCompressedFormat m_bcFormat = BlockCompressedFormat::None;
    ColorSpace m_rgbColorSpace = ColorSpace::Linear;
    ColorSpace m_alphaColorSpace = ColorSpace::Linear;
    Vector<ModeBufferInfo> m_modeBuffers;
};


}