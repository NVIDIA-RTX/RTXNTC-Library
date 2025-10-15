/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <optional>
#include "StdTypes.h"

namespace ntc::json
{

// Conservative estimate for the JSON chunk, normally around 2-4 kB
static constexpr size_t JsonChunkSizeLimit = 8192;

// Header for the container, stored in binary form at the beginning of NTC files
struct FileHeader
{
    static constexpr uint32_t SignatureValue = 0x5845544E; // "NTEX"
    static constexpr uint32_t CurrentVersion = 0x100; // Version of the container, not the JSON schema

    uint32_t signature = SignatureValue;
    uint32_t version = CurrentVersion;

    uint64_t jsonChunkOffset = 0;
    uint64_t jsonChunkSize = 0;
    uint64_t binaryChunkOffset = 0;
    uint64_t binaryChunkSize = 0;
};

// The rest of the structures and enums below are representing the schema for the JSON chunk
// stored in the NTC files. When any changes to these structures are made, corresponsing changes
// need to be made to the schema metadata in JsonFileFormat.cpp.
// See docs/TextureSetFile.md in the NTC SDK for more information about the schema.
// The main object in the JSON chunk is represented by the Document struct below.
// Any fields not described in the schema will be silently skipped by the serializer and the parser.

enum class Compression
{
    None
};

struct BufferView
{
    uint64_t offset = 0;
    uint64_t storedSize = 0;
    std::optional<Compression> compression;
    std::optional<uint64_t> uncompressedSize;

    BufferView() = default;
    BufferView(IAllocator* allocator) { }
};

struct LatentShape
{
    uint32_t numFeatures = 0;

    LatentShape() = default;
    LatentShape(IAllocator* allocator) { }
};

enum class MatrixLayout
{
    RowMajor,
    ColumnMajor
};

enum class ActivationType
{
    HGELUClamp
};

enum class MlpDataType
{
    Int8,
    Int32,
    FloatE4M3,
    FloatE5M2,
    Float16,
    Float32
};

struct MLPLayer
{
    uint32_t inputChannels = 0;
    uint32_t outputChannels = 0;
    uint32_t weightView = 0;
    std::optional<uint32_t> scaleView;
    uint32_t biasView = 0;
    MlpDataType weightType = MlpDataType::Int8;
    MlpDataType biasType = MlpDataType::Int8;
    std::optional<MlpDataType> scaleType;

    MLPLayer() = default;
    MLPLayer(IAllocator* allocator) { }
};

struct MLP
{
    Vector<MLPLayer> layers;
    std::optional<ActivationType> activation;
    std::optional<MatrixLayout> weightLayout;

    MLP(IAllocator* allocator)
        : layers(allocator)
    { }
};

struct Texture
{
    String name;
    uint32_t firstChannel = 0;
    uint32_t numChannels = 0;
    std::optional<ChannelFormat> channelFormat;
    std::optional<ColorSpace> rgbColorSpace;
    std::optional<ColorSpace> alphaColorSpace;
    std::optional<BlockCompressedFormat> bcFormat;
    std::optional<uint32_t> bcQuality;
    std::optional<uint32_t> bcAccelerationDataView;

    Texture(IAllocator* allocator)
        : name(allocator)
    { }
};

struct Channel
{
    std::optional<ColorSpace> colorSpace;

    Channel(IAllocator* allocator)
    { }
};

struct LatentImage
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t arraySize = 0;
    uint32_t view = 0;

    LatentImage(IAllocator* allocator)
    { }
};

struct ColorImageData
{
    uint32_t view = 0;
    std::optional<ChannelFormat> uncompressedFormat;
    std::optional<BlockCompressedFormat> bcFormat;
    std::optional<uint32_t> rowPitch;
    std::optional<uint32_t> pixelStride;
    std::optional<uint32_t> numChannels;

    ColorImageData(IAllocator* allocator)
    { }
};

struct ColorMip
{
    std::optional<uint32_t> width;
    std::optional<uint32_t> height;
    std::optional<uint32_t> latentMip;
    std::optional<float> positionLod;
    std::optional<float> positionScale;
    std::optional<ColorImageData> combinedColorData;
    Vector<ColorImageData> perTextureColorData;

    ColorMip(IAllocator* allocator)
        : perTextureColorData(allocator)
    { }
};

struct Document
{
    static constexpr uint32_t SchemaVersion = 2;

    IAllocator* allocator;

    uint32_t schemaVersion = SchemaVersion;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t numChannels = 0;
    std::optional<uint32_t> numColorMips;
    std::optional<LatentShape> latentShape;
    Vector<MLP> mlpVersions;
    Vector<Texture> textures;
    Vector<Channel> channels;
    Vector<LatentImage> latents;
    Vector<ColorMip> colorMips;
    Vector<BufferView> views;

    Document(IAllocator* allocator)
        : allocator(allocator)
        , mlpVersions(allocator)
        , textures(allocator)
        , channels(allocator)
        , latents(allocator)
        , colorMips(allocator)
        , views(allocator)
    { }
};

// Serializes the document into a JSON string.
// Returns true if successful, false on error.
bool SerializeDocument(Document const& document, String& outString, String& outErrorMessage);

// Parses a JSON string into a document with basic validation of the required fields.
// Returns true if successful, false on error.
// Warning: in-situ parsing is used, input will be corrupted!
bool ParseDocument(Document& outDocument, char* input, String& outErrorMessage);

}