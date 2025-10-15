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

#include "StdTypes.h"
#include "FeatureGridMath.h"
#include <array>

struct NtcDecompressConstants;

namespace ntc
{

class Context;
class TextureMetadata;
class GraphicsResources;
struct MlpDesc;

namespace json
{
    struct Document;
    struct LatentImage;
    struct MLP;
}

struct LatentImageDesc
{
    StreamRange range;
    int width = 0;
    int height = 0;
};

struct ColorMipDesc
{
    int neuralLod = -1;
    float positionLod = 0.f;
    float positionScale = 0.f;
};

class TextureSetMetadata : virtual public ITextureSetMetadata
{
public:
    TextureSetMetadata(IAllocator* allocator, Context const* context, TextureSetDesc const& desc,
        LatentShape const& latentShape);

    TextureSetDesc const& GetDesc() const override { return m_desc; }
    
    LatentShape const& GetLatentShape() const override { return m_latentShape; }

    ITextureMetadata* AddTexture() override;
    Status RemoveTexture(ITextureMetadata* texture) override;
    void ClearTextureMetadata() override;
    int GetTextureCount() const override;
    ITextureMetadata* GetTexture(int textureIndex) override;
    ITextureMetadata const* GetTexture(int textureIndex) const override;
    ColorSpace GetChannelStorageColorSpace(int channel) const override;
    LatentTextureDesc GetLatentTextureDesc() const override;
    Status GetLatentTextureFootprint(int latentMipLevel, LatentTextureFootprint& outFootprint) const override;
    Status GetFusedMipLevels(int mipLevel, int* pOutFirstFusedMip, int* pOutLastFusedMip) const override;
    int GetNumLatentImages() const override;
    Status GetMipLevelsForLatentImage(int latentImageIndex, int* pOutFirstColorMip, int* pOutLastColorMip) const override;
    InferenceWeightType GetBestSupportedWeightType() const override;
    Status GetInferenceWeights(InferenceWeightType weightType, void const** pOutData, size_t* pOutSize,
        size_t* pOutConvertedSize) override;
    Status ConvertInferenceWeights(InferenceWeightType weightType, void* commandList,
        void* srcBuffer, uint64_t srcOffset, void* dstBuffer, uint64_t dstOffset) override;
    bool IsInferenceWeightTypeSupported(InferenceWeightType weightType) const override;
    Status ShuffleInferenceOutputs(ShuffleSource mapping[NTC_MAX_CHANNELS]) override;
    
    Status LoadMetadataFromStream(json::Document const& document, uint64_t binaryChunkOffset,
        uint64_t binaryChunkSize, LatentShape const& latentShape, IStream* inputStream);

    Status LoadWeightsFromStream(json::Document const& document, IStream* inputStream);
    
    void GetWeightOffsets(InferenceWeightType weightType,
        int weightOffsets[NTC_MLP_LAYERS],
        int biasOffsets[NTC_MLP_LAYERS],
        int scaleOffsets[NTC_MLP_LAYERS]) const;

    uint64_t GetSourceStreamSize() const { return m_sourceStreamSize; }

    uint32_t GetValidChannelMask() const override;

    uint32_t GetPackedColorSpaces() const;

    void FillColorMipConstants(NtcColorMipConstants& colorMip, int mipLevel, int firstLatentMipInTexture);

    void FillDecompressionConstants(
        NtcDecompressConstants& output,
        MakeDecompressionComputePassParameters const& params,
        Rect srcRect,
        Point dstOffset);

    bool ValidateBufferView(uint32_t view, uint64_t minSize, json::Document const& document);

    bool ValidateMLP(json::Document const& document, json::MLP const& mlp);

    bool ReadViewFromStream(IStream* stream, json::Document const& document, uint32_t view, void* pData, uint64_t size);

    LatentImageDesc const* GetLatentImageDesc(int neuralLod) const;

    int ColorMipToNeuralLod(int colorMip) const;
    
    bool ConvertWeightsForCoopVec(Vector<uint8_t> const& src, Vector<uint8_t>& dst,
        bool useFP8, size_t& outWeightSize, int outWeightOffsets[4]);

    static Status ValidateLatentShape(LatentShape const& latentShape);

    static Status ValidateTextureSetDesc(TextureSetDesc const& desc);

    static Status DeserializeTextureSetDesc(json::Document const& document,
        TextureSetDesc& desc, LatentShape& latentShape);

    static Status LoadFileHeadersFromStream(IAllocator* allocator, IStream* inputStream, json::Document& outDocument,
        uint64_t& outBinaryChunkOffset, uint64_t& outBinaryChunkSize);
    
protected:
    IAllocator* m_allocator;
    Context const* m_context;

    TextureSetDesc m_desc{ };
    LatentShape m_latentShape = LatentShape::Empty();
    Vector<UniquePtr<TextureMetadata>> m_textureInfos;

    uint64_t m_binaryChunkOffset = 0;
    uint64_t m_binaryChunkSize = 0;
    uint64_t m_sourceStreamSize = 0;
    Vector<uint8_t> m_rowMajorWeightDataInt8;
    Vector<uint8_t> m_rowMajorWeightDataFP8;

    std::array<ColorSpace, NTC_MAX_CHANNELS> m_channelColorSpaces;
    std::array<ShuffleSource, NTC_MAX_CHANNELS> m_channelShuffleMapping;

    Vector<LatentImageDesc> m_latentImages;
    std::array<ColorMipDesc, NTC_MAX_MIPS> m_colorMips;
};

}