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

#include "TextureSetMetadata.h"
#include "CoopVecWeightConverter.h"
#include "Errors.h"
#include "FeatureGridMath.h"
#include "JsonFileFormat.h"
#include "MathUtils.h"
#include "MlpDesc.h"
#include "TextureMetadata.h"
#include "Context.h"
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <cmath>

#include <libntc/shaders/DecompressConstants.h>

namespace ntc
{

constexpr float c_ConstantBiasScale = 65536.f;

TextureSetMetadata::TextureSetMetadata(IAllocator* allocator, Context const* context,
    TextureSetDesc const& desc, LatentShape const& latentShape)
    : m_allocator(allocator)
    , m_context(context)
    , m_desc(desc)
    , m_latentShape(latentShape)
    , m_textureInfos(allocator)
    , m_rowMajorWeightDataInt8(allocator)
    , m_rowMajorWeightDataFP8(allocator)
    , m_latentImages(allocator)
{
    m_channelColorSpaces.fill(ntc::ColorSpace::Linear);

    // Initialize the shuffle pattern with an identity map (i -> i)
    for (int ch = 0; ch < NTC_MAX_CHANNELS; ++ch)
    {
        m_channelShuffleMapping[ch].type = ShuffleSourceType::Channel;
        m_channelShuffleMapping[ch].channelIndex = ch;
    }
}

ITextureMetadata* TextureSetMetadata::AddTexture()
{
    UniquePtr<TextureMetadata> texture = MakeUniqueWithAllocator<TextureMetadata>(m_allocator, m_context, this);
    ITextureMetadata* result = texture.get();
    m_textureInfos.push_back(std::move(texture));
    return result;
}

Status TextureSetMetadata::RemoveTexture(ITextureMetadata* texture)
{
    for (auto it = m_textureInfos.begin(); it != m_textureInfos.end(); ++it)
    {
        if (it->get() == texture)
        {
            m_textureInfos.erase(it);
            return Status::Ok;
        }
    }
    return Status::OutOfRange;
}

void TextureSetMetadata::ClearTextureMetadata()
{
    m_textureInfos.clear();
}

int TextureSetMetadata::GetTextureCount() const
{
    return int(m_textureInfos.size());
}

ITextureMetadata* TextureSetMetadata::GetTexture(int textureIndex)
{
    if (textureIndex < 0 || textureIndex >= int(m_textureInfos.size()))
        return nullptr;

    return m_textureInfos[textureIndex].get();
}

ITextureMetadata const* TextureSetMetadata::GetTexture(int textureIndex) const
{
    if (textureIndex < 0 || textureIndex >= int(m_textureInfos.size()))
        return nullptr;

    return m_textureInfos[textureIndex].get();
}

ColorSpace TextureSetMetadata::GetChannelStorageColorSpace(int channel) const
{
    if (channel < NTC_MAX_CHANNELS)
        return m_channelColorSpaces[channel];
    return ColorSpace::Linear;
}

LatentTextureDesc TextureSetMetadata::GetLatentTextureDesc() const
{
    LatentTextureDesc desc;
    desc.width = m_latentImages[0].width;
    desc.height = m_latentImages[0].height;
    desc.arraySize = FeatureGridMath::GetNumLayers(m_latentShape.numFeatures);
    desc.mipLevels = int(m_latentImages.size());
    return desc;
}

Status TextureSetMetadata::GetLatentTextureFootprint(int latentMipLevel, int arrayLayer,
    LatentTextureFootprint& outFootprint) const
{
    if (latentMipLevel < 0 || latentMipLevel >= m_latentImages.size())
    {
        SetErrorMessage("latentMipLevel (%d) must be between 0 and %d", latentMipLevel, int(m_latentImages.size() - 1));
        return Status::OutOfRange;
    }

    ntc::LatentImageDesc const& latentImage = m_latentImages[latentMipLevel];

    int const arraySize = int(latentImage.footprintsPerLayer.size());
    if (arrayLayer < 0 || arrayLayer >= arraySize)
    {
        SetErrorMessage("arrayLayer (%d) must be between 0 and %d", arrayLayer, arraySize);
        return Status::OutOfRange;
    }

    
    outFootprint.buffer = latentImage.footprintsPerLayer[arrayLayer];
    outFootprint.width = latentImage.width;
    outFootprint.height = latentImage.height;
    outFootprint.rowPitch = outFootprint.width * FeatureGridMath::BytesPerLatentPixel;
    
    return Status::Ok;
}

Status TextureSetMetadata::GetFusedMipLevels(int mipLevel, int* pOutFirstFusedMip, int* pOutLastFusedMip) const
{
    if (mipLevel < 0 || mipLevel >= m_desc.mips)
    {
        SetErrorMessage("mipLevel (%d) must be between 0 and %d", mipLevel, m_desc.mips - 1);
        return Status::OutOfRange;
    }

    int const neuralLod = m_colorMips[mipLevel].neuralLod;

    if (pOutFirstFusedMip)
    {
        // Go down from mipLevel to find the first mismatching neural LOD index
        // Start with mip 0 in case we don't find a mismatching index.
        *pOutFirstFusedMip = 0;
        for (int i = mipLevel - 1; i >= 0; --i)
        {
            if (m_colorMips[i].neuralLod != neuralLod)
            {
                *pOutFirstFusedMip = i + 1;
                break;
            }
        }
    }

    if (pOutLastFusedMip)
    {
        // Go up from mipLevel to find the first mismatching neural LOD index.
        // Start with the last mip in case we don't find a mismatching index.
        *pOutLastFusedMip = m_desc.mips - 1;
        for (int i = mipLevel + 1; i < m_desc.mips; ++i)
        {
            if (m_colorMips[i].neuralLod != neuralLod)
            {
                *pOutLastFusedMip = i - 1;
                return Status::Ok;
            }
        }
    }

    return Status::Ok;
}

int TextureSetMetadata::GetNumLatentImages() const
{
    return int(m_latentImages.size());
}

Status TextureSetMetadata::GetMipLevelsForLatentImage(int latentImageIndex, int* pOutFirstColorMip, int* pOutLastColorMip) const
{
    if (latentImageIndex < 0 || latentImageIndex >= GetNumLatentImages())
    {
        SetErrorMessage("Invalid latentImageIndex (%d), it must be between 0 and %d",
            latentImageIndex, GetNumLatentImages());
        return Status::OutOfRange;
    }
    int first = -1;
    int last = -1;
    for (int mipLevel = 0; mipLevel < m_desc.mips; ++mipLevel)
    {
        auto const& colorMip = m_colorMips[mipLevel];
        if (colorMip.neuralLod == latentImageIndex)
        {
            if (first < 0)
                first = mipLevel;
            last = mipLevel;
        }
        else if (last >= 0)
        {
            break;
        }
    }
    
    if (first < 0)
    {
        SetErrorMessage("No color MIPs found that are represented by latent image %d", latentImageIndex);
        return Status::OutOfRange;
    }

    if (pOutFirstColorMip)
        *pOutFirstColorMip = first;
    if (pOutLastColorMip)
        *pOutLastColorMip = last;

    return Status::Ok;
}

InferenceWeightType TextureSetMetadata::GetBestSupportedWeightType() const
{
    for (auto type : { InferenceWeightType::CoopVecFP8,
                       InferenceWeightType::GenericInt8 })
    {
        if (IsInferenceWeightTypeSupported(type))
            return type;
    }

    return InferenceWeightType::Unknown;
}

Status TextureSetMetadata::GetInferenceWeights(InferenceWeightType weightType,
    void const** pOutData, size_t* pOutSize, size_t* pOutConvertedSize)
{
    WeightLayout const* layout = m_context->GetWeightLayout(weightType);

    if (!IsInferenceWeightTypeSupported(weightType) || layout == nullptr)
    {
        SetErrorMessage("No weights available for weightType = %s.",
            InferenceWeightTypeToString(weightType));
        return Status::Unsupported;
    }

    void const* pData = nullptr;
    size_t size = 0;
    size_t convertedSize = 0;
    
    switch(weightType)
    {
        case InferenceWeightType::GenericInt8:
            size = m_rowMajorWeightDataInt8.size();
            pData = m_rowMajorWeightDataInt8.data();
            break;
        case InferenceWeightType::GenericFP8:
            size = m_rowMajorWeightDataFP8.size();
            pData = m_rowMajorWeightDataFP8.data();
            break;
        case InferenceWeightType::CoopVecFP8:
            size = m_rowMajorWeightDataFP8.size();
            pData = m_rowMajorWeightDataFP8.data();
            convertedSize = layout->bufferSize;
            break;
        default:
            assert(!"Unsupported value - should be verified by IsInferenceWeightTypeSupported above");
            break;
    }

    if (pOutData) *pOutData = pData;
    if (pOutSize) *pOutSize = size;
    if (pOutConvertedSize) *pOutConvertedSize = convertedSize;

    return Status::Ok;
}

bool TextureSetMetadata::IsInferenceWeightTypeSupported(InferenceWeightType weightType) const
{
    if (!m_context->GetWeightLayout(weightType))
        return false;

    switch(weightType)
    {
    case InferenceWeightType::GenericInt8:
        return !m_rowMajorWeightDataInt8.empty();
    case InferenceWeightType::GenericFP8:
    case InferenceWeightType::CoopVecFP8:
        return !m_rowMajorWeightDataFP8.empty();
    default:
        return false;
    }
}

Status TextureSetMetadata::ShuffleInferenceOutputs(ShuffleSource mapping[NTC_MAX_CHANNELS])
{
    if (mapping == nullptr)
    {
        SetErrorMessage("mapping is NULL");
        return Status::InvalidArgument;
    }

    // Validate the mapping
    for (int ch = 0; ch < NTC_MAX_CHANNELS; ++ch)
    {
        switch(mapping[ch].type)
        {
            case ShuffleSourceType::Undefined:
            case ShuffleSourceType::Channel:
            case ShuffleSourceType::Constant:
                break;
            default:
                SetErrorMessage("mapping[%d] has invalid type %d", ch, int(mapping[ch].type));
                return Status::InvalidArgument;
        }

        if (mapping[ch].type == ShuffleSourceType::Channel &&
            (mapping[ch].channelIndex < 0 || mapping[ch].channelIndex >= NTC_MAX_CHANNELS))
        {
            SetErrorMessage("mapping[%d] is using invalid channel %d, must be between 0 and %d",
                ch, mapping[ch].channelIndex, NTC_MAX_CHANNELS - 1);
            return Status::OutOfRange;
        }
    }

    auto shuffleWeights = [this, mapping](Vector<uint8_t>& data, InferenceWeightType weightType)
    {
        if (data.empty())
            return;

        WeightLayout const* layout = m_context->GetWeightLayout(weightType);
        assert(layout);

        Span const& outputLayerWeights = layout->weights[NTC_MLP_LAYERS - 1];
        Span const& outputLayerScales = layout->scales[NTC_MLP_LAYERS - 1];
        Span const& outputLayerBiases = layout->biases[NTC_MLP_LAYERS - 1];

        int const inputChannels = MlpDesc::GetLayerInputChannels(NTC_MLP_LAYERS - 1);
        int const outputChannels = MlpDesc::GetLayerOutputChannels(NTC_MLP_LAYERS - 1);

#if NTC_MLP_LAYERS == 4
        constexpr int OutputLayerInputChannels = NTC_MLP_HIDDEN2_CHANNELS;
#elif NTC_MLP_LAYERS == 3
        constexpr int OutputLayerInputChannels = NTC_MLP_HIDDEN1_CHANNELS;
#else
        #error "Unsupported NTC_MLP_LAYERS value"
#endif
        // Sanity check
        assert(inputChannels == OutputLayerInputChannels);
        assert(outputChannels == NTC_MLP_OUTPUT_CHANNELS);
        static_assert(NTC_MAX_CHANNELS == NTC_MLP_OUTPUT_CHANNELS,
            "This function assumes that NTC_MAX_CHANNELS == NTC_MLP_OUTPUT_CHANNELS");

        // Output layer for Int8 and FP8 modes has 8-bit weights, Float32 scale and Int32 bias
        assert(outputLayerWeights.type == DataType::FP8 || outputLayerWeights.type == DataType::Int8);
        assert(outputLayerScales.type == DataType::FP32);
        assert(outputLayerBiases.type == DataType::Int32);

        uint8_t* weights = data.data() + outputLayerWeights.offset;
        float* scale = (float*)(data.data() + outputLayerScales.offset);
        int32_t* bias = (int32_t*)(data.data() + outputLayerBiases.offset);
        std::array<uint8_t, OutputLayerInputChannels * NTC_MLP_OUTPUT_CHANNELS> tmpWeights;
        std::array<float, NTC_MLP_OUTPUT_CHANNELS> tmpScale;
        std::array<int32_t, NTC_MLP_OUTPUT_CHANNELS> tmpBias;

        // Shuffle the row data into 'tmpWeights', scale and bias into 'tmpScale' and 'tmpBias'
        for (int dstRow = 0; dstRow < outputChannels; ++dstRow)
        {
            ShuffleSource& src = mapping[dstRow];
            uint8_t* pDstRow = tmpWeights.data() + inputChannels * dstRow;
            if (src.type == ShuffleSourceType::Channel)
            {
                uint8_t const* pSrcRow = weights + inputChannels * src.channelIndex;
                memcpy(pDstRow, pSrcRow, inputChannels);
                tmpScale[dstRow] = scale[src.channelIndex];
                tmpBias[dstRow] = bias[src.channelIndex];
            }
            else
            {
                // The row will produce zero as the output of the matrix-vector multiplication, plus (bias * scale).
                // Set the scale and bias to produce the constant value.
                memset(pDstRow, 0, inputChannels);
                tmpScale[dstRow] = 1.f / c_ConstantBiasScale;
                tmpBias[dstRow] = (src.type == ShuffleSourceType::Constant) ? int(roundf(src.constantValue * c_ConstantBiasScale)) : 0;
            }
        }

        // Copy the shuffled data back into the MLP vector
        memcpy(weights, tmpWeights.data(), outputLayerWeights.size);
        memcpy(scale, tmpScale.data(), outputLayerScales.size);
        memcpy(bias, tmpBias.data(), outputLayerBiases.size);
    };

    shuffleWeights(m_rowMajorWeightDataInt8, InferenceWeightType::GenericInt8);
    shuffleWeights(m_rowMajorWeightDataFP8, InferenceWeightType::GenericFP8);

    std::array<ColorSpace, NTC_MAX_CHANNELS> newColorSpaces;
    std::array<ShuffleSource, NTC_MAX_CHANNELS> newMapping;

    for (int ch = 0; ch < NTC_MAX_CHANNELS; ++ch)
    {
        switch(mapping[ch].type)
        {
            case ShuffleSourceType::Undefined:
            case ShuffleSourceType::Constant:
                newColorSpaces[ch] = ColorSpace::Linear;
                newMapping[ch] = mapping[ch];
                break;
            case ShuffleSourceType::Channel: {
                int const srcChannel = mapping[ch].channelIndex;
                newColorSpaces[ch] = m_channelColorSpaces[srcChannel];
                newMapping[ch] = m_channelShuffleMapping[srcChannel];
                break;
            }
        }
    }

    m_channelColorSpaces = newColorSpaces;
    m_channelShuffleMapping = newMapping;

    return Status::Ok;
}

Status TextureSetMetadata::LoadMetadataFromStream(json::Document const& document, uint64_t binaryChunkOffset,
        uint64_t binaryChunkSize, LatentShape const& latentShape, IStream* inputStream)
{
    ClearTextureMetadata();

    m_sourceStreamSize = inputStream->Size();
    m_binaryChunkOffset = binaryChunkOffset;
    m_binaryChunkSize = binaryChunkSize;
    m_latentShape = latentShape;
    
    // MLP versions

    if (document.mlpVersions.empty())
    {
        SetErrorMessage("File doesn't contain MLP information");
        return Status::FileUnrecognized;
    }

    for (json::MLP const& mlp : document.mlpVersions)
    {
        if (!ValidateMLP(document, mlp))
            return Status::FileUnrecognized;
    }

    // Texture headers

    for (auto const& jsonTexture : document.textures)
    {    
        TextureMetadata* texture = static_cast<TextureMetadata*>(AddTexture());
        texture->SetName(jsonTexture.name.c_str());
        texture->SetChannels(jsonTexture.firstChannel, jsonTexture.numChannels);
        texture->SetChannelFormat(jsonTexture.channelFormat.value_or(ChannelFormat::UNORM8));
        texture->SetBlockCompressedFormat(jsonTexture.bcFormat.value_or(BlockCompressedFormat::None));
        texture->SetRgbColorSpace(jsonTexture.rgbColorSpace.value_or(ColorSpace::Linear));
        texture->SetAlphaColorSpace(jsonTexture.alphaColorSpace.value_or(ColorSpace::Linear));
        
        for (auto const& mipData : jsonTexture.bcModeBuffers)
        {
            if (!ValidateBufferView(mipData.view, 0, document))
                return Status::FileUnrecognized;

            texture->SetBC7ModeBufferFootprint(mipData.mipLevel, document.views[mipData.view], m_binaryChunkOffset);
        }
    }

    // Channel headers

    if (!document.channels.empty())
    {
        if (document.channels.size() != m_desc.channels)
        {
            SetErrorMessage("The file describes %d texture channels while %d are expected",
                document.numChannels, m_desc.channels);
            return Status::FileUnrecognized;
        }

        for (size_t i = 0; i < document.channels.size(); ++i)
        {
            json::Channel const& jsonChannel = document.channels[i];
            m_channelColorSpaces[i] = jsonChannel.colorSpace.value_or(ColorSpace::Linear);
        }
    }

    // Validate the neural LOD headers

    int neuralLod = 0;
    m_latentImages.clear();
    for (const auto& mipHeader : document.latents)
    {
        int const numLayers = FeatureGridMath::GetNumLayers(m_latentShape.numFeatures);

        size_t const layerSize = size_t(mipHeader.width) * size_t(mipHeader.height)
            * FeatureGridMath::BytesPerLatentPixel;

        LatentImageDesc& imageDesc = m_latentImages.emplace_back(m_allocator);

        if (mipHeader.layerViews.size() != numLayers)
        {
            SetErrorMessage("Expected %d layers, got %d for the latent image in MIP %d", numLayers,
                int(mipHeader.layerViews.size()), neuralLod);
            return Status::FileUnrecognized;
        }

        for (int layerIndex = 0; layerIndex < numLayers; ++layerIndex)
        {
            uint32_t viewIndex = mipHeader.layerViews[layerIndex];
            if (!ValidateBufferView(viewIndex, layerSize, document))
                return Status::FileUnrecognized;

            json::BufferView const& view = document.views[viewIndex];
        
            BufferFootprint& layerFootprint = imageDesc.footprintsPerLayer.emplace_back();
            layerFootprint.rangeInStream.offset = view.offset + m_binaryChunkOffset;
            layerFootprint.rangeInStream.size = view.storedSize;
            layerFootprint.compressionType = view.compression.value_or(CompressionType::None);
            layerFootprint.uncompressedSize = view.uncompressedSize.value_or(layerFootprint.rangeInStream.size);
            layerFootprint.uncompressedCrc32 = view.crc32.value_or(0);
        }
        
        imageDesc.width = mipHeader.width;
        imageDesc.height = mipHeader.height;

        ++neuralLod;
    }

    // Validate the color MIP levels

    int mipLevel = 0;
    m_colorMips.fill(ColorMipDesc());
    for (auto const& colorMip : document.colorMips)
    {
        if (!colorMip.latentMip.has_value())
        {
            SetErrorMessage("Color MIP %d doesn't have a mapping to a latent image, "
                "which is currently unsupported.", mipLevel);
            return Status::FileUnrecognized;
        }

        m_colorMips[mipLevel].neuralLod = *colorMip.latentMip;
        
        if (!colorMip.positionLod.has_value() || !colorMip.positionScale.has_value())
        {
            SetErrorMessage("Color MIP %d doesn't have the positionLod and positionScale parameters.", mipLevel);
            return Status::FileUnrecognized;
        }

        m_colorMips[mipLevel].positionLod = *colorMip.positionLod;
        m_colorMips[mipLevel].positionScale = *colorMip.positionScale;

        if (colorMip.width.has_value() && colorMip.height.has_value())
        {
            int mipWidth = std::max(m_desc.width >> mipLevel, 1);
            int mipHeight = std::max(m_desc.height >> mipLevel, 1);
            if (*colorMip.width != mipWidth || *colorMip.height != mipHeight)
            {
                SetErrorMessage("Color MIP %d specifies dimensions of %dx%d, "
                    "but it is expected to be %dx%d.", mipLevel,
                    *colorMip.width, *colorMip.height,
                    mipWidth, mipHeight);
                return Status::FileUnrecognized;
            }
        }

        ++mipLevel;
    }

    return Status::Ok;
}

bool TextureSetMetadata::ReadViewFromStream(IStream* stream, json::Document const& document,
     uint32_t view, void* pData, uint64_t size)
{
    json::BufferView const& viewDesc = document.views[view];
    assert(size <= viewDesc.storedSize); // Should be validated by ValidateBufferView before
    return stream->Seek(m_binaryChunkOffset + viewDesc.offset) && stream->Read(pData, size);
}

LatentImageDesc const* TextureSetMetadata::GetLatentImageDesc(int neuralLod) const
{
    if (neuralLod < 0 || neuralLod >= int(m_latentImages.size()))
        return nullptr;

    return &m_latentImages[neuralLod];
}

int TextureSetMetadata::ColorMipToNeuralLod(int colorMip) const
{
    if (colorMip < 0 || colorMip >= m_desc.mips)
        return -1;

    return m_colorMips[colorMip].neuralLod;
}

Status TextureSetMetadata::ConvertInferenceWeights(InferenceWeightType weightType, void* commandList,
    void* srcBuffer, uint64_t srcOffset, void* dstBuffer, uint64_t dstOffset)
{
    if (!commandList)
    {
        SetErrorMessage("commandList is NULL");
        return Status::InvalidArgument;
    }

    if (!srcBuffer)
    {
        SetErrorMessage("srcBuffer is NULL");
        return Status::InvalidArgument;
    }
    
    if (!dstBuffer)
    {
        SetErrorMessage("dstBuffer is NULL");
        return Status::InvalidArgument;
    }

    if (dstBuffer == srcBuffer)
    {
        SetErrorMessage("dstBuffer must not be the same as srcBuffer");
        return Status::InvalidArgument;
    }

    if (!CoopVecWeightConverter::IsCoopVecWeightType(weightType))
    {
        SetErrorMessage("Unsupported weightType (%s), must be one of the CoopVec types",
            InferenceWeightTypeToString(weightType));
        return Status::InvalidArgument;
    }
    
    WeightLayout const* srcLayout = m_context->GetWeightLayout(
        CoopVecWeightConverter::GetGenericWeightType(weightType));

    assert(srcLayout); // Row-major layouts are always available

    WeightLayout const* dstLayout = m_context->GetWeightLayout(weightType);

    if (dstLayout == nullptr)
    {
        SetErrorMessage("The requested conversion operation is not supported by the graphics device or "
            "disabled by the context settings.");
        return Status::Unsupported;
    }

    CoopVecWeightConverter::ConvertWeights(m_context->GetGraphicsResources(),
        *srcLayout, srcBuffer, srcOffset,
        *dstLayout, dstBuffer, dstOffset, commandList);

    return Status::Ok;
}

static void ExpandWeightMatrix(uint8_t* dstWeights, int dstInputChannels, int dstOutputChannels,
    uint8_t const* srcWeights, int srcInputChannels, int srcOutputChannels)
{
    // Copy the provided weights into the top-left corner of the destination matrix.
    // Assume that the destination buffer is already zero-initialized.

    for (int outCh = 0; outCh < srcOutputChannels; ++outCh)
    {
        uint8_t* pDstRow = dstWeights + outCh * dstInputChannels;
        uint8_t const* pSrcRow = srcWeights + outCh * srcInputChannels;
        memcpy(pDstRow, pSrcRow, srcInputChannels);
    }
}

Status TextureSetMetadata::LoadWeightsFromStream(json::Document const& document, IStream* inputStream)
{
    auto readMlpData = [this, &inputStream, &document]
    (json::MLP const& mlp, Vector<uint8_t>& dst, InferenceWeightType weightType)
    {
        WeightLayout const* weightLayout = m_context->GetWeightLayout(weightType);
        dst.resize(weightLayout->bufferSize);

        Vector<uint8_t> tmpWeights(m_allocator);
        
        for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
        {
            json::MLPLayer const& layer = mlp.layers[layerIndex];

            Span const& layerWeights = weightLayout->weights[layerIndex];
            Span const& layerScale = weightLayout->scales[layerIndex];
            Span const& layerBias = weightLayout->biases[layerIndex];

            int const expectedInputChannels = MlpDesc::GetLayerInputChannels(layerIndex);
            int const expectedOutputChannels = MlpDesc::GetLayerOutputChannels(layerIndex);

            if (layer.inputChannels < expectedInputChannels || layer.outputChannels < expectedOutputChannels)
            {
                // If the layer provided by the file is smaller than one that we expect, expand its weights.
                // Read the weights from the file into a temporary buffer first.

                size_t const providedWeightSize = size_t(layer.inputChannels) * size_t(layer.outputChannels) *
                    GetDataTypeSize(layerWeights.type);
                assert(providedWeightSize <= layerWeights.size); // ValidateMLP makes sure of this

                tmpWeights.resize(providedWeightSize);
                if (!ReadViewFromStream(inputStream, document, layer.weightView,
                    tmpWeights.data(), providedWeightSize))
                    return Status::IOError;

                ExpandWeightMatrix(dst.data() + layerWeights.offset, expectedInputChannels, expectedOutputChannels,
                    tmpWeights.data(), layer.inputChannels, layer.outputChannels);
            }
            else
            {
                // Provided layer matches expected size - read weights directly into destination buffer.

                if (!ReadViewFromStream(inputStream, document, layer.weightView,
                    dst.data() + layerWeights.offset, layerWeights.size))
                    return Status::IOError;
            }

            if (layerScale.size != 0)
            {
                uint32_t const providedScaleCount = layer.scaleView.has_value() ? layer.outputChannels : 0;
                size_t const providedScaleSize = providedScaleCount * GetDataTypeSize(layerScale.type);
                assert(providedScaleSize <= layerScale.size); // ValidateMLP makes sure of this
                
                if (layer.scaleView.has_value())
                {
                    if (!ReadViewFromStream(inputStream, document, *layer.scaleView,
                        dst.data() + layerScale.offset, providedScaleSize))
                        return Status::IOError;
                }
                
                if (providedScaleSize < layerScale.size)
                {
                    // None or not enough scale vectors provided - fill the remaining memory with 1.0
                    assert(layerScale.type == DataType::FP32);

                    float* scales = (float*)(dst.data() + layerScale.offset);
                    for (uint32_t i = providedScaleCount; i < uint32_t(expectedOutputChannels); ++i)
                        scales[i] = 1.0f;
                }
            }

            size_t const providedBiasSize = size_t(layer.outputChannels) * GetDataTypeSize(layerBias.type);
            assert(providedBiasSize <= layerBias.size); // ValidateMLP makes sure of this

            // Read the bias vector directly into the destination buffer.
            // If the provided bias is smaller than expected, the remaining values will stay zero-initialized.
            if (!ReadViewFromStream(inputStream, document, layer.biasView,
                dst.data() + layerBias.offset, providedBiasSize))
                return Status::IOError;
        }

        return Status::Ok;
    };

    Status status;

    for (json::MLP const& mlp : document.mlpVersions)
    {
        if (mlp.layers.empty())
            continue;
        
        json::MlpDataType const mlpWeightType = mlp.layers[0].weightType;

        if (mlpWeightType == json::MlpDataType::Int8)
        {
            status = readMlpData(mlp, m_rowMajorWeightDataInt8, InferenceWeightType::GenericInt8);
            if (status != Status::Ok)
                return status;
        }
        else if (mlpWeightType == json::MlpDataType::FloatE4M3)
        {
            status = readMlpData(mlp, m_rowMajorWeightDataFP8, InferenceWeightType::GenericFP8);
            if (status != Status::Ok)
                return status;
        }
    }

    return Status::Ok;
}

void TextureSetMetadata::GetWeightOffsets(InferenceWeightType weightType,
    int weightOffsets[NTC_MLP_LAYERS],
    int biasOffsets[NTC_MLP_LAYERS],
    int scaleOffsets[NTC_MLP_LAYERS]) const
{
    WeightLayout const* layout = m_context->GetWeightLayout(weightType);
    assert(layout); // Support should be validated by the caller

    for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
    {
        weightOffsets[layer] = int(layout->weights[layer].offset);
        biasOffsets[layer] = int(layout->biases[layer].offset);
        scaleOffsets[layer] = int(layout->scales[layer].offset);
    }
}

static uint32_t GetChannelMask(int firstChannel, int numChannels)
{
    return ((1u << numChannels) - 1u) << firstChannel;
}

uint32_t TextureSetMetadata::GetValidChannelMask() const
{
    uint32_t validMask = 0;
    for (auto& texture : m_textureInfos)
    {
        int firstChannel, numChannels;
        texture->GetChannels(firstChannel, numChannels);

        validMask |= GetChannelMask(firstChannel, numChannels);
    }
    
    // Textures are defined in un-shuffled space, but the channels might be shuffled,
    // so shuffle the valid mask as well. Also consider constant outputs as valid.
    uint32_t shuffledMask = 0;
    for (int ch = 0; ch < NTC_MAX_CHANNELS; ++ch)
    {
        ShuffleSource const& src = m_channelShuffleMapping[ch];
        if (src.type == ShuffleSourceType::Channel && (validMask & (1 << src.channelIndex)) != 0 ||
            src.type == ShuffleSourceType::Constant)
        {
            shuffledMask |= (1 << ch);
        }
    }
        
    return shuffledMask;
}

uint32_t TextureSetMetadata::GetPackedColorSpaces() const
{
    // Pack the color space data into the constant, 2 bits per channel.
    // The packing is somewhat fragile; if we add more channels or have more than 4 color spaces, it will have to change.
    static_assert(NTC_MAX_CHANNELS <= 16);
    uint32_t packed = 0;
    for (int channel = 0; channel < NTC_MAX_CHANNELS; ++channel)
    {
        ColorSpace const colorSpace = GetChannelStorageColorSpace(channel);
        assert(int(colorSpace) <= 3);

        packed |= uint32_t(colorSpace) << (2 * channel);
    }
    return packed;
}

void TextureSetMetadata::FillColorMipConstants(
    NtcColorMipConstants& colorMip,
    int mipLevel,
    int firstLatentMipInTexture)
{
    colorMip.neuralMip = m_colorMips[mipLevel].neuralLod - firstLatentMipInTexture;
    colorMip.positionLod = m_colorMips[mipLevel].positionLod;
    colorMip.positionScale = m_colorMips[mipLevel].positionScale;
    colorMip.pad = 0;
}

void TextureSetMetadata::FillDecompressionConstants(
    NtcDecompressConstants& output,
    MakeDecompressionComputePassParameters const& params,
    Rect srcRect,
    Point dstOffset)
{
    int const mipWidth = std::max(m_desc.width >> params.mipLevel, 1);
    int const mipHeight = std::max(m_desc.height >> params.mipLevel, 1);

    output.srcLeft = srcRect.left;
    output.srcTop = srcRect.top;
    output.srcRight = srcRect.left + srcRect.width;
    output.srcBottom = srcRect.top + srcRect.height;
    output.dstLeft = dstOffset.x;
    output.dstTop = dstOffset.y;
    
    output.imageWidth = mipWidth;
    output.imageHeight = mipHeight;
    
    GetWeightOffsets(
        params.weightType,
        output.networkWeightOffsets,
        output.networkBiasOffsets,
        output.networkScaleOffsets);
        
    for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
    {
        output.networkWeightOffsets[layer] += params.weightOffset;
        output.networkBiasOffsets[layer] += params.weightOffset;
        output.networkScaleOffsets[layer] += params.weightOffset;
    }

    OutputTextureDesc const* pOutputTextures = params.pOutputTextures;
    int numOutputTextures = params.numOutputTextures;

    // If there are no user-specified outputs, fill out the output descriptors from metadata
    OutputTextureDesc defaultOutputs[DECOMPRESS_CS_MAX_OUTPUTS] {};
    if (!numOutputTextures)
    {
        pOutputTextures = defaultOutputs;
        numOutputTextures = GetTextureCount();

        for (int index = 0; index < numOutputTextures; ++index)
        {
            ITextureMetadata const* src = GetTexture(index);
            OutputTextureDesc& dst = defaultOutputs[index];

            dst.descriptorIndex = index;
            src->GetChannels(dst.firstChannel, dst.numChannels);
            dst.rgbColorSpace = src->GetRgbColorSpace();
            dst.alphaColorSpace = src->GetAlphaColorSpace();

            // Apply dithering to all UNORM8 textures
            dst.ditherScale = src->GetChannelFormat() == ChannelFormat::UNORM8 ? 1.f / 255.f : 0.f;
        }
    }

    // Fill the output constants from either user-specified pOutputTextures or the automatic values
    output.numOutputs = numOutputTextures;
    for (int index = 0; index < numOutputTextures; ++index)
    {
        OutputTextureDesc const& src = pOutputTextures[index];
        NtcDecompressOutputDesc& dst = output.outputs[index];
        
        dst.firstChannel = src.firstChannel;
        dst.numChannels = src.numChannels;
        dst.dstRgbColorSpace = int(src.rgbColorSpace);
        dst.dstAlphaColorSpace = int(src.alphaColorSpace);
        dst.ditherScale = src.ditherScale;
        dst.quantizationScale = src.quantizationScale;
        dst.invQuantizationScale = (src.quantizationScale > 0.f) ? 1.f / src.quantizationScale : 0.f;
        dst.pad0 = 0;
        dst.pad1 = 0;
        dst.textureIndex = params.firstOutputDescriptorIndex + src.descriptorIndex;
        
        int alphaChannel = dst.firstChannel + 3;
        // TODO: validate that all 3 RGB channels have the same storage color space, or support them being different.
        // It would be really weird if they were different, but still valid through the WriteChannels API
        // and that would lead to incorrect output data in the decompression pass.
        dst.srcRgbColorSpace = int(m_channelColorSpaces[dst.firstChannel]);
        dst.srcAlphaColorSpace = (alphaChannel < NTC_MAX_CHANNELS)
            ? int(m_channelColorSpaces[alphaChannel])
            : int(ColorSpace::Linear);
    }
    
    FillColorMipConstants(output.colorMip, params.mipLevel, params.firstLatentMipInTexture);
}

bool TextureSetMetadata::ValidateBufferView(uint32_t view, uint64_t minSize,
    json::Document const& document)
{
    if (view >= document.views.size())
    {
        SetErrorMessage("Invalid view index %u", view);
        return false;
    }

    json::BufferView const& viewDesc = document.views[view];
    if ((viewDesc.offset & 3) != 0)
    {
        SetErrorMessage("View %u offset %" PRIu64 " is not 4-byte aligned", view, viewDesc.offset);
        return false;
    }

    uint64_t uncompressedSize = viewDesc.compression.value_or(CompressionType::None) != CompressionType::None
        ? viewDesc.uncompressedSize.value_or(0)
        : viewDesc.storedSize;
        
    if (uncompressedSize < minSize)
    {
        SetErrorMessage("View %u size %" PRIu64 " is less than minimum expected size %" PRIu64 ".",
            view, viewDesc.storedSize, minSize);
        return false;
    }

    if (viewDesc.offset + viewDesc.storedSize > m_binaryChunkSize)
    {
        SetErrorMessage("View %u ends at byte offset %" PRIu64 " from the binary chunk start, "
            "which is outside of the chunk size %" PRIu64 ".",
            view, viewDesc.offset + viewDesc.storedSize, m_binaryChunkSize);
        return false;
    }

    return true;
}

static size_t GetMlpDataTypeSize(json::MlpDataType dataType)
{
    switch(dataType)
    {
    case json::MlpDataType::Int8:       return sizeof(uint8_t);
    case json::MlpDataType::FloatE4M3:  return sizeof(uint8_t);
    case json::MlpDataType::FloatE5M2:  return sizeof(uint8_t);
    case json::MlpDataType::Float16:    return sizeof(uint16_t);
    case json::MlpDataType::Float32:    return sizeof(float);
    default: return 0;
    }
}

bool TextureSetMetadata::ValidateMLP(json::Document const& document, json::MLP const& mlp)
{
    if (mlp.layers.size() != NTC_MLP_LAYERS)
    {
        SetErrorMessage("File describes an MLP with %d layers, while only %d layers are supported",
            int(mlp.layers.size()), NTC_MLP_LAYERS);
        return false;
    }

    // Validate the MLP geometry

    if (mlp.weightLayout != json::MatrixLayout::RowMajor)
    {
        SetErrorMessage("Only row-major MLP weight layout is supported at this time.");
        return false;
    }

    // We support two MLP configurations:
    // 1. All layers have Int8 weights, Float32 scale, Int32 bias
    // 2. All layers except the last one have FP8 weights and Float16 bias;
    //    Last layer has Int8 weights, Float32 scale, Int32 bias
    // Validate that the provided config matches one of these.

#if NTC_MLP_LAYERS == 4
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_Int8WeightTypes = {
        json::MlpDataType::Int8, json::MlpDataType::Int8, json::MlpDataType::Int8, json::MlpDataType::Int8 };
    static std::array<std::optional<json::MlpDataType>, NTC_MLP_LAYERS> const c_Int8ScaleTypes = {
        json::MlpDataType::Float32, json::MlpDataType::Float32, json::MlpDataType::Float32, json::MlpDataType::Float32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_Int8BiasTypes = {
        json::MlpDataType::Int32, json::MlpDataType::Int32, json::MlpDataType::Int32, json::MlpDataType::Int32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_FP8WeightTypes = {
        json::MlpDataType::FloatE4M3, json::MlpDataType::FloatE4M3, json::MlpDataType::FloatE4M3, json::MlpDataType::Int8 };
    static std::array<std::optional<json::MlpDataType>, NTC_MLP_LAYERS> const c_FP8ScaleTypes = {
        std::nullopt, std::nullopt, std::nullopt, json::MlpDataType::Float32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_FP8BiasTypes = {
        json::MlpDataType::Float16, json::MlpDataType::Float16, json::MlpDataType::Float16, json::MlpDataType::Int32 };
#elif NTC_MLP_LAYERS == 3
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_Int8WeightTypes = {
        json::MlpDataType::Int8, json::MlpDataType::Int8, json::MlpDataType::Int8 };
    static std::array<std::optional<json::MlpDataType>, NTC_MLP_LAYERS> const c_Int8ScaleTypes = {
        json::MlpDataType::Float32, json::MlpDataType::Float32, json::MlpDataType::Float32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_Int8BiasTypes = {
        json::MlpDataType::Int32, json::MlpDataType::Int32, json::MlpDataType::Int32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_FP8WeightTypes = {
        json::MlpDataType::FloatE4M3, json::MlpDataType::FloatE4M3, json::MlpDataType::Int8 };
    static std::array<std::optional<json::MlpDataType>, NTC_MLP_LAYERS> const c_FP8ScaleTypes = {
        std::nullopt, std::nullopt, json::MlpDataType::Float32 };
    static std::array<json::MlpDataType, NTC_MLP_LAYERS> const c_FP8BiasTypes = {
        json::MlpDataType::Float16, json::MlpDataType::Float16, json::MlpDataType::Int32 };
#else
    #error "Unsupported NTC_MLP_LAYERS value"
#endif

    std::array<json::MlpDataType, NTC_MLP_LAYERS> weightTypes;
    std::array<std::optional<json::MlpDataType>, NTC_MLP_LAYERS> scaleTypes;
    std::array<json::MlpDataType, NTC_MLP_LAYERS> biasTypes;
    for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        json::MLPLayer const& layer = mlp.layers[layerIndex];
        weightTypes[layerIndex] = layer.weightType;
        biasTypes[layerIndex] = layer.biasType;
        scaleTypes[layerIndex] = layer.scaleType;
    }

    bool const isValidInt8MLP =
        weightTypes == c_Int8WeightTypes &&
        scaleTypes == c_Int8ScaleTypes &&
        biasTypes == c_Int8BiasTypes;

    bool const isValidFP8MLP =
        weightTypes == c_FP8WeightTypes &&
        scaleTypes == c_FP8ScaleTypes &&
        biasTypes == c_FP8BiasTypes;

    if (!isValidInt8MLP && !isValidFP8MLP)
    {
        SetErrorMessage("Only Int8 weights + Float32 scale + Int32 bias or FloatE4M3 + Float16 bias"
            " MLP configurations are supported at this time.");
        return false;
    }

    for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        json::MLPLayer const& layer = mlp.layers[layerIndex];
        
        // We allow smaller hidden layers, but not input or output layers.
        bool const allowSmallerInputs = layerIndex > 0;
        bool const allowSmallerOutputs = layerIndex < NTC_MLP_LAYERS - 1;
        
        int const expectedInputChannels = MlpDesc::GetLayerInputChannels(layerIndex);
        int const expectedOutputChannels = MlpDesc::GetLayerOutputChannels(layerIndex);
        
        bool const supportedInputCount = allowSmallerInputs
            ? layer.inputChannels <= expectedInputChannels
            : layer.inputChannels == expectedInputChannels;
        bool const supportedOutputCount = allowSmallerOutputs
            ? layer.outputChannels <= expectedOutputChannels
            : layer.outputChannels == expectedOutputChannels;

        if (!supportedInputCount || !supportedOutputCount)
        {
            SetErrorMessage("File describes MLP layer %d with %d inputs and %d outputs, "
                "but that layer should have %s%d inputs and %s%d outputs.",
                layerIndex,
                layer.inputChannels,
                layer.outputChannels,
                allowSmallerInputs ? "at most " : "", expectedInputChannels,
                allowSmallerOutputs ? "at most " : "", expectedOutputChannels);
            return false;
        }

        size_t const expectedWeightSize = layer.inputChannels * layer.outputChannels *
            GetMlpDataTypeSize(weightTypes[layerIndex]);

        if (!ValidateBufferView(layer.weightView, expectedWeightSize, document))
            return false;

        if (scaleTypes[layerIndex].has_value())
        {
            size_t const expectedScaleSize = layer.outputChannels * GetMlpDataTypeSize(*scaleTypes[layerIndex]);

            if (layer.scaleView.has_value())
            {
                if (!ValidateBufferView(*layer.scaleView, expectedScaleSize, document))
                    return false;
            }
        }

        size_t const expectedBiasSize = layer.outputChannels * GetMlpDataTypeSize(biasTypes[layerIndex]);

        if (!ValidateBufferView(layer.biasView, expectedBiasSize, document))
            return false;
    }

    return true;
}

Status TextureSetMetadata::ValidateLatentShape(LatentShape const& latentShape)
{
    if (latentShape.IsEmpty())
        return Status::Ok;
        
    if (latentShape.numFeatures <= 0 ||
        latentShape.numFeatures > NTC_MLP_FEATURES ||
        (latentShape.numFeatures % NTC_FEATURES_PER_LAYER) != 0)
    {
        SetErrorMessage("Invalid LatentShape: numFeatures (%d) must be a positive multiple of %d up to %d.",
            latentShape.numFeatures, NTC_FEATURES_PER_LAYER, NTC_MLP_FEATURES);
        return Status::OutOfRange;
    }

    if (latentShape.gridSizeScale < 1 || latentShape.gridSizeScale > 6)
    {
        SetErrorMessage("Invalid LatentShape: gridSizeScale (%d) must be between 1 and 6.",
            latentShape.gridSizeScale);
        return Status::OutOfRange;
    }

    return Status::Ok;
}

Status TextureSetMetadata::ValidateTextureSetDesc(const TextureSetDesc& desc)
{
    if (desc.width <= 0 || desc.height <= 0 || desc.channels <= 0 || desc.mips <= 0)
    {
        SetErrorMessage("Invalid TextureSetDesc: width (%d), height (%d), channels (%d) and mips (%d) "
            "must be positive numbers.", desc.width, desc.height, desc.channels, desc.mips);
        return Status::OutOfRange;
    }

    if (desc.channels > NTC_MAX_CHANNELS)
    {
        SetErrorMessage("Invalid TextureSetDesc: too many channels (%d). "
            "Only up to %d channels are supported.", desc.channels, NTC_MAX_CHANNELS);
        return Status::OutOfRange;
    }

    return Status::Ok;
}

Status TextureSetMetadata::DeserializeTextureSetDesc(json::Document const& document, TextureSetDesc& desc,
    LatentShape& latentShape)
{
    desc.width = document.width;
    desc.height = document.height;
    desc.channels = document.numChannels;
    desc.mips = document.numColorMips.value_or(1);
    if (document.latentShape.has_value() && !document.latents.empty())
    {
        latentShape.numFeatures = document.latentShape->numFeatures;
        
        json::LatentImage const& firstLatents = document.latents[0];
        float const widthRatio = float(desc.width) / float(std::max(firstLatents.width, 1u));
        float const heightRatio = float(desc.height) / float(std::max(firstLatents.height, 1u));
        latentShape.gridSizeScale = int(roundf(std::min(widthRatio, heightRatio)));
    }
    else
        latentShape = LatentShape::Empty();

    ntc::Status status = ValidateTextureSetDesc(desc);
    if (status != Status::Ok)
        return status;

    status = ValidateLatentShape(latentShape);
    if (status != Status::Ok)
        return status;

    return Status::Ok;
}

Status TextureSetMetadata::LoadFileHeadersFromStream(IAllocator* allocator, IStream* inputStream,
    json::Document& outDocument, uint64_t& outBinaryChunkOffset, uint64_t& outBinaryChunkSize)
{
    json::FileHeader fileHeader;
    if (!inputStream->Read(&fileHeader, sizeof fileHeader))
    {
        SetErrorMessage("Failed to read the file header - file smaller than %zu bytes?", sizeof(fileHeader));
        return Status::IOError;
    }

    if (fileHeader.signature != json::FileHeader::SignatureValue)
    {
        SetErrorMessage("Unrecognized file format.");
        return Status::FileUnrecognized;
    }

    if (fileHeader.version != json::FileHeader::CurrentVersion)
    {
        SetErrorMessage("Incompatible file format version: expected %d, got %d.",
            json::FileHeader::CurrentVersion, fileHeader.version);
        return Status::FileUnrecognized;
    }

    uint64_t const streamSize = inputStream->Size();
    uint64_t const expectedStreamSize = std::max(
        fileHeader.jsonChunkOffset + fileHeader.jsonChunkSize,
        fileHeader.binaryChunkOffset + fileHeader.binaryChunkSize);
    if (streamSize < expectedStreamSize)
    {
        SetErrorMessage("File incomplete: expected at least %" PRIu64 " bytes, actual size %" PRIu64 " bytes.",
            expectedStreamSize, streamSize);
        return Status::FileUnrecognized;
    }

    if (!inputStream->Seek(fileHeader.jsonChunkOffset))
        return Status::IOError;
    
    Vector<char> jsonData(allocator);
    jsonData.resize(fileHeader.jsonChunkSize + 1);
    if (!inputStream->Read(jsonData.data(), fileHeader.jsonChunkSize))
        return Status::IOError;
    jsonData[fileHeader.jsonChunkSize] = 0;

    String errorMessage(allocator);
    if (!json::ParseDocument(outDocument, jsonData.data(), errorMessage))
    {
        SetUnformattedErrorMessage(errorMessage.c_str());
        return Status::FileUnrecognized;
    }

    outBinaryChunkOffset = fileHeader.binaryChunkOffset;
    outBinaryChunkSize = fileHeader.binaryChunkSize;

    return Status::Ok;
}

}
