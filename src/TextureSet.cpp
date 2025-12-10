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

#include "TextureSet.h"
#include "BinaryChunkBuilder.h"
#include "Context.h"
#include "CudaDeviceGuard.h"
#include "CudaRandomGen.h"
#include "Errors.h"
#include "GDeflate.h"
#include "JsonFileFormat.h"
#include "MathUtils.h"
#include "MlpDesc.h"
#include "Optimizer.h"
#include "Quantizer.h"
#include "Regression.h"
#include "SharedTexture.h"
#include "TextureMetadata.h"
#include <cassert>
#include <cinttypes>

namespace ntc
{

constexpr size_t c_PixelsPerKPixel = 1024; // Obvious, but literals are worse

static const char* NetworkStateToString(TextureSetNetworkState state)
{
    switch(state)
    {
    case TextureSetNetworkState::Empty:
        return "Empty";
    case TextureSetNetworkState::Initialized:
        return "Initialized";
    case TextureSetNetworkState::TrainingInProgress:
        return "TrainingInProgress";
    case TextureSetNetworkState::TrainingFinished:
        return "TrainingFinished";
    case TextureSetNetworkState::Complete:
        return "Complete";
    default:
        static char string[16];
        snprintf(string, sizeof(string), "%d", int(state));
        return string;
    }
}

TextureSet::TextureSet(IAllocator* allocator, Context const* context, const TextureSetDesc& desc)
    : TextureSetMetadata(allocator, context, desc, LatentShape::Empty())
    , m_featureGrid(allocator)
    , m_mlpDataInt8(allocator)
    , m_mlpDataFP8(allocator)
    , m_lossReduction(allocator)
{
}

TextureSet::~TextureSet()
{
    if (m_eventStart)
    {
        cudaEventDestroy(m_eventStart);
        m_eventStart = nullptr;
    }

    if (m_eventStop)
    {
        cudaEventDestroy(m_eventStop);
        m_eventStop = nullptr;
    }
}

Status TextureSet::Initialize(const TextureSetFeatures& features)
{
    m_features = features;

    // Round up the channel count to a multiple of 2
    m_desc.channels = (m_desc.channels + 1) & ~1;

    int mipWidth = m_desc.width;
    int mipHeight = m_desc.height;
    uint64_t mipDataOffset = 0;
    m_textureMipOffsets.fill(0);
    for (int mipLevel = 0; mipLevel < m_desc.mips; ++mipLevel)
    {
        uint64_t mipSize = uint64_t(mipWidth) * uint64_t(mipHeight);
        m_textureMipOffsets[mipLevel] = mipDataOffset;
        mipDataOffset += mipSize;

        mipWidth = std::max(1, mipWidth >> 1);
        mipHeight = std::max(1, mipHeight >> 1);
    }
    m_textureMipOffsets[m_desc.mips] = mipDataOffset;

    const size_t textureDataLength = mipDataOffset * m_desc.channels;
    if (!m_textureData.Allocate(textureDataLength))
    {
        SetErrorMessage("Failed to allocate %zu bytes of device memory for the reference texture data.",
            m_textureData.Size());
        return Status::OutOfMemory;
    }

    if (features.separateRefOutData && !m_textureDataOut.Allocate(textureDataLength))
    {
        SetErrorMessage("Failed to allocate %zu bytes of device memory for the output texture data.",
            m_textureDataOut.Size());
        return Status::OutOfMemory;
    }

    cudaError_t err;
    err = cudaMemset(m_textureData.DevicePtr(), 0, m_textureData.Size());
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaMemset", err);
        return Status::CudaError;
    }

    int const stagingWidth = features.stagingWidth > 0 ? features.stagingWidth : m_desc.width;
    int const stagingHeight = features.stagingHeight > 0 ? features.stagingHeight : m_desc.height;
    
    size_t stagingSize = size_t(stagingWidth) * size_t(stagingHeight) * size_t(features.stagingBytesPerPixel);
    if (stagingSize != 0 && !m_textureStaging.Allocate(stagingSize))
    {
        SetErrorMessage("Failed to allocate %zu bytes of device memory for the staging buffer.",
            m_textureStaging.Size());
        return Status::OutOfMemory;
    }
    
    err = cudaEventCreate(&m_eventStart);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaEventCreate", err);
        return Status::CudaError;
    }

    err = cudaEventCreate(&m_eventStop);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaEventCreate", err);
        return Status::CudaError;
    }

    return Status::Ok;
}

Status TextureSet::LoadFromStreamPostHeader(json::Document const& document, uint64_t binaryChunkOffset,
        uint64_t binaryChunkSize, IStream* inputStream, LatentShape latentShape)
{
    Status status = LoadMetadataFromStream(document, binaryChunkOffset, binaryChunkSize, latentShape, inputStream);
    if (status != Status::Ok)
        return status;

    // Validate the neural LOD dimensions here because TextureSetMetadata doesn't do that,
    // it is supposed to accept any sizes of latent images and any color->neural mapping.
    // The full TextureSet implementation relies on a specific geometry though.
    for (int neuralLod = 0; neuralLod < m_latentImages.size(); ++neuralLod)
    {
        LatentImageDesc const& latentImage = m_latentImages[neuralLod];

        int const width = FeatureGridMath::GetGridDimension(m_desc.width, neuralLod, latentShape.gridSizeScale);
        int const height = FeatureGridMath::GetGridDimension(m_desc.height, neuralLod, latentShape.gridSizeScale);

        if (latentImage.width != width || latentImage.height != height)
        {
            SetErrorMessage("Neural MIP %d dimensions (%dx%d) don't match "
                "the expected dimensions (%dx%d)",
                neuralLod,
                latentImage.width,
                latentImage.height,
                width,
                height);
            return Status::FileUnrecognized;
        }
    }

    // Validate the color->neural LOD mapping, same reason as neural LOD dimensions above.
    for (int mipLevel = 0; mipLevel < m_desc.mips; ++mipLevel)
    {
        int const expectedNeuralLod = FeatureGridMath::LodToNeuralLod(mipLevel,
            m_latentShape.gridSizeScale, GetNumLatentImages());

        if (m_colorMips[mipLevel].neuralLod != expectedNeuralLod)
        {
            SetErrorMessage("Color MIP %d specifies latent image index %d, but it is expected to be %d.",
                mipLevel, m_colorMips[mipLevel].neuralLod, expectedNeuralLod);
            return Status::FileUnrecognized;
        }
    }
    
    // Reset m_latentShape so that SetLatentShape doesn't exit right away
    m_latentShape = LatentShape::Empty();

    status = SetLatentShape(latentShape);
    if (status != Status::Ok)
        return status;

    // MLP data
    
    status = LoadWeightsFromStream(document, inputStream);
    if (status != Status::Ok)
        return status;

    if (m_rowMajorWeightDataInt8.data())
    {
        if (m_mlpDataInt8.Size() < m_rowMajorWeightDataInt8.size())
        {
            SetErrorMessage("Inconsistent sizes for MLP data");
            return Status::InternalError;
        }

        memcpy(m_mlpDataInt8.HostPtr(), m_rowMajorWeightDataInt8.data(), m_rowMajorWeightDataInt8.size());
    }

    if (m_rowMajorWeightDataFP8.data())
    {
        if (m_mlpDataFP8.Size() < m_rowMajorWeightDataFP8.size())
        {
            SetErrorMessage("Inconsistent sizes for MLP data");
            return Status::InternalError;
        }

        memcpy(m_mlpDataFP8.HostPtr(), m_rowMajorWeightDataFP8.data(), m_rowMajorWeightDataFP8.size());
    }
    
    // Latents data
    for (int neuralLod = 0; neuralLod < m_featureGrid.GetNumMipLevels(); ++neuralLod)
    {
        json::LatentImage const& latentImage = document.latents[neuralLod];
        for (int layerIndex = 0; layerIndex < m_featureGrid.GetNumLayers(); ++layerIndex)
        {
            uint32_t viewIndex = latentImage.layerViews[layerIndex];
            json::BufferView const& view = document.views[viewIndex];
            CompressionType compression = view.compression.value_or(CompressionType::None);

            if (compression == CompressionType::GDeflate)
            {
                Vector<uint8_t> compressedData(m_allocator);
                compressedData.resize(view.storedSize);
                if (!ReadViewFromStream(inputStream, document, viewIndex, compressedData.data(), compressedData.size()))
                    return Status::IOError;

                ntc::Status status = DecompressGDeflate(
                    compressedData.data(),
                    compressedData.size(),
                    m_featureGrid.GetEncodedPixelsHostPtr(neuralLod, layerIndex),
                    m_featureGrid.GetEncodedPixelsSizePerLayer(neuralLod),
                    m_allocator,
                    view.crc32.value_or(0));

                if (status != Status::Ok)
                {
                    return status;
                }
            }
            else if (compression == CompressionType::None)
            {
                if (!ReadViewFromStream(inputStream, document, viewIndex,
                    m_featureGrid.GetEncodedPixelsHostPtr(neuralLod, layerIndex),
                    m_featureGrid.GetEncodedPixelsSizePerLayer(neuralLod)))
                    return Status::IOError;
                }
            else
            {
                SetErrorMessage("Latent image view %d has unsupported compression type %d.",
                    viewIndex, int(compression));
                return Status::FileUnrecognized;
            }
        }
    }

    // BC7 mode buffers.
    // They're likely not needed right away, but we need to load them now in case the texture set will be saved later.
    for (auto& textureInfo : m_textureInfos)
    {
        // The stream ranges were already loaded in LoadMetadataFromStream
       status = textureInfo->LoadBC7ModeBuffers(inputStream);
       if (status != Status::Ok)
           return status;
    }

    // Deserialized network is equivalent to one that's just completed training, both can be decompressed.
    m_networkState = TextureSetNetworkState::Complete;

    ClearErrorMessage();
    return Status::Ok;
}

static size_t GetDecompressionLossItemsPerChannel(int width, int height)
{
    size_t lossItemsPerChannel = (size_t(width) * size_t(height) + LOCAL_PIXELS - 1) / LOCAL_PIXELS;
    // Round up to a multiple of 4 to satisfy the alignment requirements in ReduceLoss
    lossItemsPerChannel = (lossItemsPerChannel + 3) & ~3;
    return lossItemsPerChannel ;
}

Status TextureSet::SetLatentShape(LatentShape const& newShape)
{
    // Early out if we already have the same shape
    if (m_latentShape == newShape)
        return Status::Ok;

    Status status = ValidateLatentShape(newShape);
    if (status != Status::Ok)
        return status;
        
    CudaDeviceGuard cudaGuard(m_context);
    if (!cudaGuard.Success())
        return Status::CudaError;

    m_latentShape = newShape;

    m_networkState = TextureSetNetworkState::Empty;

    // Deallocate all the TextureSet buffers in case they existed before
    m_featureGrid.Deallocate();
    m_mlpWeightsQuantized.Deallocate();
    m_mlpDataInt8.Deallocate();
    m_mlpDataFP8.Deallocate();
    m_weightGradients.Deallocate();
    m_mlpWeightsBase.Deallocate();
    m_mlpMoment1.Deallocate();
    m_mlpMoment2.Deallocate();
    m_loss.Deallocate();
    m_lossReduction.Deallocate();

    // Early out if the new shape is empty
    if (newShape.IsEmpty())
    {
        ClearErrorMessage();
        return Status::Ok;
    }

    status = m_featureGrid.Initialize(m_desc.width, m_desc.height, m_desc.mips, m_latentShape.gridSizeScale,
        m_latentShape.numFeatures, m_features.enableCompression);

    if (status != Status::Ok)
    {
        // TODO: move this to FeatureGrid, expand the specific errors.
        SetErrorMessage("Failed to initialize the feature grid.");
        return status;
    }
    
    // Trainable MLP parameters: weights and bias. No scales at training time, they are added during quantization.
    m_numNetworkParams = MlpDesc::GetTotalWeightCount() + MlpDesc::GetTotalOutputCount();

    if (!m_mlpWeightsQuantized.Allocate(m_numNetworkParams * 2))
    {
        SetErrorMessage("Failed to allocate %zu bytes of device+host memory for the MLP "
            "weights buffer (quantized).", m_mlpWeightsQuantized.Size());
        return Status::OutOfMemory;
    }
    
    WeightLayout const* int8WeightLayout = m_context->GetWeightLayout(InferenceWeightType::GenericInt8);
    WeightLayout const* fp8WeightLayout = m_context->GetWeightLayout(InferenceWeightType::GenericFP8);
    
    if (!m_mlpDataInt8.Allocate(int8WeightLayout->bufferSize))
    {
        SetErrorMessage("Failed to allocate %zu bytes of device+host memory "
            "for the Int8 MLP data buffer.", m_mlpDataInt8.Size());
        return Status::OutOfMemory;
    }

    if (!m_mlpDataFP8.Allocate(fp8WeightLayout->bufferSize))
    {
        SetErrorMessage("Failed to allocate %zu bytes of device+host memory "
            "for the FP8 MLP data buffer.", m_mlpDataFP8.Size());
        return Status::OutOfMemory;
    }

    size_t requiredLossLength = 0;
    
    if (m_features.enableCompression)
    {
        // Allocate the loss array for the maximum supported batch size.
        // With the current values (NTC_MAX_KPIXELS_PER_BATCH = 2048, LOCAL_PIXELS = 64), that's just 32K floats.
        constexpr size_t maxPixelsPerBatch = NTC_MAX_KPIXELS_PER_BATCH * c_PixelsPerKPixel;
        constexpr size_t lossLength = maxPixelsPerBatch / LOCAL_PIXELS;
        requiredLossLength = lossLength;
        
        constexpr size_t maxGradientSlices = (NTC_MAX_KPIXELS_PER_BATCH * c_PixelsPerKPixel) 
            / (TILE_SIZE_X * TB_SIZE_Y); // assume NW_GRAD_ATOMICS = false

        if (!m_weightGradients.Allocate(m_numNetworkParams * maxGradientSlices))
        {
            SetErrorMessage("Failed to allocate %zu bytes of device memory "
                "for the weight gradients buffer.", m_weightGradients.Size());
            return Status::OutOfMemory;
        }

        // Two versions of MLP weights - for int8 and fp8 optimization
        if (!m_mlpWeightsBase.Allocate(m_numNetworkParams * 2))
        {
            SetErrorMessage("Failed to allocate %zu bytes of device memory "
                "for the MLP weights buffer (base).", m_mlpWeightsBase.Size());
            return Status::OutOfMemory;
        }

        if (!m_mlpMoment1.Allocate(m_numNetworkParams))
        {
            SetErrorMessage("Failed to allocate %zu bytes of device memory "
                "for the MLP 1st moments buffer.", m_mlpMoment1.Size());
            return Status::OutOfMemory;
        }

        if (!m_mlpMoment2.Allocate(m_numNetworkParams))
        {
            SetErrorMessage("Failed to allocate %zu bytes of device memory "
                "for the MLP 2nd moments buffer.", m_mlpMoment2.Size());
            return Status::OutOfMemory;
        }
    }
    
    // Loss for the CUDA decompression pass
    size_t const lossLength = GetDecompressionLossItemsPerChannel(m_desc.width, m_desc.height) * NTC_MAX_CHANNELS;
    requiredLossLength = std::max(requiredLossLength, lossLength);
    
    if (!m_loss.Allocate(requiredLossLength))
    {
        SetErrorMessage("Failed to allocate %zu bytes of device memory "
            "for the loss buffer.", m_loss.Size());
        return Status::OutOfMemory;
    }

    size_t lossReductionLength = (requiredLossLength + cuda::LOSS_ITEMS_PER_GROUP - 1) / cuda::LOSS_ITEMS_PER_GROUP;

    if (!m_lossReduction.Allocate(lossReductionLength))
    {
        SetErrorMessage("Failed to allocate %zu bytes of device+host memory "
            "for the loss reduction buffer.", m_lossReduction.Size());
        return Status::OutOfMemory;
    }

    // Fill the latent image dimension cache
    m_latentImages.clear();
    for (int neuralLod = 0; neuralLod < m_featureGrid.GetNumMipLevels(); ++neuralLod)
    {
        LatentImageDesc& imageDesc = m_latentImages.emplace_back(m_allocator);
        imageDesc.width = FeatureGridMath::GetGridDimension(m_desc.width, neuralLod, m_latentShape.gridSizeScale);
        imageDesc.height = FeatureGridMath::GetGridDimension(m_desc.height, neuralLod, m_latentShape.gridSizeScale);
    }

    // Fill the neural LOD indexing cache
    m_colorMips.fill(ColorMipDesc());
    for (int mipLevel = 0; mipLevel < m_desc.mips; ++mipLevel)
    {
        ColorMipDesc& colorMip = m_colorMips[mipLevel];
        colorMip.neuralLod = m_featureGrid.LodToNeuralLod(mipLevel);

        int mipWidth = std::max(1, m_desc.width >> mipLevel);
        int mipHeight = std::max(1, m_desc.height >> mipLevel);
        float widthScale = float(m_latentImages[colorMip.neuralLod].width) / float(mipWidth);
        float heightScale = float(m_latentImages[colorMip.neuralLod].height) / float(mipHeight);
        colorMip.positionScale = 0.5f * std::max(widthScale, heightScale);
        colorMip.positionLod = std::max(-1.f, std::min(1.f, 0.25f * log2f(colorMip.positionScale)));
    }

    ClearErrorMessage();
    return Status::Ok;
}

uint64_t TextureSet::GetOutputStreamSize()
{
    // Headers
    uint64_t size = json::JsonChunkSizeLimit;
    
    // Texture names and BC acceleration data
    for (const auto& info : m_textureInfos)
    {
        size += info->GetNameString().size();
        for (int mipLevel = 0; mipLevel < m_desc.mips; ++mipLevel)
        {
            size_t mipLevelModeBufferSize = 0;
            info->GetBC7ModeBuffer(mipLevel, nullptr, &mipLevelModeBufferSize);
            size += mipLevelModeBufferSize;
        }
    }
    
    // MLP
    size += m_mlpDataInt8.Size();
    size += m_mlpDataFP8.Size();

    // Latents
    size += m_featureGrid.GetTotalPixelCount() * m_featureGrid.GetNumLayers() * FeatureGridMath::BytesPerLatentPixel;

    size = RoundUp4(size);
    
    return size;
}

Status TextureSet::SaveToStream(IStream* outputStream, LosslessCompressionStats* pOutCompressionStats)
{
    if (!outputStream)
    {
        SetErrorMessage("outputStream is NULL.");
        return Status::InvalidArgument;
    }

    if (m_networkState != TextureSetNetworkState::Complete)
    {
        SetErrorMessage("Invalid network state for SaveToStream (%s), must be Complete.",
            NetworkStateToString(m_networkState));
        return Status::InvalidState;
    }
    
    BinaryChunkBuilder builder(m_allocator, m_losslessCompression);

    json::Document document(m_allocator);
    document.width = m_desc.width;
    document.height = m_desc.height;
    document.numChannels = m_desc.channels;
    document.numColorMips = m_desc.mips;

    document.latentShape = json::LatentShape(m_allocator);
    document.latentShape->numFeatures = m_latentShape.numFeatures;
    
    document.mlpVersions.reserve(2);

    json::MLP& mlpInt8 = document.mlpVersions.emplace_back(m_allocator);
    mlpInt8.activation = json::ActivationType::HGELUClamp;
    mlpInt8.weightLayout = json::MatrixLayout::RowMajor;

    json::MLP& mlpFP8 = document.mlpVersions.emplace_back(m_allocator);
    mlpFP8.activation = json::ActivationType::HGELUClamp;
    mlpFP8.weightLayout = json::MatrixLayout::RowMajor;

    WeightLayout const* weightLayoutInt8 = m_context->GetWeightLayout(InferenceWeightType::GenericInt8);
    assert(weightLayoutInt8);

    WeightLayout const* weightLayoutFP8 = m_context->GetWeightLayout(InferenceWeightType::GenericFP8);
    assert(weightLayoutFP8);

    // Fill out the MLP layers - Int8
    for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        json::MLPLayer& layer = mlpInt8.layers.emplace_back(m_allocator);
        layer.inputChannels = MlpDesc::GetLayerInputChannels(layerIndex);
        layer.outputChannels = MlpDesc::GetLayerOutputChannels(layerIndex);
        layer.weightType = json::MlpDataType::Int8;
        layer.scaleType = json::MlpDataType::Float32;
        layer.biasType = json::MlpDataType::Int32;
        layer.weightView = builder.AllocateViewAndRegisterData(m_mlpDataInt8.HostPtrOffset(
            weightLayoutInt8->weights[layerIndex].offset),
            weightLayoutInt8->weights[layerIndex].size, false);
        layer.scaleView = builder.AllocateViewAndRegisterData(m_mlpDataInt8.HostPtrOffset(
            weightLayoutInt8->scales[layerIndex].offset),
            weightLayoutInt8->scales[layerIndex].size, false);
        layer.biasView = builder.AllocateViewAndRegisterData(m_mlpDataInt8.HostPtrOffset(
            weightLayoutInt8->biases[layerIndex].offset),
            weightLayoutInt8->biases[layerIndex].size, false);
    }

    // Fill out the MLP layers - FP8
    // The MLP versions need to be in separate loops to keep view offsets consistent
    // with the actual writing order below
    for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        json::MLPLayer& layer = mlpFP8.layers.emplace_back(m_allocator);
        layer.inputChannels = MlpDesc::GetLayerInputChannels(layerIndex);
        layer.outputChannels = MlpDesc::GetLayerOutputChannels(layerIndex);
        layer.weightView = builder.AllocateViewAndRegisterData(m_mlpDataFP8.HostPtrOffset(
            weightLayoutFP8->weights[layerIndex].offset),
            weightLayoutFP8->weights[layerIndex].size, false);

        if (layerIndex == NTC_MLP_LAYERS - 1)
        {
            layer.weightType = json::MlpDataType::Int8;
            layer.scaleType = json::MlpDataType::Float32;
            layer.biasType = json::MlpDataType::Int32;
            layer.scaleView = builder.AllocateViewAndRegisterData(m_mlpDataFP8.HostPtrOffset(
                weightLayoutFP8->scales[layerIndex].offset),
                weightLayoutFP8->scales[layerIndex].size, false);
        }
        else
        {
            layer.weightType = json::MlpDataType::FloatE4M3;
            layer.biasType = json::MlpDataType::Float16;
        }

        layer.biasView = builder.AllocateViewAndRegisterData(m_mlpDataFP8.HostPtrOffset(
            weightLayoutFP8->biases[layerIndex].offset),
            weightLayoutFP8->biases[layerIndex].size, false);
    }

    // Fill out the textures
    for (const auto& info : m_textureInfos)
    {
        json::Texture& texture = document.textures.emplace_back(m_allocator);
        texture.name = info->GetNameString();
        texture.firstChannel = info->GetFirstChannel();
        texture.numChannels = info->GetNumChannels();
        texture.channelFormat = info->GetChannelFormat();

        if (info->GetRgbColorSpace() != ColorSpace::Linear)
            texture.rgbColorSpace = info->GetRgbColorSpace();

        if (info->GetAlphaColorSpace() != ColorSpace::Linear)
            texture.alphaColorSpace = info->GetAlphaColorSpace();

        if (info->GetBlockCompressedFormat() != BlockCompressedFormat::None)
            texture.bcFormat = info->GetBlockCompressedFormat();
        
    }

    // Fill out the channels
    for (int channelIndex = 0; channelIndex < m_desc.channels; ++channelIndex)
    {
        json::Channel& channel = document.channels.emplace_back(m_allocator);
        if (m_channelColorSpaces[channelIndex] != ColorSpace::Linear)
            channel.colorSpace = m_channelColorSpaces[channelIndex];
    }

    // Fill out the latent image descriptors
    for (int neuralLod = 0; neuralLod < m_featureGrid.GetNumMipLevels(); ++neuralLod)
    {
        LatentImageDesc const& src = m_latentImages[neuralLod];
        json::LatentImage& image = document.latents.emplace_back(m_allocator);
        image.width = src.width;
        image.height = src.height;

        size_t const pixelsSize = m_featureGrid.GetEncodedPixelsSizePerLayer(neuralLod);
        for (int layerIndex = 0; layerIndex < m_featureGrid.GetNumLayers(); ++layerIndex)
        {
            uint32_t const viewIndex = builder.AllocateViewAndRegisterData(
                m_featureGrid.GetEncodedPixelsHostPtr(neuralLod, layerIndex),
                m_featureGrid.GetEncodedPixelsSizePerLayer(neuralLod),
                m_losslessCompression.compressLatents);

            image.layerViews.push_back(viewIndex);
        }
    }

    // Fill out the color MIP descriptors
    for (int mipLevel = 0; mipLevel < m_desc.mips; ++mipLevel)
    {
        json::ColorMip& mip = document.colorMips.emplace_back(m_allocator);
        mip.width = std::max(m_desc.width >> mipLevel, 1);
        mip.height = std::max(m_desc.height >> mipLevel, 1);
        mip.latentMip = m_colorMips[mipLevel].neuralLod;
        mip.positionLod = m_colorMips[mipLevel].positionLod;
        mip.positionScale = m_colorMips[mipLevel].positionScale;
    }

    // Allocate views for the texture BC7 mode buffers - it's probably better to keep them at the end of the file
    for (size_t textureIndex = 0; textureIndex < m_textureInfos.size(); ++textureIndex)
    {
        json::Texture& texture = document.textures[textureIndex];
        TextureMetadata& info = *m_textureInfos[textureIndex];

        for (int mipLevel = 0; mipLevel < m_desc.mips; ++mipLevel)
        {
            ModeBufferInfo const* versions = info.GetModeBufferInfo(mipLevel);
            if (!versions || versions->data.empty())
                continue;

            json::BCModeBuffer& bc7ModeBuffer = texture.bcModeBuffers.emplace_back(m_allocator);
            bc7ModeBuffer.mipLevel = mipLevel;
            bc7ModeBuffer.view = builder.AllocateViewAndRegisterData(versions->data.data(),
                versions->data.size(), m_losslessCompression.compressBCModeBuffers);
        }
    }

    // Export the view information from the builder into the JSON document.
    // This has to be done after all the views have been allocated.
    builder.WriteViewInfosToDocument(document);

    // Serialize the document into a JSON string
    String jsonString(m_allocator);
    String errorMessage(m_allocator);
    if (!json::SerializeDocument(document, jsonString, errorMessage))
    {
        // Serialization failed - that should never happen if the saving code is correct.
        SetUnformattedErrorMessage(errorMessage.c_str());
        return Status::InternalError;
    }

    // Write the container header
    json::FileHeader header;
    header.jsonChunkOffset = sizeof(json::FileHeader);
    header.jsonChunkSize = jsonString.size() + 1;
    header.binaryChunkOffset = RoundUp4(header.jsonChunkOffset + header.jsonChunkSize);
    header.binaryChunkSize = builder.GetBinaryChunkSize();

    if (!outputStream->Write(&header, sizeof(header)))
        return Status::IOError;

    if (!outputStream->Write(jsonString.c_str(), jsonString.size() + 1))
        return Status::IOError;

    if (!builder.WriteAllViewsToStream(outputStream, header.binaryChunkOffset))
        return Status::IOError;

    // Verify that we've written no more than the number of bytes predicted by GetOutputStreamSize
    uint64_t expectedSize = GetOutputStreamSize();
    uint64_t actualSize = outputStream->Tell();

    if (actualSize > expectedSize)
    {
        SetErrorMessage("SaveToStream produced a stream with %" PRIu64 " bytes, while GetOutputStreamSize "
            "predicted no more than %" PRIu64 " bytes.", actualSize, expectedSize);
        return Status::InternalError;
    }

    if (pOutCompressionStats)
        *pOutCompressionStats = builder.GetStatistics();

    ClearErrorMessage();
    return Status::Ok;
}

Status TextureSet::LoadFromStream(IStream* stream)
{
    if (!stream)
    {
        SetErrorMessage("inputStream is NULL.");
        return Status::InvalidArgument;
    }

    json::Document document(m_allocator);
    uint64_t binaryChunkOffset, binaryChunkSize;
    Status status = LoadFileHeadersFromStream(m_allocator, stream, document, binaryChunkOffset, binaryChunkSize);
    if (status != Status::Ok)
        return status;

    TextureSetDesc desc;
    LatentShape latentShape;
    status = TextureSetMetadata::DeserializeTextureSetDesc(document, desc, latentShape);
    if (status != Status::Ok)
        return status;

    if (desc != m_desc)
    {
        SetErrorMessage("Incompatible texture set in the file - dimensions do not match.");
        return Status::FileIncompatible;
    }

    return LoadFromStreamPostHeader(document, binaryChunkOffset, binaryChunkSize, stream, latentShape);
}

Status TextureSet::SaveToMemory(void *pData, size_t* pSize, LosslessCompressionStats* pOutCompressionStats)
{
    if (!pSize)
    {
        SetErrorMessage("pSize is NULL");
        return Status::InvalidArgument;
    }

    MemoryStreamWrapper stream(m_context);

    Status status = m_context->OpenMemory(pData, *pSize, stream.ptr());
    if (status != Status::Ok)
        return status;

    status = SaveToStream(stream, pOutCompressionStats);
    if (status != Status::Ok)
        return status;

    *pSize = size_t(stream->Tell());

    return Status::Ok;
}

Status TextureSet::LoadFromMemory(void const *pData, size_t size)
{
    MemoryStreamWrapper stream(m_context);

    Status status = m_context->OpenReadOnlyMemory(pData, size, stream.ptr());
    if (status != Status::Ok)
        return status;
        
    return LoadFromStream(stream);
}

Status TextureSet::SaveToFile(char const *fileName, LosslessCompressionStats* pOutCompressionStats)
{
    FileStreamWrapper stream(m_context);

    Status status = m_context->OpenFile(fileName, true, stream.ptr());
    if (status != Status::Ok)
        return status;
        
    return SaveToStream(stream, pOutCompressionStats);
}

Status TextureSet::LoadFromFile(char const *fileName)
{
    FileStreamWrapper stream(m_context);

    Status status = m_context->OpenFile(fileName, false, stream.ptr());
    if (status != Status::Ok)
        return status;
        
    return LoadFromStream(stream);
}

Status TextureSet::ConfigureLosslessCompression(LosslessCompressionSettings const& params)
{
    switch(params.algorithm)
    {
    case CompressionType::None:
        break;

    case CompressionType::GDeflate:
        if (params.compressionLevel < 0 || params.compressionLevel > 12)
        {
            SetErrorMessage("For GDeflate, compressionLevel (%d) must be between 0 and 12.");
            return Status::InvalidArgument;
        }
        break;

    default:
        SetErrorMessage("Unrecognized algorithm value %d.", int(params.algorithm));
        return Status::InvalidArgument;
    }

    if (params.compressionRatioThreshold <= 0.f || params.compressionRatioThreshold > 1.f)
    {
        SetErrorMessage("compressionRatioThreshold (%.2f) must be between 0 and 1.");
        return Status::InvalidArgument;
    }

    m_losslessCompression = params;

    ClearErrorMessage();
    return Status::Ok;
}

Status TextureSet::WriteChannels(WriteChannelsParameters const& params)
{
    if (!params.pData)
    {
        SetErrorMessage("pData is NULL.");
        return Status::InvalidArgument;
    }

    const size_t sizeToCopy = size_t(params.height) * params.rowPitch;

    Status status = ValidateReadWriteChannelsArgs(params.mipLevel, params.firstChannel, params.numChannels,
        params.width, params.height, params.pixelStride, params.rowPitch, sizeToCopy, params.channelFormat);
    if (status != Status::Ok)
        return status;
    
    // Make sure that the user doesn't accidentally overwrite some texture data while training is in progress.
    // That wouldn't affect the training process, which may be unexpected.
    if (m_networkState != TextureSetNetworkState::Empty &&
        m_networkState != TextureSetNetworkState::Complete)
    {
        SetErrorMessage("Invalid network state (%s) for WriteChannels, must be Empty or Complete.",
            NetworkStateToString(m_networkState));
        return Status::InvalidState;
    }
    
    CudaDeviceGuard cudaGuard(m_context);
    if (!cudaGuard.Success())
        return Status::CudaError;

    if (params.addressSpace == AddressSpace::Host)
    {
        cudaError_t err = cudaMemcpy(m_textureStaging.DevicePtr(), params.pData, sizeToCopy, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaMemcpy", err);
            return Status::CudaError;
        }
    }
    
    PitchLinearImageSlice src{};
    src.pData = (params.addressSpace == AddressSpace::Device)
        ? const_cast<uint8_t*>(params.pData)
        : m_textureStaging.DevicePtr();
    src.width = params.width;
    src.height = params.height;
    src.pixelStride = int(params.pixelStride);
    src.rowPitch = int(params.rowPitch);
    src.channels = int(params.numChannels);
    src.firstChannel = 0;
    src.logChannelGroupSize = PitchLinearImageSlice::AllChannelsTogether;
    src.channelGroupStride = 0;
    src.format = params.channelFormat;

    PitchLinearImageSlice dst = GetTextureDataSlice(TextureDataPage::Reference, params.mipLevel,
        params.firstChannel, params.numChannels);

    for (int channel = 0; channel < params.numChannels; ++channel)
    {
        if (params.srcColorSpaces)
            src.channelColorSpaces[channel] = params.srcColorSpaces[channel];
        else
            src.channelColorSpaces[channel] = ColorSpace::Linear;
            
        if (params.dstColorSpaces)
            dst.channelColorSpaces[channel] = params.dstColorSpaces[channel];
        else
            dst.channelColorSpaces[channel] = ColorSpace::Linear;

        // Update the stored color spaces.
        // TODO: If the client uses different values of compressColorSpace for the same channel on different mips,
        //       that will lead to inconsistent behavior / data corruption, which would be nice to prevent.
        m_channelColorSpaces[channel + params.firstChannel] = dst.channelColorSpaces[channel];
    }

    cuda::CopyImage(src, dst, false, params.verticalFlip);

    ClearErrorMessage();
    return Status::Ok;
}

Status TextureSet::ReadChannels(ReadChannelsParameters const& params)
{
    if (!params.pOutData)
    {
        SetErrorMessage("pOutData is NULL.");
        return Status::InvalidArgument;
    }
    
    const size_t sizeToCopy = size_t(params.height) * params.rowPitch;

    Status status = ValidateReadWriteChannelsArgs(params.mipLevel, params.firstChannel, params.numChannels, params.width, params.height,
        params.pixelStride, params.rowPitch, sizeToCopy, params.channelFormat);
    if (status != Status::Ok)
        return status;
    
    CudaDeviceGuard cudaGuard(m_context);
    if (!cudaGuard.Success())
        return Status::CudaError;

    cudaError_t err;

    // If the copy kernel won't overwrite the entire rows, fill the staging area with zeros 
    // to avoid copying garbage into the client memory.
    if (params.rowPitch > params.pixelStride * uint32_t(params.width) && params.addressSpace == AddressSpace::Host)
    {
        err = cudaMemset(m_textureStaging.DevicePtr(), 0, sizeToCopy);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaMemset", err);
            return Status::CudaError;
        }
    }
    
    PitchLinearImageSlice src = GetTextureDataSlice(params.page, params.mipLevel,
        params.firstChannel, params.numChannels);

    PitchLinearImageSlice dst{};
    dst.pData = (params.addressSpace == AddressSpace::Device) ? params.pOutData : m_textureStaging.DevicePtr();
    dst.width = params.width;
    dst.height = params.height;
    dst.pixelStride = int(params.pixelStride);
    dst.rowPitch = int(params.rowPitch);
    dst.channels = int(params.pixelStride / GetBytesPerPixelComponent(params.channelFormat));
    dst.firstChannel = 0;
    dst.logChannelGroupSize = PitchLinearImageSlice::AllChannelsTogether;
    dst.channelGroupStride = 0;
    dst.format = params.channelFormat;

    for (int channel = 0; channel < params.numChannels; ++channel)
    {
        if (params.dstColorSpaces)
            dst.channelColorSpaces[channel] = params.dstColorSpaces[channel];
        else
            dst.channelColorSpaces[channel] = ColorSpace::Linear;
    }

    cuda::CopyImage(src, dst, params.useDithering, /* verticalFlip = */ false);

    if (params.addressSpace == AddressSpace::Host)
    {
        err = cudaMemcpy(params.pOutData, m_textureStaging.DevicePtr(), sizeToCopy, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaMemcpy", err);
            return Status::CudaError;
        }
    }

    ClearErrorMessage();
    return Status::Ok;
}

Status TextureSet::WriteChannelsFromTexture(WriteChannelsFromTextureParameters const& params)
{
    if (!params.texture)
    {
        SetErrorMessage("texture is NULL.");
        return Status::InvalidArgument;
    }

    SharedTexture* texture = static_cast<SharedTexture*>(params.texture);

    cudaSurfaceObject_t surface = texture->GetSurfaceObject(params.textureMipLevel);

    if (!surface)
    {
        SetErrorMessage("surface is NULL.");
        return Status::InvalidArgument;
    }

    Status status = ValidateReadWriteChannelsArgs(params.mipLevel, params.firstChannel, params.numChannels, /* width = */ 1,
        /* height = */ 1, /* pixelStride = */ 1, /* rowPitch = */ 1, /* sizeToCopy = */ 0, texture->GetDesc().format);
    if (status != Status::Ok)
        return status;
    
    CudaDeviceGuard cudaGuard(m_context);
    if (!cudaGuard.Success())
        return Status::CudaError;

    const SharedTextureDesc& textureDesc = texture->GetDesc();
    const int textureMipWidth = std::max(textureDesc.width >> params.textureMipLevel, 1);
    const int textureMipHeight = std::max(textureDesc.height >> params.textureMipLevel, 1);
    
    // Make sure that the user doesn't accidentally overwrite some texture data while training is in progress.
    // That wouldn't affect the training process, which may be unexpected.
    if (m_networkState != TextureSetNetworkState::Empty &&
        m_networkState != TextureSetNetworkState::Complete)
    {
        SetErrorMessage("Invalid network state (%s) for WriteChannelsFromTexture, "
            "must be Empty or Complete.", NetworkStateToString(m_networkState));
        return Status::InvalidState;
    }

    SurfaceInfo src{};
    src.surface = surface;
    src.width = textureMipWidth;
    src.height = textureMipHeight;
    src.pixelStride = texture->GetPixelStride();
    src.channels = textureDesc.channels;
    src.format = textureDesc.format;
    src.rgbColorSpace = params.srcRgbColorSpace;
    src.alphaColorSpace = params.srcAlphaColorSpace;

    PitchLinearImageSlice dst = GetTextureDataSlice(TextureDataPage::Reference, params.mipLevel,
        params.firstChannel, params.numChannels);
    dst.channelColorSpaces[0] = dst.channelColorSpaces[1] = dst.channelColorSpaces[2] = params.dstRgbColorSpace;
    dst.channelColorSpaces[3] = params.dstAlphaColorSpace;

    cuda::CopySurfaceToImage(src, dst, params.verticalFlip);
    
    // Update the compressed color bits for these channels.
    // TODO: If the client uses different values of compressColorSpace for the same channel on different mips,
    //       that will lead to inconsistent behavior / data corruption, which would be nice to prevent.
    for (int channel = 0; channel < params.numChannels; ++channel)
        m_channelColorSpaces[params.firstChannel + channel] = dst.channelColorSpaces[channel];

    ClearErrorMessage();
    return Status::Ok;
}

Status TextureSet::ReadChannelsIntoTexture(ReadChannelsIntoTextureParameters const& params)
{
    if (!params.texture)
    {
        SetErrorMessage("texture is NULL.");
        return Status::InvalidArgument;
    }

    SharedTexture* texture = static_cast<SharedTexture*>(params.texture);

    cudaSurfaceObject_t surface = texture->GetSurfaceObject(params.textureMipLevel);

    if (!surface)
    {
        SetErrorMessage("surface is NULL.");
        return Status::InvalidArgument;
    }

    Status status = ValidateReadWriteChannelsArgs(params.mipLevel, params.firstChannel, params.numChannels, /* width = */ 1,
        /* height = */ 1, /* pixelStride = */ 1, /* rowPitch = */ 1, /* sizeToCopy = */ 0, texture->GetDesc().format);
    if (status != Status::Ok)
        return status;
    
    CudaDeviceGuard cudaGuard(m_context);
    if (!cudaGuard.Success())
        return Status::CudaError;

    const SharedTextureDesc& textureDesc = texture->GetDesc();
    const int textureMipWidth = std::max(textureDesc.width >> params.textureMipLevel, 1);
    const int textureMipHeight = std::max(textureDesc.height >> params.textureMipLevel, 1);
    
    PitchLinearImageSlice src = GetTextureDataSlice(params.page, params.mipLevel,
        params.firstChannel, params.numChannels);

    SurfaceInfo dst{};
    dst.surface = surface;
    dst.width = textureMipWidth;
    dst.height = textureMipHeight;
    dst.pixelStride = texture->GetPixelStride();
    dst.channels = textureDesc.channels;
    dst.format = textureDesc.format;
    dst.rgbColorSpace = params.dstRgbColorSpace;
    dst.alphaColorSpace = params.dstAlphaColorSpace;

    cuda::CopyImageToSurface(src, dst, params.useDithering, /* verticalFlip = */ false);

    // Wait until the copy is done on the GPU side.
    // TODO: use a GPU sync primitive (fence) to exit early and synchronize with the client properly.
    cudaError_t err = cudaEventRecord(m_eventStop);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaEventRecord", err);
        return Status::CudaError;
    }

    err = cudaEventSynchronize(m_eventStop);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaEventSynchronize", err);
        return Status::CudaError;
    }

    ClearErrorMessage();
    return Status::Ok;
}

Status TextureSet::GenerateMips()
{
    if (m_desc.mips <= 1)
    {
        ClearErrorMessage();
        return Status::Ok;
    }
    
    CudaDeviceGuard cudaGuard(m_context);
    if (!cudaGuard.Success())
        return Status::CudaError;

    for (int mip = 1; mip < m_desc.mips; ++mip)
    {
        PitchLinearImageSlice src = GetTextureDataSlice(TextureDataPage::Reference, mip - 1, 0, m_desc.channels);
        PitchLinearImageSlice dst = GetTextureDataSlice(TextureDataPage::Reference, mip, 0, m_desc.channels);

        cuda::ResizeMultichannelImage(src, dst, m_channelColorSpaces);
    }

    ClearErrorMessage();
    return Status::Ok;
}

Status TextureSet::BeginCompression(const CompressionSettings& settings)
{
    if (m_latentShape.IsEmpty())
    {
        SetErrorMessage("Latent shape must not be empty.");
        return Status::InvalidState;
    }

    if (settings.trainingSteps <= 0 || settings.stepsPerIteration <= 0)
    {
        SetErrorMessage("CompressionSettings.trainingSteps (%d) and "
            "CompressionSettings.stepsPerIteration (%d) must be positive.",
            settings.trainingSteps, settings.stepsPerIteration);
        return Status::OutOfRange;
    }
    
    if (settings.kPixelsPerBatch <= 0 || settings.kPixelsPerBatch > NTC_MAX_KPIXELS_PER_BATCH)
    {
        SetErrorMessage("CompressionSettings.kPixelsPerBatch (%d) must be "
            "between 1 and NTC_MAX_KPIXELS_PER_BATCH (%d).", settings.kPixelsPerBatch, NTC_MAX_KPIXELS_PER_BATCH);
        return Status::OutOfRange;
    }
    
    if (m_networkState != TextureSetNetworkState::Empty &&
        m_networkState != TextureSetNetworkState::Complete)
    {
        SetErrorMessage("Invalid network state for BeginCompression (%s), must be Empty or Complete.",
            NetworkStateToString(m_networkState));
        return Status::InvalidState;
    }

    CudaDeviceGuard cudaGuard(m_context);
    if (!cudaGuard.Success())
        return Status::CudaError;

    std::array<ChannelInfo, NTC_MAX_CHANNELS> channelInfos;
    Status status = ComputeChannelNormalizationParameters(channelInfos);
    if (status != Status::Ok)
        return status;
    
    for (int channel = 0; channel < NTC_MAX_CHANNELS; ++channel)
    {
        channelInfos[channel].lossFunctionScale = std::max(0.f, settings.lossFunctionScales[channel]);
    }

    CudaRandomGen cudaRng;
    
    // Set the random seed if specified
    if (settings.randomSeed)
        cudaRng.SetSeed(settings.randomSeed);
    else
        cudaRng.RandomizeSeed();

    // Transfer the seed to the mt19937 RNG used on the host side
    m_rng = std::mt19937(cudaRng.GetSeed());

    m_featureGrid.Fill(cudaRng);
    
    // Zero initialize various buffers.
    // Use a loop to reuse the error handling code.
    std::tuple<void*, size_t> buffers[] = {
        { m_mlpMoment1.DevicePtr(), m_mlpMoment1.Size() },
        { m_mlpMoment2.DevicePtr(), m_mlpMoment2.Size() },
        { m_weightGradients.DevicePtr(), m_weightGradients.Size() },
        { m_loss.DevicePtr(), m_loss.Size() },
        { m_mlpWeightsBase.DevicePtr(), m_mlpWeightsBase.Size() },
        { m_mlpWeightsQuantized.DevicePtr(), m_mlpWeightsQuantized.Size() }
    };
    
    for (auto [ptr, size] : buffers)
    {
        if (!ptr)
            continue;

        cudaError_t err = cudaMemset(ptr, 0, size);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaMemset", err);
            return Status::CudaError;
        }
    }

    // Fill the layers' data with normal distributed random numbers
    int weightOffset = 0;

    for (int i = 0; i < NTC_MLP_LAYERS; i++)
    {
        int const inputs = MlpDesc::GetLayerInputChannels(i);
        int const outputs = MlpDesc::GetLayerOutputChannels(i);
        int const layerWeights = inputs * outputs;

        float scale = sqrtf(2.f / float(inputs));
        cudaRng.FillRandomNormalHalf(m_mlpWeightsBase.DevicePtr() + weightOffset,
            layerWeights, scale, 0.f, -1000.f, 1000.f);
        
        weightOffset += layerWeights;
    }

    // Copy the random weights to the quantized buffer to be used on the first training step
    cudaError_t err = cudaMemcpy(m_mlpWeightsQuantized.DevicePtr(), m_mlpWeightsBase.DevicePtr(),
        m_mlpWeightsBase.Size(), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaMemcpy", err);
        return Status::CudaError;
    }

    // Fill out the mip information array before training
    MipInfo mipInfos[NTC_MAX_MIPS]{};
    float mipPdf[NTC_MAX_MIPS]{};
    float pdfSum = 0.f;
    int mipWidth = m_desc.width;
    int mipHeight = m_desc.height;
    for (int mip = 0; mip < m_desc.mips; ++mip)
    {
        ColorMipDesc const& colorMip = m_colorMips[mip];
        LatentImageDesc const& latentImage = m_latentImages[colorMip.neuralLod];

        // Texture and latent data offsets
        mipInfos[mip].referenceTextureOffset = m_textureMipOffsets[mip];
        mipInfos[mip].highResLatentOffset = m_featureGrid.GetLatentOffset(colorMip.neuralLod);
        mipInfos[mip].lowResLatentOffset = m_featureGrid.GetLatentOffset(colorMip.neuralLod + 1);
        mipInfos[mip].highResMaskOffset = m_featureGrid.GetMaskOffset(colorMip.neuralLod);
        mipInfos[mip].lowResMaskOffset = m_featureGrid.GetMaskOffset(colorMip.neuralLod + 1);

        mipInfos[mip].neuralLod = colorMip.neuralLod;
        mipInfos[mip].positionLod = colorMip.positionLod;
        mipInfos[mip].positionScale = colorMip.positionScale;

        mipInfos[mip].highResLatentWidth = latentImage.width;
        mipInfos[mip].highResLatentHeight = latentImage.height;
        mipInfos[mip].lowResLatentWidth = std::max(latentImage.width >> 1, 1);
        mipInfos[mip].lowResLatentHeight = std::max(latentImage.height >> 1, 1);

        // Calculate the PDF for sampling this particular mip level based on its area,
        // clamp at the lower end to make sure the coarsest mips are sampled at all
        mipPdf[mip] = float(std::max(size_t(mipWidth) * size_t(mipHeight), size_t(512)));
        pdfSum += mipPdf[mip];

        // Advance to the next mip level
        mipWidth = std::max(mipWidth >> 1, 1);
        mipHeight = std::max(mipHeight >> 1, 1);
    }

    // Normalize the PDF and accumulate it into the CDF for use in the shader
    for (int mip = 0; mip < m_desc.mips; ++mip)
    {
        mipInfos[mip].cdf = (mipPdf[mip] / pdfSum) + ((mip > 0) ? mipInfos[mip - 1].cdf : 0.f);
    }

    // Copy the mip infos to the device
    cuda::SetMipInfos(mipInfos, m_desc.mips);

    // Copy the channel normalization infos to the device
    cuda::SetChannelInfos(channelInfos.data(), m_desc.channels);
    
    m_currentStep = 0;
    m_compressionSettings = settings;

    m_lossScale = 256.f;

    m_networkState = TextureSetNetworkState::Initialized;
    
    // Invalidate the weight vectors
    m_rowMajorWeightDataInt8.clear();
    m_rowMajorWeightDataFP8.clear();

    ClearErrorMessage();
    return Status::Ok;
}

static float cosine_schedule(int step, float lr_min, float lr_max, int train_steps)
{
    return lr_min + 0.5f * (lr_max - lr_min) * (1 + cos(step * float(M_PI) / train_steps));
}

Status TextureSet::RunCompressionSteps(CompressionStats* pOutStats)
{
    if (m_networkState != TextureSetNetworkState::Initialized &&
        m_networkState != TextureSetNetworkState::TrainingInProgress)
    {
        SetErrorMessage("Invalid network state for RunCompressionSteps (%s), "
            "must be Initialized or TrainingInProgress.", NetworkStateToString(m_networkState));
        return Status::InvalidState;
    }

    CudaDeviceGuard cudaGuard(m_context);
    if (!cudaGuard.Success())
        return Status::CudaError;

    cudaError_t err = cudaEventRecord(m_eventStart);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaEventRecord (1)", err);
        return Status::CudaError;
    }

    const int finalStepCount = std::min(m_currentStep + m_compressionSettings.stepsPerIteration,
        m_compressionSettings.trainingSteps);
    const int stepsInThisIteration = finalStepCount - m_currentStep;

    std::uniform_int_distribution<uint32_t> intDistribution;

    const float minLearningRate = 0.f;
    float networkLearningRate = 0.f;
    float gridLearningRate = 0.f;
    const size_t pixelsPerBatch = std::min(size_t(m_desc.width) * size_t(m_desc.height),
        size_t(m_compressionSettings.kPixelsPerBatch) * c_PixelsPerKPixel);
    bool const stableTraining = m_compressionSettings.stableTraining;

    uint32_t validMask = GetValidChannelMask();
    
    for (; m_currentStep < finalStepCount; ++m_currentStep)
    {
        networkLearningRate = cosine_schedule(m_currentStep, minLearningRate,
            m_compressionSettings.networkLearningRate, m_compressionSettings.trainingSteps);

        gridLearningRate = cosine_schedule(m_currentStep, minLearningRate,
            m_compressionSettings.gridLearningRate, m_compressionSettings.trainingSteps);
        
        // Quantize and stop updating the latents after this percentage of steps
        const float freezeRatio = 0.95f;
        // Quantize the MLP to FP8 and start refining it after this percentage of steps
        const float quantizeFP8Ratio = 0.96f;
        // Quantize the MLP to Int8 and start refining it after this percentage of steps
        const float quantizeInt8Ratio = 0.98f;

        const int freezeLatentsStep = int(float(m_compressionSettings.trainingSteps) * freezeRatio);
        const int quantizeFP8Step = int(float(m_compressionSettings.trainingSteps) * quantizeFP8Ratio);
        const int quantizeInt8Step = int(float(m_compressionSettings.trainingSteps) * quantizeInt8Ratio);
        bool quantizeWeightsFP8 = m_currentStep > quantizeFP8Step && m_currentStep <= quantizeInt8Step;
        bool quantizeWeightsInt8 = m_currentStep > quantizeInt8Step;

        // Start updating the Int8 weights with a higher learning rate
        if (quantizeWeightsInt8)
        {
            gridLearningRate = cosine_schedule(m_currentStep - quantizeInt8Step, minLearningRate,
                m_compressionSettings.gridLearningRate, m_compressionSettings.trainingSteps - quantizeInt8Step);
        }

        int const networkWeightOffset = quantizeWeightsInt8 ? m_numNetworkParams : 0;

        // Before starting the quantized FP8 MLP refining, save the pre-quantization weights to the second part
        // of the buffer, to be used later for Int8 refining.
        if (m_currentStep == quantizeFP8Step)
        {
            err = cudaMemcpy(m_mlpWeightsBase.DevicePtr() + m_numNetworkParams,
                m_mlpWeightsBase.DevicePtr(), m_numNetworkParams * sizeof(half), cudaMemcpyDeviceToDevice);

            if (err == cudaSuccess)
            {
                err = cudaMemcpy(m_mlpWeightsQuantized.DevicePtr() + m_numNetworkParams,
                    m_mlpWeightsQuantized.DevicePtr(), m_numNetworkParams * sizeof(half), cudaMemcpyDeviceToDevice);
            }

            if (err != cudaSuccess)
            {
                SetCudaErrorMessage("cudaMemcpy (mlpWeights)", err);
                return Status::CudaError;
            }
        }

        if (quantizeWeightsInt8 || quantizeWeightsFP8)
        {
            WeightLayout const* weightLayout = m_context->GetWeightLayout(
                quantizeWeightsFP8 ? InferenceWeightType::GenericFP8 : InferenceWeightType::GenericInt8);
            assert(weightLayout);

            cuda::QuantizeNetwork(
                m_context->GetFP16WeightLayout(),
                *weightLayout,
                m_mlpWeightsQuantized.DevicePtr() + networkWeightOffset,
                /* outputData = */ nullptr, // We don't need the quantized weights and scales mid-training,
                                            // only for the final output
                /* useFP8 = */ quantizeWeightsFP8
            );
        }

        // Clear the gradient mask memory - it will be atomically updated in the Regression function
        m_featureGrid.ClearGradientMask();

        // Only calculate the loss if this is the last step in a batch, i.e. if we're going to read it later
        bool calculateLoss = (m_currentStep == finalStepCount - 1);

        RegressionKernelParams params{};
        params.referenceWidth = m_desc.width;
        params.referenceHeight = m_desc.height;
        params.numChannels = m_desc.channels;
        params.numMips = m_desc.mips;
        params.numNeuralMips = m_featureGrid.GetNumMipLevels();
        params.numFeatures = m_latentShape.numFeatures;
        params.latentStride = m_featureGrid.GetLatentStride();
        params.maskChannelIndex = m_maskChannelIndex;
        params.discardMaskedOutPixels = m_discardMaskedOutPixels;
        params.useFP8Quantization = !quantizeWeightsInt8;
        params.validChannelMask = validMask;
        params.randomSeed = intDistribution(m_rng);
        params.lossScale = m_lossScale;
        params.experimentalKnob = m_experimentalKnob;
        params.referenceImage = m_textureData.DevicePtr();
        params.latents = m_featureGrid.GetQuantizedLatentsDevicePtr(0);
        params.networkWeights = m_mlpWeightsQuantized.DevicePtr() + networkWeightOffset;
        params.latentGradients = m_featureGrid.GetGradientsDevicePtr();
        params.networkGradients = m_weightGradients.DevicePtr();
        params.loss = calculateLoss ? m_loss.DevicePtr() : nullptr;
        params.gradientMask = m_featureGrid.GetGradientMaskDevicePtr();

        // Forward + backprop
        cuda::Regression(pixelsPerBatch, stableTraining, params);

        if (stableTraining)
        {
            size_t const sliceSize = TILE_SIZE_X * TB_SIZE_Y;
            size_t gradientSlices = (pixelsPerBatch + sliceSize - 1) / sliceSize;
            int numNetworkParams = MlpDesc::GetTotalWeightCount() + MlpDesc::GetTotalOutputCount();

            cuda::ReduceNetworkGrad(
                numNetworkParams,
                gradientSlices,
                /* useFloatGradients = */ stableTraining,
                m_weightGradients.DevicePtr());
        }

        // NW optimizer
        cuda::OptimizeNetwork(
            m_numNetworkParams,
            /* useFloatGradients = */ stableTraining,
            m_mlpWeightsBase.DevicePtr() + networkWeightOffset,
            m_mlpWeightsQuantized.DevicePtr() + networkWeightOffset,
            m_weightGradients.DevicePtr(),
            m_mlpMoment1.DevicePtr(),
            m_mlpMoment2.DevicePtr(),
            m_lossScale,
            float(m_currentStep + 1),
            intDistribution(m_rng),
            networkLearningRate);

        // Latent optimizer
        if (m_currentStep == freezeLatentsStep)
        {
            cuda::FreezeQuantization(
                m_featureGrid.GetTotalPixelCount(),
                m_latentShape.numFeatures,
                m_featureGrid.GetBaseLatentsDevicePtr(0),
                m_featureGrid.GetQuantizedLatentsDevicePtr(0)
            );
        }
        else if (m_currentStep < freezeLatentsStep)
        {
            cuda::OptimizeLatentGrid(
                m_featureGrid.GetTotalPixelCount(),
                m_latentShape.numFeatures,
                m_featureGrid.GetLatentStride(),
                /* useFloatGradients = */ stableTraining,
                m_featureGrid.GetBaseLatentsDevicePtr(0),
                m_featureGrid.GetQuantizedLatentsDevicePtr(0),
                m_featureGrid.GetGradientsDevicePtr(),
                m_featureGrid.GetMoment1DevicePtr(),
                m_featureGrid.GetMoment2DevicePtr(),
                m_featureGrid.GetGradientMaskDevicePtr(),
                m_lossScale,
                float(m_currentStep),
                intDistribution(m_rng),
                gridLearningRate);
        }
    }

    err = cudaEventRecord(m_eventStop);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaEventRecord (2)", err);
        return Status::CudaError;
    }

    err = cudaEventSynchronize(m_eventStop);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaEventSynchronize", err);
        return Status::CudaError;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_eventStart, m_eventStop);
    
    if (pOutStats)
    {
        memset(pOutStats, 0, sizeof(CompressionStats));
        pOutStats->currentStep = m_currentStep;
        pOutStats->learningRate = networkLearningRate;
        pOutStats->millisecondsPerStep = milliseconds / float(m_compressionSettings.stepsPerIteration);
    }

    int validChannels = 0;
    for (int channel = 0; channel < m_desc.channels; ++channel)
    {
        if (validMask & (1 << channel))
            ++validChannels;
    }
    
    {
        // Loss reduction
        float loss_red;
        cudaError_t err = cuda::ReduceLoss((pixelsPerBatch + LOCAL_PIXELS - 1) / LOCAL_PIXELS,
            m_loss.DevicePtr(), m_lossReduction, loss_red);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("ReduceLoss", err);
            return Status::CudaError;
        }

        loss_red /= float(validChannels);

        m_lossScale = std::min(32768.f, 128.f / sqrtf(loss_red));
        
        if (pOutStats)
        {
            pOutStats->loss = loss_red;
            pOutStats->lossScale = m_lossScale;
        }
    }

    if (m_currentStep < m_compressionSettings.trainingSteps)
    {
        m_networkState = TextureSetNetworkState::TrainingInProgress;
        return Status::Incomplete;
    }

    m_networkState = TextureSetNetworkState::TrainingFinished;
    ClearErrorMessage();
    return Status::Ok;
}

Status TextureSet::FinalizeCompression()
{
    if (m_networkState != TextureSetNetworkState::TrainingFinished)
    {
        SetErrorMessage("Invalid network state for FinalizeCompression (%s), must be TrainingFinished.",
            NetworkStateToString(m_networkState));
        return Status::InvalidState;
    }

    CudaDeviceGuard cudaGuard(m_context);
    if (!cudaGuard.Success())
        return Status::CudaError;

    // Encode the HR and LR grids
    for (int i = 0; i < m_featureGrid.GetNumMipLevels(); ++i)
    {
        assert(i < m_latentImages.size());
        LatentImageDesc const& latentImage = m_latentImages[i];

        cuda::PackLatents(
            latentImage.width,
            latentImage.height,
            m_featureGrid.GetNumLayers(),
            m_featureGrid.GetLatentStride(),
            m_featureGrid.GetQuantizedLatentsDevicePtr(i),
            m_featureGrid.GetEncodedPixelsDevicePtr(i));
    }

    // Download the encoded latents from device
    cudaError_t err = m_featureGrid.GetEncodedPixelsArray().CopyToHost();
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaMemcpy (QuantizedLatents)", err);
        return Status::CudaError;
    }

    // Quantize and compute FP8 weights, scale and bias values
    WeightLayout const* weightLayoutFP8 = m_context->GetWeightLayout(InferenceWeightType::GenericFP8);
    assert(weightLayoutFP8);
    cuda::QuantizeNetwork(
        m_context->GetFP16WeightLayout(),
        *weightLayoutFP8,
        m_mlpWeightsQuantized.DevicePtr(),
        (int8_t*)m_mlpDataFP8.DevicePtr(),
        /* useFP8 = */ true
    );

    // Quantize and compute Int8 weights, scale and bias values
    WeightLayout const* weightLayoutInt8 = m_context->GetWeightLayout(InferenceWeightType::GenericInt8);
    assert(weightLayoutInt8);
    cuda::QuantizeNetwork(
        m_context->GetFP16WeightLayout(),
        *weightLayoutInt8,
        m_mlpWeightsQuantized.DevicePtr() + m_numNetworkParams,
        (int8_t*)m_mlpDataInt8.DevicePtr(),
        /* useFP8 = */ false
    );

    // Download the MLP data from device
    err = m_mlpDataInt8.CopyToHost();
    if (err == cudaSuccess)
        err = m_mlpDataFP8.CopyToHost();
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage("cudaMemcpy (MlpData*)", err);
        return Status::CudaError;
    }

    // Copy the FP8 data into the m_rowMajorWeightDataFP8 vector for possible GAPI inference and support queries
    m_rowMajorWeightDataFP8.resize(m_mlpDataFP8.Size());
    memcpy(m_rowMajorWeightDataFP8.data(), m_mlpDataFP8.HostPtr(), m_mlpDataFP8.Size());

    // Copy the Int8 data into the m_rowMajorWeightDataInt8 vector, same reason
    m_rowMajorWeightDataInt8.resize(m_mlpDataInt8.Size());
    memcpy(m_rowMajorWeightDataInt8.data(), m_mlpDataInt8.HostPtr(), m_mlpDataInt8.Size());

    m_networkState = TextureSetNetworkState::Complete;

    ClearErrorMessage();
    return Status::Ok;
}

void TextureSet::AbortCompression()
{
    m_networkState = TextureSetNetworkState::Empty;
}

Status TextureSet::Decompress(DecompressionStats* pOutStats, bool useInt8Weights)
{
    if (m_networkState != TextureSetNetworkState::TrainingInProgress &&
        m_networkState != TextureSetNetworkState::TrainingFinished &&
        m_networkState != TextureSetNetworkState::Complete)
    {
        SetErrorMessage("Invalid network state for Decompress (%s), must be TrainingInProgress, TrainingFinished or Complete.",
            NetworkStateToString(m_networkState));
        return Status::InvalidState;
    }

    if (useInt8Weights)
    {
        if (m_networkState == TextureSetNetworkState::TrainingInProgress)
        {
            SetErrorMessage("In-progress decompression with Int8 weights is not supported.");
            return Status::InvalidState;
        }
    }

    if (pOutStats)
        memset(pOutStats, 0, sizeof(DecompressionStats));

    CudaDeviceGuard cudaGuard(m_context);
    if (!cudaGuard.Success())
        return Status::CudaError;
        
    if (m_networkState != TextureSetNetworkState::TrainingInProgress &&
        m_networkState != TextureSetNetworkState::TrainingFinished)
    {
        // Upload the encoded latents to device
        cudaError_t err = m_featureGrid.GetEncodedPixelsArray().CopyToDevice();
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaMemcpy (QuantizedLatents)", err);
            return Status::CudaError;
        }

        // Decode the HR and LR grids
        for (int i = 0; i < m_featureGrid.GetNumMipLevels(); ++i)
        {
            LatentImageDesc const& latentImage = m_latentImages[i];

            cuda::UnpackLatents(
                latentImage.width,
                latentImage.height,
                m_featureGrid.GetNumLayers(),
                m_featureGrid.GetLatentStride(),
                m_featureGrid.GetEncodedPixelsDevicePtr(i),
                m_featureGrid.GetQuantizedLatentsDevicePtr(i));
        }

        // Upload the MLP weights to device
        err = m_mlpDataInt8.CopyToDevice();
        if (err == cudaSuccess)
            err = m_mlpDataFP8.CopyToDevice();
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaMemcpy (MlpData*)", err);
            return Status::CudaError;
        }

        // Copy dummy channel infos to the device: the per-channel scale and bias are already baked into the MLP
        std::array<ChannelInfo, NTC_MAX_CHANNELS> channelInfos;
        channelInfos.fill(ChannelInfo());
        cuda::SetChannelInfos(channelInfos.data(), m_desc.channels);

        // Clear the FP16 weight buffer, mostly for debugging - to avoid using stale values
        cudaMemset(m_mlpWeightsQuantized.DevicePtr(), 0, m_mlpWeightsQuantized.Size());

        // Convert the FP8 weights to FP16 in the MMA layout for CUDA decompression to work
        WeightLayout const* weightLayoutFP8 = m_context->GetWeightLayout(InferenceWeightType::GenericFP8);
        assert(weightLayoutFP8);
        cuda::ConvertNetworkFromQuantizedToFp16(
            m_context->GetFP16WeightLayout(),
            *weightLayoutFP8,
            m_mlpWeightsQuantized.DevicePtr(),
            (int8_t*)m_mlpDataFP8.DevicePtr(),
            /* useFP8 = */ true
        );

        // Convert the Int8 weights to FP16 into the second page of the FP16 MLP data,
        // in case we want to decompress using those for validation
        WeightLayout const* weightLayoutInt8 = m_context->GetWeightLayout(InferenceWeightType::GenericInt8);
        assert(weightLayoutInt8);
        cuda::ConvertNetworkFromQuantizedToFp16(
            m_context->GetFP16WeightLayout(),
            *weightLayoutInt8,
            m_mlpWeightsQuantized.DevicePtr() + m_numNetworkParams,
            (int8_t*)m_mlpDataInt8.DevicePtr(),
            /* useFP8 = */ false
        );

    }
    
    uint32_t validMask = GetValidChannelMask();
    int validChannels = 0;
    for (int channel = 0; channel < m_desc.channels; ++channel)
    {
        if (validMask & (1 << channel))
            ++validChannels;
    }

    std::array<float, NTC_MAX_CHANNELS> overallPerChannelLoss;
    overallPerChannelLoss.fill(0.f);
    float overallLoss = 0.f;
    int overallPixels = 0;
    
    if (pOutStats)
    {
        cudaError_t err = cudaEventRecord(m_eventStart);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaEventRecord (1)", err);
            return Status::CudaError;
        }
    }
    
    for (int mipLevel = 0; mipLevel < m_desc.mips; mipLevel++)
    {
        int neuralLod = ColorMipToNeuralLod(mipLevel);
        LatentImageDesc const& latentImage = m_latentImages[neuralLod];

        int colorMipWidth = std::max(m_desc.width >> mipLevel, 1);
        int colorMipHeight = std::max(m_desc.height >> mipLevel, 1);
        size_t const lossItemsPerChannel = GetDecompressionLossItemsPerChannel(colorMipWidth, colorMipHeight);
        size_t const lossLength = lossItemsPerChannel * NTC_MAX_CHANNELS;
        
        ColorMipDesc const& colorMip = m_colorMips[mipLevel];

        // Clear the loss buffer
        cudaError_t err = cudaMemset(m_loss.DevicePtr(), 0, lossLength * sizeof(float));
        if (err != cudaSuccess)
            return Status::CudaError;

        // If we have separate ref and out pages, use the out page for decompression
        half* const textureDataOut = m_textureDataOut.DevicePtr()
            ? m_textureDataOut.DevicePtr()
            : m_textureData.DevicePtr();

        uint64_t const textureDataOffset = m_textureMipOffsets[mipLevel] * m_desc.channels;

        InferenceKernelParams params{};
        params.referenceWidth = colorMipWidth;
        params.referenceHeight = colorMipHeight;
        params.numChannels = m_desc.channels;
        params.maskChannelIndex = m_maskChannelIndex;
        params.discardMaskedOutPixels = m_discardMaskedOutPixels;
        params.useFP8Quantization = !useInt8Weights;
        params.validChannelMask = validMask;
        params.highResLatentWidth = latentImage.width;
        params.highResLatentHeight = latentImage.height;
        params.lowResLatentWidth = std::max(latentImage.width >> 1, 1);
        params.lowResLatentHeight = std::max(latentImage.height >> 1, 1);
        params.numFeatures = m_latentShape.numFeatures;
        params.latentStride = m_featureGrid.GetLatentStride();
        params.positionScale = colorMip.positionScale;
        params.positionLod = colorMip.positionLod;
        params.lossItemsPerChannel = lossItemsPerChannel;
        params.experimentalKnob = m_experimentalKnob;
        params.highResLatents = m_featureGrid.GetQuantizedLatentsDevicePtr(neuralLod);
        params.lowResLatents = m_featureGrid.GetQuantizedLatentsDevicePtr(neuralLod + 1);
        params.mlpWeights = m_mlpWeightsQuantized.DevicePtr() + (useInt8Weights ? m_numNetworkParams : 0);
        params.referenceImage = m_textureData.DevicePtr() + textureDataOffset;
        params.outputImage = textureDataOut + textureDataOffset;
        params.outputLoss = m_loss.DevicePtr();

        cuda::Inference(params);

        if (validChannels > 0)
        {
            int mipPixels = colorMipWidth * colorMipHeight;

            float meanLossForAllChannels = 0.f;
            for (int ch = 0; ch < m_desc.channels; ++ch)
            {
                if ((validMask & (1 << ch)) == 0)
                    continue;
                    
                float lossInChannel;
                cudaError_t err = cuda::ReduceLoss(lossItemsPerChannel, m_loss.DevicePtr() + lossItemsPerChannel * ch,
                    m_lossReduction, lossInChannel);
                if (err != cudaSuccess)
                {
                    SetCudaErrorMessage("ReduceLoss", err);
                    return Status::CudaError;
                }

                meanLossForAllChannels += lossInChannel;
                overallPerChannelLoss[ch] += float(mipPixels) * lossInChannel;
            }

            meanLossForAllChannels /= float(validChannels);

            if (pOutStats)
                pOutStats->perMipLoss[mipLevel] = meanLossForAllChannels;

            overallLoss += float(mipPixels) * meanLossForAllChannels;
            overallPixels += mipPixels;
        }
    }
    
    if (pOutStats)
    {
        cudaError_t err = cudaEventRecord(m_eventStop);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaEventRecord (2)", err);
            return Status::CudaError;
        }

        err = cudaEventSynchronize(m_eventStop);
        if (err != cudaSuccess)
        {
            SetCudaErrorMessage("cudaEventSynchronize", err);
            return Status::CudaError;
        }
        
        cudaEventElapsedTime(&pOutStats->gpuTimeMilliseconds, m_eventStart, m_eventStop);
    
        if (overallPixels > 0)
        {
            pOutStats->overallLoss = overallLoss / float(overallPixels);

            for (int ch = 0; ch < NTC_MAX_CHANNELS; ++ch)
            {
                pOutStats->perChannelLoss[ch] = overallPerChannelLoss[ch] / float(overallPixels);
            }
        }
    }
    
    ClearErrorMessage();
    return Status::Ok;
}

Status TextureSet::SetMaskChannelIndex(int index, bool discardMaskedOutPixels)
{
    if (index >= m_desc.channels)
        return Status::OutOfRange;

    m_maskChannelIndex = index;
    m_discardMaskedOutPixels = discardMaskedOutPixels;
    return Status::Ok;
}

void TextureSet::SetExperimentalKnob(float value)
{
    m_experimentalKnob = value;
}

Status TextureSet::ValidateReadWriteChannelsArgs(int mipLevel, int firstChannel, int numChannels,
    int width, int height, size_t pixelStride, size_t rowPitch, size_t sizeToCopy, ChannelFormat format)
{
    if (mipLevel < 0 || mipLevel >= m_desc.mips)
    {
        SetErrorMessage("mipLevel (%d) must be between 0 and %d.", mipLevel, m_desc.mips - 1);
        return Status::OutOfRange;
    }

    if (firstChannel < 0 || firstChannel >= m_desc.channels)
    {
        SetErrorMessage("firstChannel (%d) must be between 0 and %d.", firstChannel, m_desc.channels - 1);
        return Status::OutOfRange;
    }

    if (numChannels < 1 || numChannels + firstChannel > m_desc.channels)
    {
        SetErrorMessage("For the provided firstChannel (%d), numChannels (%d) must be between 1 and %d.",
            firstChannel, numChannels, m_desc.channels - firstChannel);
        return Status::OutOfRange;
    }

    if (width <= 0 || height <= 0)
    {
        SetErrorMessage("width (%d) and height (%d) must be positive.", width, height);
        return Status::OutOfRange;
    }

    if (pixelStride < 1 || rowPitch < pixelStride)
    {
        SetErrorMessage("pixelStride (%d) must be between 1 and rowPitch (%d).", pixelStride, rowPitch);
        return Status::InvalidArgument;
    }

    if (sizeToCopy > m_textureStaging.Size())
    {
        SetErrorMessage("The operation requires copying too much data (%zu bytes), must fit into "
            "the staging buffer (%zu bytes).", sizeToCopy, m_textureStaging.Size());
        return Status::OutOfRange;
    }

    return Status::Ok;
}

PitchLinearImageSlice TextureSet::GetTextureDataSlice(TextureDataPage page, int mipLevel,
    int firstChannel, int numChannels)
{
    // When output data is requested and we have separate ref and out data arrays, select the output page
    half* const textureData = (page == TextureDataPage::Output && m_textureDataOut.DevicePtr())
        ? m_textureDataOut.DevicePtr()
        : m_textureData.DevicePtr();

    // See the comment to PitchLinearImageSlice structure for the texture data layout explanation.
    PitchLinearImageSlice slice{};
    slice.pData = (uint8_t*)(textureData + m_textureMipOffsets[mipLevel] * m_desc.channels);
    slice.width = std::max(m_desc.width >> mipLevel, 1);
    slice.height = std::max(m_desc.height >> mipLevel, 1);
    slice.pixelStride = 2 * int(sizeof(half));
    slice.rowPitch = slice.width * m_desc.channels * int(sizeof(half));
    slice.channels = numChannels;
    slice.firstChannel = firstChannel;
    slice.logChannelGroupSize = 1;
    slice.channelGroupStride = int(sizeof(half)) * slice.width * 2;
    slice.format = ChannelFormat::FLOAT16;
    for (int ch = 0; ch < numChannels && firstChannel + ch < NTC_MAX_CHANNELS; ++ch)
    {
        slice.channelColorSpaces[ch] = m_channelColorSpaces[firstChannel + ch];
    }
    return slice;
}

Status TextureSet::ComputeChannelNormalizationParameters(std::array<ChannelInfo, NTC_MAX_CHANNELS>& outChannelInfos)
{
    float minimums[NTC_MAX_CHANNELS];
    float maximums[NTC_MAX_CHANNELS];

    assert(m_loss.Length() >= NTC_MAX_CHANNELS * 2);
    
    PitchLinearImageSlice slice = GetTextureDataSlice(TextureDataPage::Reference, 0, 0, m_desc.channels);
    cudaError_t err = cuda::ComputeMinMaxChannelValues(slice, (int*)m_loss.DevicePtr(), minimums, maximums);
    if (err != cudaSuccess)
    {
        SetCudaErrorMessage(__func__, err);
        return Status::CudaError;
    }

    // Use identity mapping for the mask channel.
    if (m_maskChannelIndex >= 0 && m_maskChannelIndex < NTC_MAX_CHANNELS)
    {
        minimums[m_maskChannelIndex] = 0.f;
        maximums[m_maskChannelIndex] = 1.f;
    }

    for (int ch = 0; ch < m_desc.channels; ++ch)
    {
        ChannelInfo& info = outChannelInfos[ch];
        if (minimums[ch] < maximums[ch])
        {
            // Map the (min..max) range to (0..1) using the equation (optimal = linear * scale + bias)
            info.linearToOptimalScale = 1.f / (maximums[ch] - minimums[ch]);
            info.linearToOptimalBias = -minimums[ch] * info.linearToOptimalScale;

            // Inverse mapping using the equation (linear = optimal * scale + bias)
            info.optimalToLinearScale = maximums[ch] - minimums[ch];
            info.optimalToLinearBias = minimums[ch];
        }
        else if (minimums[ch] == maximums[ch])
        {
            // Degenerate channel containing a constant value.
            // Make the network ignore it and produce a constant value with bias (scale = 0, bias = min)
            info.linearToOptimalScale = 0.f;
            info.linearToOptimalBias = -minimums[ch];

            info.optimalToLinearScale = 0.f;
            info.optimalToLinearBias = minimums[ch];
        }
        else // if (minimums[ch] > maximums[ch])
        {
            // Invalid range, probably a bug.
            // Use identity mapping to be safe (scale = 1, bias = 0).
            info = ChannelInfo();
        }
    }

    return Status::Ok;
}

}