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
#include "Context.h"
#include "CudaDeviceGuard.h"
#include "CudaRandomGen.h"
#include "Errors.h"
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

/* Note on the data formats and layouts used in various places...

=== Training and CUDA decompression ===

Training and CUDA decompression use FP16 weights and bias vectors. They are stored in the `m_mlpWeightsBase`
and `m_mlpWeightsQuantized` arrays. The "Base" array contains non-quantized weights that are used by the optimizer:
it takes the gradients from the regression kernel and applies them to the base weights. The "Quantized" weights are
used by the regression kernel. They are produced by the optimizer and at that point are not yet quantized. They become
quantized on the next step using the `QuantizeNetwork` calls in the final part of the training session.

Both of these arrays contain two copies of weights: the first set at offset 0 that get FP8 quantization, and 
the second set at offset `m_numNetworkParams` that get the Int8 quantization.

There are two parts in each set of weights: the actual layer weights, and the bias vectors. The layer weights are
stored in the obscure "MMA" layout that is compatible with the tensor core matrix multiplication operations. Their size
is equal to the number of elements in all layer matrices, i.e. there are no holes in the layout. Weights for layer 0
are immediately followed by weights for layer 1, and so on until layer 3. After the matrix weights, all bias vectors
are stored consecutively: dense bias vector for layer 0, followed by bias for layer 1, and so on until layer 3.

The FP16 weights and bias vectors are converted into the Int8 or FP8 format and row-major layout by the
`QuantizeNetwork` function after training is complete. These Int8 or FP8 weights are stored in the NTC files.
Before CUDA decompression, these low-precision weights are converted back into FP16 using the
`ConvertNetworkFromQuantizedToFp16` function.

=== Int8 inference ===

The Int8 inference weights are available in one generic layout that works for DP4a based inference on any GPU.

The weights package contains three components: the matrix weights, the scale vectors, and the bias vectors. The matrix
weights for all layers are stored in Int8 format and densely packed one after another. Then the scale vectors for all
layers are stored in Float32 format and are densely packed one after another. Finally, the bias vectors for all layers
are stored in Int32 format and also densely packed.

=== FP8 inference ===

Technically, it's hybrid FP8 and Int8 inference: layers 0-2 are using FP8 weights, and the output layer uses Int8
weights. This is done to improve the output precision, which is lacking with FP8 because that format cannot even
represent all integers in the range 0-255.

FP8 weights come in two flavors, the generic one (with row-major matrices) and the CoopVec specific one. The generic
weights are only used for storage, and not used by any GPU for inference because there are currently no scalar FP8
operations in GPUs.  The CoopVec weights are derived from the generic weights when the texture set is loaded from disk.
See `TextureSetMetadata::LoadWeightsFromStream`, `TextureSetMetadata::ConvertWeightsForCoopVec` and
`CoopVecWeightConverter.cpp`.

The generic weights contain the weight matrices, bias and scale vectors. They are laid out in the following order:
- Weight matrices for layers 0-2 using the FP8 type, in row-major layout
- Weight matrices for layer 3 using the Int8 type, in row-major layout
- Bias vectors for layers 0-2 using the FP16 type
- Scale vector for layer 3 using the FP32 type
- Bias vector for layer 3 using the Int32 type

The CoopVec weights follow the same general layout. The matrix weights are converted to the CoopVec-compatible
layout on load, which is normally larger than the dense row-major layout.

=== Important code locations using these layouts ===

                                                 | FP16 | GI8* | GFP8 | CVFP8 |
-------------------------------------------------+------+------+------+-------+
CUDA training and inference                      |      |      |      |       |
    RegressionKernels.h                          |  X   |      |      |       |
Weight quantization and conversion               |      |      |      |       |
    Quantizer.cu                                 |  X   |  X   |  X   |       |
Serialization                                    |      |      |      |       |
    TextureSet::SaveToStream                     |      |  X   |  X   |       |
Deserialization                                  |      |      |      |       |
    TextureSetMetadata::LoadWeightsFromStream    |      |  X   |  X   |       |
CoopVec layout conversion                        |      |      |      |       |
    TextureSetMetadata::ConvertWeightsForCoopVec |      |  X   |  X   |   X   |
    CoopVecWeightConverter.cpp                   |      |  X   |  X   |   X   |
GAPI decompression                               |      |      |      |       |
    * DecompressINT8.hlsl                        |      |  X   |      |       |
    * DecompressCoopVecFP8.slang                 |      |      |      |   X   |
GAPI inference                                   |      |      |      |       |
    * Inference.hlsli                            |      |  X   |      |       |
    * InferenceCoopVec.hlsli                     |      |      |      |   X   |

[*] GI8 = GenericInt8, GFP8 = GenericFP8, CVFP8 = CoopVecFP8

These layout descriptions and names match the InferenceWeightType enum declared in ntc.h

*/
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
        
        if (!ReadViewFromStream(inputStream, document, latentImage.view,
            m_featureGrid.GetEncodedPixelsHostPtr(neuralLod),
            m_featureGrid.GetEncodedPixelsSize(neuralLod)))
            return Status::IOError;
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
    MlpDesc const& mlpDesc = MlpDesc::Get();
    m_numNetworkParams = mlpDesc.GetWeightCount() + mlpDesc.GetLayerOutputCount();

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
        LatentImageDesc& imageDesc = m_latentImages.emplace_back();
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
        size_t bcDataSize = 0;
        info->GetBlockCompressionModeHistogram(nullptr, &bcDataSize);
        size += info->GetNameString().size() + bcDataSize;
    }
    
    // MLP
    size += m_mlpDataInt8.Size();
    size += m_mlpDataFP8.Size();

    // Latents
    size += m_featureGrid.GetTotalPixelCount() * m_featureGrid.GetNumLayers() * FeatureGridMath::BytesPerLatentPixel;

    size = RoundUp4(size);
    
    return size;
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

Status TextureSet::SaveToStream(IStream* outputStream)
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

    uint64_t binaryChunkSize = 0;
    auto appendView = [&document, &binaryChunkSize](uint64_t size) {
        uint32_t const viewIndex = uint32_t(document.views.size());
        json::BufferView& view = document.views.emplace_back(document.allocator);
        view.offset = binaryChunkSize;
        view.storedSize = size;
        binaryChunkSize += RoundUp4(size);
        return viewIndex;
    };

    WeightLayout const* weightLayoutInt8 = m_context->GetWeightLayout(InferenceWeightType::GenericInt8);
    assert(weightLayoutInt8);

    WeightLayout const* weightLayoutFP8 = m_context->GetWeightLayout(InferenceWeightType::GenericFP8);
    assert(weightLayoutFP8);

    MlpDesc const& mlpDesc = MlpDesc::Get();

    // Fill out the MLP layers - Int8
    for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        json::MLPLayer& layer = mlpInt8.layers.emplace_back(m_allocator);
        layer.inputChannels = mlpDesc.GetLayerInputChannels(layerIndex);
        layer.outputChannels = mlpDesc.GetLayerOutputChannels(layerIndex);
        layer.weightType = json::MlpDataType::Int8;
        layer.scaleType = json::MlpDataType::Float32;
        layer.biasType = json::MlpDataType::Int32;
        layer.weightView = appendView(weightLayoutInt8->weights[layerIndex].size);
        layer.scaleView = appendView(weightLayoutInt8->scales[layerIndex].size);
        layer.biasView = appendView(weightLayoutInt8->biases[layerIndex].size);
    }

    // Fill out the MLP layers - FP8
    // The MLP versions need to be in separate loops to keep view offsets consistent
    // with the actual writing order below
    for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        json::MLPLayer& layer = mlpFP8.layers.emplace_back(m_allocator);
        layer.inputChannels = mlpDesc.GetLayerInputChannels(layerIndex);
        layer.outputChannels = mlpDesc.GetLayerOutputChannels(layerIndex);
        layer.weightView = appendView(weightLayoutFP8->weights[layerIndex].size);

        if (layerIndex == NTC_MLP_LAYERS - 1)
        {
            layer.weightType = json::MlpDataType::Int8;
            layer.scaleType = json::MlpDataType::Float32;
            layer.biasType = json::MlpDataType::Int32;
            layer.scaleView = appendView(weightLayoutFP8->scales[layerIndex].size);
        }
        else
        {
            layer.weightType = json::MlpDataType::FloatE4M3;
            layer.biasType = json::MlpDataType::Float16;
        }

        layer.biasView = appendView(weightLayoutFP8->biases[layerIndex].size);
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

        if (info->GetBlockCompressionQuality() != BlockCompressionMaxQuality)
            texture.bcQuality = info->GetBlockCompressionQuality();

        if (info->HasBlockCompressionAccelerationData())
        {
            size_t bcDataSize = 0;
            info->GetBlockCompressionModeHistogram(nullptr, &bcDataSize);
            texture.bcAccelerationDataView = appendView(bcDataSize);
        }
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
        image.arraySize = m_featureGrid.GetNumLayers();
        size_t const pixelsSize = m_featureGrid.GetEncodedPixelsSize(neuralLod);
        image.view = appendView(pixelsSize);
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
    header.binaryChunkSize = binaryChunkSize;

    if (!outputStream->Write(&header, sizeof(header)))
        return Status::IOError;

    if (!outputStream->Write(jsonString.c_str(), jsonString.size() + 1))
        return Status::IOError;

    if (!PadStreamTo4Bytes(outputStream))
        return Status::IOError;

    auto validateOffset = [&document, &header, &outputStream](uint32_t view) {
        uint64_t expectedOffset = document.views[view].offset + header.binaryChunkOffset;
        uint64_t actualOffset = outputStream->Tell();
        assert(actualOffset == expectedOffset);
    };

    // Write the MLP data

    auto writeMlpData = [this, &outputStream, &validateOffset]
    (json::MLP const& mlp, DeviceAndHostArray<uint8_t>& buffer, WeightLayout const* weightLayout)
    {
        for (int layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
        {
            json::MLPLayer const& layer = mlp.layers[layerIndex];

            validateOffset(layer.weightView);

            Span const& layerWeights = weightLayout->weights[layerIndex];
            Span const& layerScale = weightLayout->scales[layerIndex];
            Span const& layerBias = weightLayout->biases[layerIndex];

            if (!outputStream->Write(buffer.HostPtr() + layerWeights.offset, layerWeights.size))
                return Status::IOError;
            if (!PadStreamTo4Bytes(outputStream))
                return Status::IOError;

            if (layer.scaleView.has_value())
            {
                validateOffset(*layer.scaleView);
                if (!outputStream->Write(buffer.HostPtr() + layerScale.offset, layerScale.size))
                    return Status::IOError;
            }

            validateOffset(layer.biasView);

            if (!outputStream->Write(buffer.HostPtr() + layerBias.offset, layerBias.size))
                return Status::IOError;
        }

        return Status::Ok;
    };

    Status status = writeMlpData(mlpInt8, m_mlpDataInt8, weightLayoutInt8);
    if (status != Status::Ok)
        return status;

    status = writeMlpData(mlpFP8, m_mlpDataFP8, weightLayoutFP8);
    if (status != Status::Ok)
        return status;
    
    // Write the texture BC data
    for (int textureIndex = 0; textureIndex < int(m_textureInfos.size()); ++textureIndex)
    {
        auto const& info = m_textureInfos[textureIndex];
        json::Texture const& texture = document.textures[textureIndex];

        void const* bcData = nullptr;
        size_t bcDataSize = 0;
        info->GetBlockCompressionModeHistogram(&bcData, &bcDataSize);

        if (bcDataSize != 0)
        {
            assert(texture.bcAccelerationDataView.has_value());
            validateOffset(*texture.bcAccelerationDataView);
         
            if (!outputStream->Write(bcData, bcDataSize))
                return Status::IOError;
            if (!PadStreamTo4Bytes(outputStream))
                return Status::IOError;
        }
    }

    // Write the latents data
    for (int neuralLod = 0; neuralLod < m_featureGrid.GetNumMipLevels(); ++neuralLod)
    {
        json::LatentImage const& image = document.latents[neuralLod];
        
        validateOffset(image.view);

        if (!outputStream->Write(
            m_featureGrid.GetEncodedPixelsHostPtr(neuralLod),
            m_featureGrid.GetEncodedPixelsSize(neuralLod)))
            return Status::IOError;
            
        if (!PadStreamTo4Bytes(outputStream))
            return Status::IOError;
    }

    // Verify that we've written no more than the number of bytes predicted by GetOutputStreamSize
    uint64_t expectedSize = GetOutputStreamSize();
    uint64_t actualSize = outputStream->Tell();

    if (actualSize > expectedSize)
    {
        SetErrorMessage("SaveToStream produced a stream with %" PRIu64 " bytes, while GetOutputStreamSize "
            "predicted no more than %" PRIu64 " bytes.", actualSize, expectedSize);
        return Status::InternalError;
    }

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

Status TextureSet::SaveToMemory(void *pData, size_t* pSize)
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

    status = SaveToStream(stream);
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

Status TextureSet::SaveToFile(char const *fileName)
{
    FileStreamWrapper stream(m_context);

    Status status = m_context->OpenFile(fileName, true, stream.ptr());
    if (status != Status::Ok)
        return status;
        
    return SaveToStream(stream);
}

Status TextureSet::LoadFromFile(char const *fileName)
{
    FileStreamWrapper stream(m_context);

    Status status = m_context->OpenFile(fileName, false, stream.ptr());
    if (status != Status::Ok)
        return status;
        
    return LoadFromStream(stream);
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

    MlpDesc const& mlpDesc = MlpDesc::Get();
    
    for (int i = 0; i < NTC_MLP_LAYERS; i++)
    {
        int const inputs = mlpDesc.GetLayerInputChannels(i);
        int const outputs = mlpDesc.GetLayerOutputChannels(i);
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

    MlpDesc const& mlpDesc = MlpDesc::Get();

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
            cuda::QuantizeNetwork(
                mlpDesc,
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
            int numNetworkParams = mlpDesc.GetWeightCount() + mlpDesc.GetLayerOutputCount();

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

    MlpDesc const& mlpDesc = MlpDesc::Get();

    // Quantize and compute FP8 weights, scale and bias values
    cuda::QuantizeNetwork(
        mlpDesc,
        m_mlpWeightsQuantized.DevicePtr(),
        (int8_t*)m_mlpDataFP8.DevicePtr(),
        /* useFP8 = */ true
    );

    // Quantize and compute Int8 weights, scale and bias values
    cuda::QuantizeNetwork(
        mlpDesc,
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
        
    MlpDesc const& mlpDesc = MlpDesc::Get();

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
        cuda::ConvertNetworkFromQuantizedToFp16(
            mlpDesc,
            m_mlpWeightsQuantized.DevicePtr(),
            (int8_t*)m_mlpDataFP8.DevicePtr(),
            /* useFP8 = */ true
        );

        // Convert the Int8 weights to FP16 into the second page of the FP16 MLP data,
        // in case we want to decompress using those for validation
        cuda::ConvertNetworkFromQuantizedToFp16(
            mlpDesc,
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