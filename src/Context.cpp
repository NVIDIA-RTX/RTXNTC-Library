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

#include "AdaptiveCompressionSession.h"
#include "Context.h"
#include "CoopVecWeightConverter.h"
#include "CudaDeviceGuard.h"
#include "Errors.h"
#include "GraphicsResources.h"
#include "JsonFileFormat.h"
#include "MathUtils.h"
#include "MlpDesc.h"
#include "Shaders.h"
#include "Stream.h"
#include "TextureMetadata.h"
#include "TextureSetMetadata.h"

#if NTC_WITH_CUDA
#include "Regression.h"
#include "SharedTexture.h"
#include "TextureSet.h"
#endif

#include <cinttypes>
#include <cassert>
#include <cmath>
#include <cstring>

#include <libntc/shaders/DecompressConstants.h>
#include <libntc/shaders/BlockCompressConstants.h>
#include <libntc/shaders/ImageDifferenceConstants.h>

namespace ntc
{

Context::Context(ContextParameters const& params)
    : m_allocator(params.pAllocator)
    , m_cudaDevice(params.cudaDevice)
{
    if (params.graphicsApi != GraphicsAPI::None)
    {
        m_graphicsResources = new (m_allocator->Allocate(sizeof(GraphicsResources))) GraphicsResources(params);
    }

    // Fill the layout cache
    for (InferenceWeightType weightType : {
        InferenceWeightType::GenericInt8,
        InferenceWeightType::GenericFP8,
        InferenceWeightType::CoopVecFP8 })
    {
        if (CoopVecWeightConverter::IsCoopVecWeightType(weightType))
        {
            if (!CoopVecWeightConverter::IsConversionSupported(m_graphicsResources, weightType))
                continue;
        }

        WeightLayout layout;
        if (CoopVecWeightConverter::GetWeightLayout(m_graphicsResources,
            MlpDesc::Get(), weightType, layout))
        {
            m_weightLayouts[GetWeightLayoutArrayIndex(weightType)] = layout;
        }
    }
}

Context::~Context()
{
    if (m_graphicsResources)
    {
        m_graphicsResources->~GraphicsResources();
        m_allocator->Deallocate(m_graphicsResources, sizeof(GraphicsResources));
        m_graphicsResources = nullptr;
    }
}

Status Context::OpenFile(const char* fileName, bool write, IStream** pOutStream) const
{
    if (!fileName)
    {
        SetErrorMessage("fileName is NULL.");
        return Status::InvalidArgument;
    }

    if (!pOutStream)
    {
        SetErrorMessage("pOutStream is NULL.");
        return Status::InvalidArgument;
    }
    
    FILE* file = fopen(fileName, write ? "wb" : "rb");

    if (!file)
    {
        SetErrorMessage("Cannot open file '%s': %s.", fileName, strerror(errno));
        return Status::FileUnavailable;
    }

    *pOutStream = new(m_allocator->Allocate(sizeof(FileStream)))FileStream(file);

    return Status::Ok;
}

void Context::CloseFile(IStream* stream) const
{
    if (!stream)
        return;

    stream->~IStream();
    m_allocator->Deallocate(stream, sizeof(FileStream));
}

Status Context::OpenMemory(void* pData, size_t size, IStream** pOutStream) const
{
    if (!pData)
    {
        SetErrorMessage("pData is NULL.");
        return Status::InvalidArgument;
    }

    if (size == 0)
    {
        SetErrorMessage("size is 0.");
        return Status::InvalidArgument;
    }

    if (!pOutStream)
    {
        SetErrorMessage("pOutStream is NULL.");
        return Status::InvalidArgument;
    }

    *pOutStream = new(m_allocator->Allocate(sizeof(MemoryStream)))MemoryStream(
        static_cast<uint8_t*>(pData), size, false);
    
    return Status::Ok;
}

Status Context::OpenReadOnlyMemory(void const* pData, size_t size, IStream** pOutStream) const
{
    if (!pData)
    {
        SetErrorMessage("pData is NULL.");
        return Status::InvalidArgument;
    }

    if (size == 0)
    {
        SetErrorMessage("size is 0.");
        return Status::InvalidArgument;
    }

    if (!pOutStream)
    {
        SetErrorMessage("pOutStream is NULL.");
        return Status::InvalidArgument;
    }

    *pOutStream = new(m_allocator->Allocate(sizeof(MemoryStream)))MemoryStream(
        const_cast<uint8_t*>(static_cast<uint8_t const*>(pData)), size, true);
    
    return Status::Ok;
}

void Context::CloseMemory(IStream* stream) const
{
    if (!stream)
        return;

    stream->~IStream();
    m_allocator->Deallocate(stream, sizeof(MemoryStream));
}

Status Context::CreateTextureSet(const TextureSetDesc& desc,
                                 const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const
{
#if NTC_WITH_CUDA
    if (!pOutTextureSet)
    {
        SetErrorMessage("pOutTextureSet is NULL.");
        return Status::InvalidArgument;
    }

    if (!IsCudaAvailable())
    {
        SetErrorMessage("Cannot create a TextureSet object when no suitable CUDA device is available.");
        return Status::CudaUnavailable;
    }

    CudaDeviceGuard cudaGuard(this);
    if (!cudaGuard.Success())
        return Status::CudaError;

    Status status = TextureSetMetadata::ValidateTextureSetDesc(desc);
    if (status != Status::Ok)
        return status;

    TextureSet* textureSet = new(m_allocator->Allocate(sizeof(TextureSet)))
        TextureSet(m_allocator, this, desc);
    
    status = textureSet->Initialize(features);
    if (status != Status::Ok)
    {
        DestroyTextureSet(textureSet);
        return status;
    }
    
    *pOutTextureSet = textureSet;
    ClearErrorMessage();
    return Status::Ok;
#else
    SetErrorMessage("Cannot create TextureSet objects when LibNTC was compiled without CUDA support.");
    return Status::CudaUnavailable;
#endif
}

void Context::DestroyTextureSet(ITextureSet* textureSet) const
{
#if NTC_WITH_CUDA
    if (!textureSet)
        return;

    CudaDeviceGuard cudaGuard(this);
    if (!cudaGuard.Success())
        return;

    TextureSet* implementation = dynamic_cast<TextureSet*>(textureSet);
    textureSet->~ITextureSet();
    m_allocator->Deallocate(implementation, sizeof(TextureSet));
#endif
}

Status Context::CreateTextureSetMetadataFromStream(IStream* inputStream, ITextureSetMetadata** pOutMetadata) const
{
    if (!pOutMetadata)
    {
        SetErrorMessage("pOutMetadata is NULL.");
        return Status::InvalidArgument;
    }

    if (!inputStream)
    {
        SetErrorMessage("inputStream is NULL.");
        return Status::InvalidArgument;
    }

    json::Document document(m_allocator);
    uint64_t binaryChunkOffset, binaryChunkSize;
    Status status = TextureSetMetadata::LoadFileHeadersFromStream(m_allocator, inputStream, document,
        binaryChunkOffset, binaryChunkSize);
    if (status != Status::Ok)
        return status;

    TextureSetDesc desc;
    LatentShape latentShape;
    status = TextureSetMetadata::DeserializeTextureSetDesc(document, desc, latentShape);
    if (status != Status::Ok)
        return status;

    TextureSetMetadata* textureSetMetadata = new(m_allocator->Allocate(sizeof(TextureSetMetadata)))
        TextureSetMetadata(m_allocator, this, desc, latentShape);

    status = textureSetMetadata->LoadMetadataFromStream(document, binaryChunkOffset, binaryChunkSize,
        latentShape, inputStream);

    if (status == Status::Ok)
    {
        status = textureSetMetadata->LoadWeightsFromStream(document, inputStream);
    }

    if (status != Status::Ok)
    {
        DestroyTextureSetMetadata(textureSetMetadata);
        return status;
    }

    ClearErrorMessage();
    *pOutMetadata = textureSetMetadata;
    return Status::Ok;
}

void Context::DestroyTextureSetMetadata(ITextureSetMetadata* textureSetMetadata) const
{
    if (!textureSetMetadata)
        return;

    TextureSetMetadata* implementation = dynamic_cast<TextureSetMetadata*>(textureSetMetadata);
    textureSetMetadata->~ITextureSetMetadata();
    m_allocator->Deallocate(implementation, sizeof(TextureSetMetadata));
}

Status Context::CreateCompressedTextureSetFromStream(IStream* inputStream,
    const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const
{
#if NTC_WITH_CUDA
    if (!pOutTextureSet)
    {
        SetErrorMessage("pOutTextureSet is NULL.");
        return Status::InvalidArgument;
    }

    if (!inputStream)
    {
        SetErrorMessage("inputStream is NULL.");
        return Status::InvalidArgument;
    }

    if (!IsCudaAvailable())
    {
        SetErrorMessage("Cannot create a TextureSet object when no suitable CUDA device is available.");
        return Status::CudaUnavailable;
    }

    CudaDeviceGuard cudaGuard(this);
    if (!cudaGuard.Success())
        return Status::CudaError;
        
    json::Document document(m_allocator);
    uint64_t binaryChunkOffset, binaryChunkSize;
    Status status = TextureSetMetadata::LoadFileHeadersFromStream(m_allocator, inputStream, document,
        binaryChunkOffset, binaryChunkSize);
    if (status != Status::Ok)
        return status;

    TextureSetDesc desc{};
    LatentShape latentShape;
    status = TextureSetMetadata::DeserializeTextureSetDesc(document, desc, latentShape);
    if (status != Status::Ok)
        return status;

    TextureSet* textureSet = nullptr;
    status = CreateTextureSet(desc, features, (ITextureSet**)&textureSet);
    if (status != Status::Ok)
        return status;

    status = textureSet->LoadFromStreamPostHeader(document, binaryChunkOffset, binaryChunkSize, inputStream, latentShape);
    if (status != Status::Ok)
    {
        DestroyTextureSet(textureSet);
        return status;
    }

    *pOutTextureSet = textureSet;
    
    ClearErrorMessage();
    return Status::Ok;
#else
    SetErrorMessage("Cannot create TextureSet objects when LibNTC was compiled without CUDA support.");
    return Status::CudaUnavailable;
#endif
}

Status Context::CreateCompressedTextureSetFromMemory(void const* pData, size_t size,
    const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const
{
    MemoryStreamWrapper stream(this);

    Status status = OpenReadOnlyMemory(pData, size, stream.ptr());
    if (status != Status::Ok)
        return status;
        
    return CreateCompressedTextureSetFromStream(stream, features, pOutTextureSet);
}

Status Context::CreateCompressedTextureSetFromFile(char const* fileName,
    const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const
{
    FileStreamWrapper stream(this);

    Status status = OpenFile(fileName, false, stream.ptr());
    if (status != Status::Ok)
        return status;
        
    return CreateCompressedTextureSetFromStream(stream, features, pOutTextureSet);
}

Status Context::RegisterSharedTexture(const SharedTextureDesc& desc, ISharedTexture** pOutTexture) const
{
#if NTC_WITH_CUDA
    if (!pOutTexture)
    {
        SetErrorMessage("pOutTexture is NULL.");
        return Status::InvalidArgument;
    }

    if (!IsCudaAvailable())
    {
        SetErrorMessage("Cannot create a SharedTexture object when no suitable CUDA device is available.");
        return Status::CudaUnavailable;
    }
    
    CudaDeviceGuard cudaGuard(this);
    if (!cudaGuard.Success())
        return Status::CudaError;

    SharedTexture* sharedTexture = new(m_allocator->Allocate(sizeof(SharedTexture))) SharedTexture(desc);
    
    Status status = sharedTexture->Initialize();
    if (status != Status::Ok)
    {
        ReleaseSharedTexture(sharedTexture);
        return status;
    }

    *pOutTexture = sharedTexture;
    ClearErrorMessage();
    return Status::Ok;
#else
    SetErrorMessage("Cannot create SharedTexture objects when LibNTC was compiled without CUDA support.");
    return Status::CudaUnavailable;
#endif
}

void Context::ReleaseSharedTexture(ISharedTexture* texture) const
{
#if NTC_WITH_CUDA
    if (!texture)
        return;

    CudaDeviceGuard cudaGuard(this);
    if (!cudaGuard.Success())
        return;

    texture->~ISharedTexture();
    m_allocator->Deallocate(texture, sizeof(SharedTexture));
#endif
}

Status Context::CreateAdaptiveCompressionSession(IAdaptiveCompressionSession** pOutSession) const
{
    if (!pOutSession)
    {
        SetErrorMessage("pOutSession is NULL");
        return Status::InvalidArgument;
    }

    *pOutSession = new (m_allocator->Allocate(sizeof(AdaptiveCompressionSession))) AdaptiveCompressionSession();
    ClearErrorMessage();
    return Status::Ok;
}

void Context::DestroyAdaptiveCompressionSession(IAdaptiveCompressionSession *session) const
{
    if (!session)
        return;

    m_allocator->Deallocate(session, sizeof(AdaptiveCompressionSession));
}

Status Context::MakeDecompressionComputePass(MakeDecompressionComputePassParameters const& params, ComputePassDesc* pOutComputePass) const
{
    if (!pOutComputePass)
    {
        SetErrorMessage("pOutComputePass is NULL");
        return Status::InvalidArgument;
    }

    if (!m_graphicsResources)
    {
        SetErrorMessage("The context was initialized with graphicsApi == None");
        return Status::InvalidArgument;
    }

    if (!params.pOutputTextures && params.numOutputTextures != 0)
    {
        SetErrorMessage("pOutputTextures is NULL while numOutputTextures (%d) is nonzero", params.numOutputTextures);
        return Status::InvalidArgument;
    }

    if (params.numOutputTextures > DECOMPRESS_CS_MAX_OUTPUTS)
    {
        SetErrorMessage("numOutputTextures (%d) is too large, must be %d or less", params.numOutputTextures, DECOMPRESS_CS_MAX_OUTPUTS);
        return Status::InvalidArgument;
    }

    // Validate the output textures
    for (int textureIndex = 0; textureIndex < params.numOutputTextures; ++textureIndex)
    {
        OutputTextureDesc const& desc = params.pOutputTextures[textureIndex];

        if (desc.firstChannel < 0 || desc.numChannels <= 0 || desc.numChannels > 4 ||
            desc.firstChannel + desc.numChannels >= NTC_MLP_OUTPUT_CHANNELS)
        {
            SetErrorMessage("pOutputTextures[%d] has invalid channel configuration: firstChannel = %d, numChannels = %d",
                textureIndex, desc.firstChannel, desc.numChannels);
            return Status::InvalidArgument;
        }

        if (desc.descriptorIndex < 0)
        {
            SetErrorMessage("pOutputTextures[%d] has invalid descriptorOffset (%d)",
                textureIndex, desc.descriptorIndex);
            return Status::InvalidArgument;
        }
    }

    TextureSetMetadata* textureSetMetadata = dynamic_cast<TextureSetMetadata*>(params.textureSetMetadata);
    if (!textureSetMetadata)
    {
        SetErrorMessage("textureSetMetadata is NULL or points at a wrong object type");
        return Status::InvalidArgument;
    }

    TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
    LatentShape const& latentShape = textureSetMetadata->GetLatentShape();
    
    if (params.mipLevel < 0 || params.mipLevel >= textureSetDesc.mips)
    {
        SetErrorMessage("mipLevel (%d) must be between 0 and %d.", params.mipLevel, textureSetDesc.mips - 1);
        return Status::OutOfRange;
    }

    // Pre-compute some parameters needed to pick the right shader
    int const mipWidth = std::max(textureSetDesc.width >> params.mipLevel, 1);
    int const mipHeight = std::max(textureSetDesc.height >> params.mipLevel, 1);
    int const neuralLod = textureSetMetadata->ColorMipToNeuralLod(params.mipLevel);
    LatentImageDesc const* latentImage = textureSetMetadata->GetLatentImageDesc(neuralLod);
    if (!latentImage)
    {
        // This shouldn't happen with all the validation above, but let's be sure
        SetErrorMessage("latentImage is NULL");
        return Status::InternalError;
    }

    if (neuralLod < params.firstLatentMipInTexture)
    {
        SetErrorMessage("mipLevel (%d) cannot be decompressed when the texture set is partially loaded, as indicated "
            " by firstLatentMipInTexture (%d).", params.mipLevel, params.firstLatentMipInTexture);
        return Status::OutOfRange;
    }

    // Select the shader version
    InferenceMath mathVersion;
    switch(params.weightType)
    {
        case InferenceWeightType::GenericInt8:
            mathVersion = InferenceMath::DP4a;
            break;
            
        case InferenceWeightType::CoopVecFP8:
            mathVersion = InferenceMath::CoopVecFP8;
            break;

        default:
            SetErrorMessage("Unsupported weightType (%s)", InferenceWeightTypeToString(params.weightType));
            return Status::InvalidArgument;
    }
    
    pOutComputePass->computeShader = nullptr;
    pOutComputePass->computeShaderSize = 0;
    
#if NTC_WITH_PREBUILT_SHADERS
    switch (m_graphicsResources->GetGraphicsApi())
    {
        case GraphicsAPI::D3D12:
#if NTC_WITH_DX12
            GetDecompressDxilShaderBytecode(mathVersion,
                &pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        case GraphicsAPI::Vulkan:
#if NTC_WITH_VULKAN
            GetDecompressSpirvShaderBytecode(mathVersion,
                &pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        default:
            // Shouldn't happen - None is handled above and invalid values are handled at context initialization
            return Status::InvalidArgument;
    }
#endif

    Rect srcRect { 0, 0, mipWidth, mipHeight };
    if (params.pSrcRect)
    {
        int const left = params.pSrcRect->left;
        int const top = params.pSrcRect->top;
        int const width = params.pSrcRect->width;
        int const height = params.pSrcRect->height;
        int const right = left + width;
        int const bottom = top + height;
        if (left < 0 || top < 0 || width <= 0 || height <= 0 || right > mipWidth || bottom > mipHeight)
        {
            SetErrorMessage("Invalid rectangle specified. For mip %d, left (%d) and top (%d) must be >= 0; "
                "width (%d) and height (%d) must be > 0; (left + width) (%d) must be <= %d; "
                "(top + height) must be <= %d.", 
                params.mipLevel, left, top, width, height, right, bottom, mipHeight);
            return Status::OutOfRange;
        }
        srcRect = *params.pSrcRect;
    }

    Point dstOffset { srcRect.left, srcRect.top };
    if (params.pDstOffset)
    {
        dstOffset = *params.pDstOffset;
    }

    NtcDecompressConstants& constants = reinterpret_cast<NtcDecompressConstants&>(pOutComputePass->constantBufferData);
    static_assert(sizeof(NtcDecompressConstants) <= sizeof(ComputePassDesc::constantBufferData),
        "NtcDecompressConstants don't fit into constantBufferData. Increase MaxComputePassConstantSize.");
    pOutComputePass->constantBufferSize = sizeof(NtcDecompressConstants);

    textureSetMetadata->FillDecompressionConstants(constants, params, srcRect, dstOffset);

    int const gridWidth = constants.srcRight - constants.srcLeft;
    int const gridHeight = constants.srcBottom - constants.srcTop;

    pOutComputePass->dispatchWidth = (gridWidth + DECOMPRESS_CS_BLOCK_WIDTH - 1) / DECOMPRESS_CS_BLOCK_WIDTH;
    pOutComputePass->dispatchHeight = (gridHeight + DECOMPRESS_CS_BLOCK_HEIGHT - 1) / DECOMPRESS_CS_BLOCK_HEIGHT;

    if (!pOutComputePass->computeShader)
    {
        SetErrorMessage("The requested shader binary is unavailable.");
        return Status::ShaderUnavailable;
    }

    return Status::Ok;
}

Status Context::MakeBlockCompressionComputePass(MakeBlockCompressionComputePassParameters const& params,
    ComputePassDesc* pOutComputePass) const
{
    if (!pOutComputePass)
    {
        SetErrorMessage("pOutComputePass is NULL");
        return Status::InvalidArgument;
    }

    if (!m_graphicsResources)
    {
        SetErrorMessage("The context was initialized with graphicsApi == None");
        return Status::InvalidArgument;
    }

    if (params.srcRect.left < 0 || params.srcRect.top < 0 || params.srcRect.width <= 0 || params.srcRect.height <= 0)
    {
        SetErrorMessage("srcRect.left (%d) and srcRect.top (%d) must be >= 0; srcRect.width (%d) and "
            "srcRect.height (%d) must be > 0",
            params.srcRect.left, params.srcRect.top, params.srcRect.width, params.srcRect.height);
        return Status::OutOfRange;
    }

    if (params.dstOffsetInBlocks.x < 0 || params.dstOffsetInBlocks.y < 0)
    {
        SetErrorMessage("dstOffsetInBlocks.x (%d) and dstOffsetInBlocks.y (%d) must be >= 0",
            params.dstOffsetInBlocks.x, params.dstOffsetInBlocks.y);
        return Status::OutOfRange;
    }

    if (params.dstFormat < BlockCompressedFormat::BC1 || params.dstFormat > BlockCompressedFormat::BC7)
    {
        SetErrorMessage("dstFormat (%s) has invalid value", BlockCompressedFormatToString(params.dstFormat));
        return Status::OutOfRange;
    }

    pOutComputePass->computeShader = nullptr;
    pOutComputePass->computeShaderSize = 0;
    
#if NTC_WITH_PREBUILT_SHADERS
    switch (m_graphicsResources->GetGraphicsApi())
    {
        case GraphicsAPI::D3D12:
#if NTC_WITH_DX12
            GetBlockCompressDxilShaderBytecode(params.dstFormat, params.writeAccelerationData,
                &pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        case GraphicsAPI::Vulkan:
#if NTC_WITH_VULKAN
            GetBlockCompressSpirvShaderBytecode(params.dstFormat, params.writeAccelerationData,
                &pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        default:
            // Shouldn't happen - None is handled above and invalid values are handled at context initialization
            return Status::InvalidArgument;
    }
#endif

    NtcBlockCompressConstants& constants = reinterpret_cast<NtcBlockCompressConstants&>(pOutComputePass->constantBufferData);
    static_assert(sizeof(NtcBlockCompressConstants) <= sizeof(ComputePassDesc::constantBufferData),
        "NtcBlockCompressConstants don't fit into constantBufferData. Increase MaxComputePassConstantSize.");
    pOutComputePass->constantBufferSize = sizeof(NtcBlockCompressConstants);

    constants.srcLeft = params.srcRect.left;
    constants.srcTop = params.srcRect.top;
    constants.dstOffsetX = params.dstOffsetInBlocks.x;
    constants.dstOffsetY = params.dstOffsetInBlocks.y;
    constants.widthInBlocks = (params.srcRect.width + 3) / 4;
    constants.heightInBlocks = (params.srcRect.height + 3) / 4;
    constants.alphaThreshold = params.alphaThreshold;

    memset(constants.allowedModes, 0xff, sizeof(constants.allowedModes));
    if (params.texture && params.dstFormat == BlockCompressedFormat::BC7 && !params.writeAccelerationData)
    {
        static_cast<TextureMetadata const*>(params.texture)->GetAllowedBCModes(constants.allowedModes,
            sizeof(constants.allowedModes), params.quality);
    }
    
    pOutComputePass->dispatchWidth = (constants.widthInBlocks + BLOCK_COMPRESS_CS_ST_GROUP_WIDTH - 1)
        / BLOCK_COMPRESS_CS_ST_GROUP_WIDTH;
    pOutComputePass->dispatchHeight = (constants.heightInBlocks + BLOCK_COMPRESS_CS_ST_GROUP_HEIGHT - 1)
        / BLOCK_COMPRESS_CS_ST_GROUP_HEIGHT;

    if (!pOutComputePass->computeShader)
    {
        SetErrorMessage("The requested shader binary is unavailable.");
        return Status::ShaderUnavailable;
    }

    return Status::Ok;
}

Status Context::MakeImageDifferenceComputePass(MakeImageDifferenceComputePassParameters const& params,
    ComputePassDesc* pOutComputePass) const
{
    if (!pOutComputePass)
    {
        SetErrorMessage("pOutComputePass is NULL");
        return Status::InvalidArgument;
    }

    if (!m_graphicsResources)
    {
        SetErrorMessage("The context was initialized with graphicsApi == None");
        return Status::InvalidArgument;
    }

    if (params.extent.left < 0 || params.extent.top < 0)
    {
        SetErrorMessage("Left (%d) and top (%d) must be non-negative", params.extent.left, params.extent.top);
        return Status::OutOfRange;
    }
    
    if (params.extent.width <= 0 || params.extent.height <= 0)
    {
        SetErrorMessage("Width (%d) and height (%d) must be positive", params.extent.width, params.extent.height);
        return Status::OutOfRange;
    }

    if ((params.outputOffset & 3) != 0)
    {
        SetErrorMessage("outputOffset (%d) must be aligned to 4 bytes", params.outputOffset);
        return Status::OutOfRange;
    }

    pOutComputePass->computeShader = nullptr;
    pOutComputePass->computeShaderSize = 0;

#if NTC_WITH_PREBUILT_SHADERS
    switch (m_graphicsResources->GetGraphicsApi())
    {
        case GraphicsAPI::D3D12:
#if NTC_WITH_DX12
            GetImageDifferenceDxilShaderBytecode(&pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        case GraphicsAPI::Vulkan:
#if NTC_WITH_VULKAN
            GetImageDifferenceSpirvShaderBytecode(&pOutComputePass->computeShader, &pOutComputePass->computeShaderSize);
#endif
            break;
        default:
            // Shouldn't happen - None is handled above and invalid values are handled at context initialization
            return Status::InvalidArgument;
    }
#endif

    NtcImageDifferenceConstants& constants = reinterpret_cast<NtcImageDifferenceConstants&>(pOutComputePass->constantBufferData);
    static_assert(sizeof(NtcImageDifferenceConstants) <= sizeof(ComputePassDesc::constantBufferData),
        "NtcImageDifferenceConstants don't fit into constantBufferData. Increase MaxComputePassConstantSize.");
    pOutComputePass->constantBufferSize = sizeof(NtcImageDifferenceConstants);

    constants.left = params.extent.left;
    constants.top = params.extent.top;
    constants.width = params.extent.width;
    constants.height = params.extent.height;
    constants.alphaThreshold = params.alphaThreshold;
    constants.useAlphaThreshold = params.useAlphaThreshold ? 1 : 0;
    constants.useMSLE = params.useMSLE ? 1 : 0;
    constants.outputOffset = params.outputOffset;
    
    pOutComputePass->dispatchWidth = (params.extent.width + IMAGE_DIFFERENCE_CS_PIXELS_PER_BLOCK_X - 1)
        / IMAGE_DIFFERENCE_CS_PIXELS_PER_BLOCK_X;
    pOutComputePass->dispatchHeight = (params.extent.height + IMAGE_DIFFERENCE_CS_PIXELS_PER_BLOCK_Y - 1)
        / IMAGE_DIFFERENCE_CS_PIXELS_PER_BLOCK_Y;

    if (!pOutComputePass->computeShader)
    {
        SetErrorMessage("The requested shader binary is unavailable.");
        return Status::ShaderUnavailable;
    }

    return Status::Ok;
}

Status Context::MakeInferenceData(ITextureSetMetadata* _textureSetMetadata,
    InferenceWeightType weightType, int firstLatentMipInTexture, InferenceData* pOutInferenceData) const
{
    if (!pOutInferenceData)
    {
        SetErrorMessage("pOutInferenceData is NULL");
        return Status::InvalidArgument;
    }

    TextureSetMetadata* textureSetMetadata = dynamic_cast<TextureSetMetadata*>(_textureSetMetadata);
    if (!textureSetMetadata)
    {
        SetErrorMessage("textureSetMetadata is NULL or points at a wrong object type");
        return Status::InvalidArgument;
    }

    switch(weightType)
    {
        case InferenceWeightType::GenericInt8:
        case InferenceWeightType::GenericFP8:
        case InferenceWeightType::CoopVecFP8:
            break;
        default:
            SetErrorMessage("Unsupported weightType (%s)", InferenceWeightTypeToString(weightType));
            return Status::Unsupported;
    }

    if (!textureSetMetadata->IsInferenceWeightTypeSupported(weightType))
    {
        SetErrorMessage("The texture set does not provide %s weights", InferenceWeightTypeToString(weightType));
        return Status::Unsupported;
    }
    
    memset(pOutInferenceData, 0, sizeof(InferenceData));

    TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
    LatentShape const& latentShape = textureSetMetadata->GetLatentShape();

    for (int mipLevel = 0; mipLevel < textureSetDesc.mips; ++mipLevel)
    {
        assert(mipLevel < NTC_MAX_MIPS);
        
        textureSetMetadata->FillColorMipConstants(pOutInferenceData->constants.colorMips[mipLevel],
            mipLevel, firstLatentMipInTexture);
    }
    
    pOutInferenceData->constants.imageWidth = textureSetDesc.width;
    pOutInferenceData->constants.imageHeight = textureSetDesc.height;
    pOutInferenceData->constants.imageMips = textureSetDesc.mips;
    textureSetMetadata->GetWeightOffsets(
        weightType,
        pOutInferenceData->constants.networkWeightOffsets,
        pOutInferenceData->constants.networkBiasOffsets,
        pOutInferenceData->constants.networkScaleOffsets);
    pOutInferenceData->constants.validChannelMask = textureSetMetadata->GetValidChannelMask();
    pOutInferenceData->constants.channelColorSpaces = textureSetMetadata->GetPackedColorSpaces();
 
    return Status::Ok;
}

bool Context::IsCooperativeVectorSupported() const
{
    if (m_graphicsResources)
        return m_graphicsResources->IsCoopVecSupported();

    return false;
}

WeightLayout const* Context::GetWeightLayout(InferenceWeightType weightType) const
{
    int const index = GetWeightLayoutArrayIndex(weightType);
    if (index < 0)
        return nullptr;

    if (m_weightLayouts[index].has_value())
        return &*m_weightLayouts[index];

    return nullptr;
}

int Context::GetWeightLayoutArrayIndex(InferenceWeightType weightType)
{
    if (weightType < InferenceWeightType::GenericInt8 || weightType > InferenceWeightType::CoopVecFP8)
        return -1;

    return int(weightType) - int(InferenceWeightType::GenericInt8);
}

}
