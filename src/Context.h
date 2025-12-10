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
#include <memory>
#include <array>
#include <optional>
#include "WeightLayout.h"

namespace ntc
{

class GraphicsResources;

class Context : public IContext
{
public:
    // For internal use
    Context(ContextParameters const& params);
    ~Context() override;

    [[nodiscard]] IAllocator* GetAllocator() const { return m_allocator; }
    int GetCudaDevice() const { return m_cudaDevice; }
    bool IsCudaAvailable() const { return m_cudaDevice >= 0; }

    Status OpenFile(const char* fileName, bool write, IStream** pOutStream) const override;
    
    void CloseFile(IStream* stream) const override;
    
    Status OpenMemory(void* pData, size_t size, IStream** pOutStream) const override;
    
    Status OpenReadOnlyMemory(void const* pData, size_t size, IStream** pOutStream) const override;
    
    void CloseMemory(IStream* stream) const override;
    
    Status CreateTextureSet(const TextureSetDesc& desc,
        const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const override;
    
    void DestroyTextureSet(ITextureSet* textureSet) const override;

    Status CreateTextureSetMetadataFromStream(IStream* inputStream, ITextureSetMetadata** pOutMetadata) const override;
    
    void DestroyTextureSetMetadata(ITextureSetMetadata* textureSetMetadata) const override;

    Status CreateCompressedTextureSetFromStream(IStream* inputStream,
        const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const override;

    Status CreateCompressedTextureSetFromMemory(void const* pData, size_t size,
        const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const override;

    Status CreateCompressedTextureSetFromFile(char const* fileName,
        const TextureSetFeatures& features, ITextureSet** pOutTextureSet) const override;

    Status RegisterSharedTexture(const SharedTextureDesc& desc, ISharedTexture** pOutTexture) const override;

    void ReleaseSharedTexture(ISharedTexture* texture) const override;

    Status CreateAdaptiveCompressionSession(IAdaptiveCompressionSession** pOutSession) const override;

    void DestroyAdaptiveCompressionSession(IAdaptiveCompressionSession* session) const override;

    Status MakeDecompressionComputePass(MakeDecompressionComputePassParameters const& params,
        ComputePassDesc* pOutComputePass) const override;

    Status MakeBlockCompressionComputePass(MakeBlockCompressionComputePassParameters const& params,
        ComputePassDesc* pOutComputePass) const override;
    
    Status MakeImageDifferenceComputePass(MakeImageDifferenceComputePassParameters const& params,
        ComputePassDesc* pOutComputePass) const override;

    Status MakeInferenceData(ITextureSetMetadata* textureSetMetadata,
        InferenceWeightType weightType, int firstLatentMipInTexture, InferenceData* pOutInferenceData) const override;
    
    bool IsCooperativeVectorSupported() const override;

    Status DecompressBuffer(CompressionType compressionType, void const* pCompressedData, size_t compressedSize,
        void* pOutDecompressedData, size_t outputBufferSize, uint32_t expectedCrc32) const override;
    
    Status DecompressGDeflateOnVulkanGPU(void* commandBuffer,
        void const* pCompressedHeader, size_t compressedHeaderSize,
        uint64_t compressedGpuVA, uint64_t decompressedGpuVA) const override;

    GraphicsResources const* GetGraphicsResources() const { return m_graphicsResources; }

    WeightLayout const* GetWeightLayout(InferenceWeightType weightType) const;

    WeightLayout const& GetFP16WeightLayout() const { return m_fp16WeightLayout; }

private:
    IAllocator* m_allocator;
    int m_cudaDevice = -1;
    GraphicsResources* m_graphicsResources = nullptr;
    std::array<std::optional<WeightLayout>, size_t(InferenceWeightType::Count) - 1> m_weightLayouts{};
    WeightLayout m_fp16WeightLayout;

    static int GetWeightLayoutArrayIndex(InferenceWeightType weightType);
};

}