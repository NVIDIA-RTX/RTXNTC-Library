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
#include "ChannelInfo.h"
#include "FeatureGridHost.h"
#include "ImageProcessing.h"
#include "TextureSetMetadata.h"
#include <random>

namespace ntc
{

class Context;

enum class TextureSetNetworkState
{
    Empty,
    Initialized,
    TrainingInProgress,
    TrainingFinished,
    Complete
};

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4250) // Suppress warnings about methods of TextureSetMetadata being inherited via dominance
#endif

// Declaring virtual inheritance here leads to crashes on MSVC, not declaring it leads to crashes on GCC.
// To repro, make two compression runs in Explorer with different bitrate and restore the first one.
#ifdef _MSC_VER
class TextureSet : public TextureSetMetadata, public ITextureSet
#else
class TextureSet : virtual public TextureSetMetadata, virtual public ITextureSet
#endif
{
public:
    TextureSet(IAllocator* allocator, Context const* context, const TextureSetDesc& desc);
    ~TextureSet() override;

    Status Initialize(const TextureSetFeatures& features);

    Status LoadFromStreamPostHeader(json::Document const& document, uint64_t binaryChunkOffset,
        uint64_t binaryChunkSize, IStream* inputStream, LatentShape latentShape);

    Status SetLatentShape(LatentShape const& newShape) override;

    uint64_t GetOutputStreamSize() override;

    Status SaveToStream(IStream* stream, LosslessCompressionStats* pOutCompressionStats) override;
    
    Status LoadFromStream(IStream* stream) override;

    Status SaveToMemory(void* pData, size_t* pSize, LosslessCompressionStats* pOutCompressionStats) override;

    Status LoadFromMemory(void const* pData, size_t size) override;

    Status SaveToFile(char const* fileName, LosslessCompressionStats* pOutCompressionStats) override;

    Status LoadFromFile(char const* fileName) override;

    Status ConfigureLosslessCompression(LosslessCompressionSettings const& params) override;

    Status WriteChannels(WriteChannelsParameters const& params) override;

    Status ReadChannels(ReadChannelsParameters const& params) override;

    Status WriteChannelsFromTexture(WriteChannelsFromTextureParameters const& params) override;

    Status ReadChannelsIntoTexture(ReadChannelsIntoTextureParameters const& params) override;

    Status GenerateMips() override;

    Status BeginCompression(const CompressionSettings& settings) override;

    Status RunCompressionSteps(CompressionStats* pOutStats) override;

    Status FinalizeCompression() override;

    void AbortCompression() override;

    Status Decompress(DecompressionStats* pOutStats, bool useInt8Weights) override;

    Status SetMaskChannelIndex(int index, bool discardMaskedOutPixels) override;

    void SetExperimentalKnob(float value) override;

private:
    TextureSetFeatures m_features{};
    std::array<uint64_t, NTC_MAX_MIPS+1> m_textureMipOffsets{};
    int m_maskChannelIndex = -1;
    bool m_discardMaskedOutPixels = false;
    DeviceArray<half> m_textureData;
    DeviceArray<half> m_textureDataOut;
    DeviceArray<uint8_t> m_textureStaging;
    LosslessCompressionSettings m_losslessCompression;

    FeatureGrid m_featureGrid;
    
    DeviceArray<float> m_loss;
    DeviceAndHostArray<float> m_lossReduction;
    DeviceArray<half> m_mlpWeightsBase;
    DeviceArray<half> m_mlpWeightsQuantized;
    DeviceAndHostArray<uint8_t> m_mlpDataInt8;
    DeviceAndHostArray<uint8_t> m_mlpDataFP8;
     // declared as uint32_t, used as either float or half depending on 'stableTraining'
    DeviceArray<uint32_t> m_weightGradients;
    DeviceArray<float> m_mlpMoment1;
    DeviceArray<float> m_mlpMoment2;

    int m_numNetworkParams = 0;
    
    CompressionSettings m_compressionSettings{};
    int m_currentStep = 0;
    float m_lossScale = 0.f;
    float m_experimentalKnob = 0.f;

    cudaEvent_t m_eventStart = nullptr;
    cudaEvent_t m_eventStop = nullptr;

    std::mt19937 m_rng;

    TextureSetNetworkState m_networkState = TextureSetNetworkState::Empty;
    
    Status ValidateReadWriteChannelsArgs(int mipLevel, int firstChannel, int numChannels, int width, int height,
        size_t pixelStride, size_t rowPitch, size_t sizeToCopy, ChannelFormat format);

    PitchLinearImageSlice GetTextureDataSlice(TextureDataPage page, int mipLevel, int firstChannel, int numChannels);

    Status ComputeChannelNormalizationParameters(std::array<ChannelInfo, NTC_MAX_CHANNELS>& outChannelInfos);
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

}