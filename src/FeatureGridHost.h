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

#include "CudaArray.h"
#include "FeatureGridMath.h"
#include <cuda_fp16.h>
#include <array>

namespace ntc
{

class CudaRandomGen;

class FeatureGrid : public FeatureGridMath
{
public:
    FeatureGrid(IAllocator* allocator);

    Status Initialize(int imageWidth, int imageHeight, int imageMips, int gridScale, int numFeatures,
        bool enableCompression);
    
    void Deallocate();
    
    void Fill(CudaRandomGen& rng);

    void ClearGradientMask();

    int LodToNeuralLod(int lod) const;
    
    half* GetBaseLatentsDevicePtr(int neuralLod);

    half* GetQuantizedLatentsDevicePtr(int neuralLod);

    float* GetMoment1DevicePtr();

    float* GetMoment2DevicePtr();

    void* GetGradientsDevicePtr();

    uint16_t* GetEncodedPixelsDevicePtr(int neuralLod);

    uint16_t* GetEncodedPixelsHostPtr(int neuralLod);

    size_t GetEncodedPixelsSize(int neuralLod);

    uint32_t* GetGradientMaskDevicePtr();

    DeviceAndHostArray<uint8_t>& GetEncodedPixelsArray();

    size_t GetLatentOffset(int neuralLod);

    size_t GetMaskOffset(int neuralLod) const;

    // Returns the total number of latent pixels across all mips, not accounting for layers or features.
    size_t GetTotalPixelCount() const { return m_totalPixels; }

    // Stride between groups of features for each pixel, in individual features (half or float).
    size_t GetLatentStride() const { return m_totalPixels * FeaturesPerGroup; }

    int GetNumLayers() const { return m_numLayers; }

    int GetNumMipLevels() const { return m_numNeuralMipLevels; }

private:
    int m_numFeatures = 0;
    int m_numLayers = 0;
    int m_gridScale = 0;

    std::array<size_t, NTC_MAX_MIPS> m_mipSizesInPixels {};
    std::array<size_t, NTC_MAX_MIPS> m_mipOffsetsInPixels {};
    size_t m_totalPixels = 0;
    int m_numNeuralMipLevels = 0;
    
    // Base, quantized latents and gradients use the NMHW2 layout: features/2, mip, height, width, 2.
    // The "2" at the end is because we store features as half2, aka FeaturesPerGroup.
    DeviceArray<half> m_baseLatentsMemory;
    DeviceArray<half> m_quantizedLatentsMemory;
    DeviceArray<uint32_t> m_gradientMemory; // declared as uint32_t, used as either float or half depending on 'stableTraining'
    DeviceArray<float> m_moment1Memory;
    DeviceArray<float> m_moment2Memory;

    // Encoded latents use the MNHW layout: mip, layer, height, width.
    DeviceAndHostArray<uint8_t> m_encodedPixelsMemory;

    // Gradient mask uses the MHW layout: mip, height, width.
    // Each item is a single bit, tightly packed into uint32_t, with no gaps between rows or mips.
    // One bit represents whether the corresponding pixel is updated in the current training step.
    DeviceArray<uint32_t> m_gradientMaskMemory;
};

}