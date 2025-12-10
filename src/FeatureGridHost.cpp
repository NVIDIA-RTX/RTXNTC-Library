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

#include "FeatureGridHost.h"
#include "Quantizer.h"
#include "CudaRandomGen.h"
#include <cmath>
#include <cassert>
#include <random>

namespace ntc
{

FeatureGrid::FeatureGrid(IAllocator* allocator)
    : m_encodedPixelsMemory(allocator)
{ }

Status FeatureGrid::Initialize(int imageWidth, int imageHeight, int imageMips, int gridScale, int numFeatures,
    bool enableCompression)
{
    m_numFeatures = numFeatures;
    m_gridScale = gridScale;

    m_numNeuralMipLevels = CalculateNumNeuralMipLevels(imageWidth, imageHeight, gridScale);

    // Clamp the theoretical mip count with the one that will be used for this texture set
    const int lastNeuralMip = FeatureGridMath::LodToNeuralLod(imageMips - 1, gridScale, NTC_MAX_NEURAL_MIPS);
    m_numNeuralMipLevels = std::min(m_numNeuralMipLevels, lastNeuralMip + 2);

    assert(m_numNeuralMipLevels > 0);
    assert((m_numNeuralMipLevels & 1) == 0); // Must be even
    assert(m_numNeuralMipLevels <= NTC_MAX_NEURAL_MIPS);

    m_numLayers = FeatureGridMath::GetNumLayers(numFeatures);

    m_totalPixels = 0;

    for (int mip = 0; mip < m_numNeuralMipLevels; mip++)
    {
        int const mipWidth = GetGridDimension(imageWidth, mip, gridScale);
        int const mipHeight = GetGridDimension(imageHeight, mip, gridScale);

        size_t const pixelsInThisMip = size_t(mipWidth) * size_t(mipHeight);
        
        m_mipSizesInPixels[mip] = pixelsInThisMip;
        m_mipOffsetsInPixels[mip] = m_totalPixels;
        m_totalPixels += pixelsInThisMip;
    }

    size_t const totalMaskUints = (m_totalPixels + 31) / 32; // Round up to the nearest multiple of 32 bits
    size_t const totalPixelsInAllLayers = m_totalPixels * size_t(m_numLayers);
    
    size_t const totalLatents = totalPixelsInAllLayers * FeaturesPerLayer;
    if (!m_quantizedLatentsMemory.Allocate(totalLatents))
        return Status::OutOfMemory;

    if (!m_encodedPixelsMemory.Allocate(totalPixelsInAllLayers * BytesPerLatentPixel))
        return Status::OutOfMemory;

    if (enableCompression)
    {
        if (!m_baseLatentsMemory.Allocate(totalLatents))    return Status::OutOfMemory;
        if (!m_gradientMemory.Allocate(totalLatents))       return Status::OutOfMemory;
        if (!m_moment1Memory.Allocate(totalLatents))        return Status::OutOfMemory;
        if (!m_moment2Memory.Allocate(totalLatents))        return Status::OutOfMemory;
        if (!m_gradientMaskMemory.Allocate(totalMaskUints)) return Status::OutOfMemory;
    }

    return Status::Ok;
}

void FeatureGrid::Deallocate()
{
    m_quantizedLatentsMemory.Deallocate();
    m_baseLatentsMemory.Deallocate();
    m_encodedPixelsMemory.Deallocate();
    m_gradientMemory.Deallocate();
    m_moment1Memory.Deallocate();
    m_moment2Memory.Deallocate();
    m_gradientMaskMemory.Deallocate();
}

void FeatureGrid::Fill(CudaRandomGen& rng)
{
    rng.FillRandomNormalHalf(m_baseLatentsMemory.DevicePtr(),
        uint32_t(m_baseLatentsMemory.Length()),
        0.25f, 0.5f, 0.f, 1.f);

    cudaMemcpy(m_quantizedLatentsMemory.DevicePtr(), m_baseLatentsMemory.DevicePtr(),
        m_baseLatentsMemory.Size(), cudaMemcpyDeviceToDevice);
    cudaMemset(m_gradientMemory.DevicePtr(), 0, m_gradientMemory.Size());
    cudaMemset(m_moment1Memory.DevicePtr(), 0, m_moment1Memory.Size());
    cudaMemset(m_moment2Memory.DevicePtr(), 0, m_moment2Memory.Size());
}

void FeatureGrid::ClearGradientMask()
{
    cudaMemset(m_gradientMaskMemory.DevicePtr(), 0, m_gradientMaskMemory.Size());
}

int FeatureGrid::LodToNeuralLod(int lod) const
{
    return FeatureGridMath::LodToNeuralLod(lod, m_gridScale, m_numNeuralMipLevels);
}

half* FeatureGrid::GetBaseLatentsDevicePtr(int neuralLod)
{
    return m_baseLatentsMemory.DevicePtrOffset(GetLatentOffset(neuralLod));
}

half* FeatureGrid::GetQuantizedLatentsDevicePtr(int neuralLod)
{
    return m_quantizedLatentsMemory.DevicePtrOffset(GetLatentOffset(neuralLod));
}

float* FeatureGrid::GetMoment1DevicePtr()
{
    return m_moment1Memory.DevicePtr();
}

float* FeatureGrid::GetMoment2DevicePtr()
{
    return m_moment2Memory.DevicePtr();
}

void* FeatureGrid::GetGradientsDevicePtr()
{
    return m_gradientMemory.DevicePtr();
}

uint16_t* FeatureGrid::GetEncodedPixelsDevicePtr(int neuralLod)
{
    assert(neuralLod < m_numNeuralMipLevels);
    size_t offset = m_mipOffsetsInPixels[neuralLod] * size_t(m_numLayers) * BytesPerLatentPixel;
    assert((offset % 2) == 0);
    return (uint16_t*)m_encodedPixelsMemory.DevicePtrOffset(offset);
}

uint16_t* FeatureGrid::GetEncodedPixelsHostPtr(int neuralLod, int layerIndex)
{
    assert(neuralLod < m_numNeuralMipLevels);
    size_t const offset = (m_mipOffsetsInPixels[neuralLod] * size_t(m_numLayers)
                         + m_mipSizesInPixels[neuralLod] * size_t(layerIndex)) * BytesPerLatentPixel;
    assert((offset % 2) == 0);
    return (uint16_t*)m_encodedPixelsMemory.HostPtrOffset(offset);
}

size_t FeatureGrid::GetEncodedPixelsSizePerLayer(int neuralLod)
{
    assert(neuralLod < m_numNeuralMipLevels);
    return m_mipSizesInPixels[neuralLod] * BytesPerLatentPixel;
}

uint32_t* FeatureGrid::GetGradientMaskDevicePtr()
{
    return m_gradientMaskMemory.DevicePtr();
}

DeviceAndHostArray<uint8_t>& FeatureGrid::GetEncodedPixelsArray()
{
    return m_encodedPixelsMemory;
}

size_t FeatureGrid::GetLatentOffset(int neuralLod)
{
    assert(neuralLod < m_numNeuralMipLevels);
    return m_mipOffsetsInPixels[neuralLod] * FeaturesPerGroup;
}

size_t FeatureGrid::GetMaskOffset(int neuralLod) const
{ 
    assert(neuralLod < m_numNeuralMipLevels);
    return m_mipOffsetsInPixels[neuralLod];
}

}