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
#include <random>

namespace ntc
{

FeatureGrid::FeatureGrid(IAllocator* allocator)
    : m_encodedLatentsMemory(allocator)
{ }

Status FeatureGrid::Initialize(int imageWidth, int imageHeight, int imageMips, int highResGridScale, int highResFeatures, int lowResFeatures,
    int highResQuantBits, int lowResQuantBits, bool enableCompression)
{
    m_highResGridScale = highResGridScale;
    m_highResQuantBits = highResQuantBits;
    m_lowResQuantBits = lowResQuantBits;
    m_highResFeatures = highResFeatures;
    m_lowResFeatures = lowResFeatures;

    m_numNeuralMipLevels = CalculateNumNeuralMipLevels(imageWidth, imageHeight, highResGridScale);

    // Clamp the theoretical mip count with the one that will be used for this texture
    const int lastNeuralMip = FeatureGridMath::LodToNeuralLod(imageMips - 1, highResGridScale, NTC_MAX_NEURAL_MIPS);
    m_numNeuralMipLevels = std::min(m_numNeuralMipLevels, lastNeuralMip + 1);

    m_totalHighResLatentCount = 0;
    m_totalLowResLatentCount = 0;
    m_totalLatentCount = 0;
    size_t quantizedLatentUintCount = 0;
    size_t totalMaskItemCount = 0;

    for (int mip = 0; mip < m_numNeuralMipLevels; mip++)
    {
        size_t const latentPixelsInThisMip = CalculateNumLatentsInNeuralMip(Grid::HighRes, imageWidth, imageHeight,
            highResGridScale, mip);
        size_t const latentCountInThisMip = latentPixelsInThisMip * highResFeatures;

        m_highResLatentCounts[mip] = latentCountInThisMip;
        m_highResLatentOffsets[mip] = m_totalLatentCount;
        m_totalHighResLatentCount += latentCountInThisMip;
        m_totalLatentCount += latentCountInThisMip;

        m_highResMaskOffsets[mip] = totalMaskItemCount;
        totalMaskItemCount += (latentPixelsInThisMip + 31) / 32;

        int quantizedSizeInThisMip = GetQuantizedLatentSizeUints(latentCountInThisMip, m_highResQuantBits);
        m_highResQuantizedGridOffsets[mip] = quantizedLatentUintCount;
        quantizedLatentUintCount += quantizedSizeInThisMip;
    }

    for (int mip = 0; mip < m_numNeuralMipLevels; mip++)
    {
        size_t const latentPixelsInThisMip = CalculateNumLatentsInNeuralMip(Grid::LowRes, imageWidth, imageHeight,
            highResGridScale, mip);
        size_t const latentCountInThisMip = latentPixelsInThisMip * lowResFeatures;

        m_lowResLatentCounts[mip] = latentCountInThisMip;
        m_lowResLatentOffsets[mip] = m_totalLatentCount;
        m_totalLowResLatentCount += latentCountInThisMip;
        m_totalLatentCount += latentCountInThisMip;

        m_lowResMaskOffsets[mip] = totalMaskItemCount;
        totalMaskItemCount += (latentPixelsInThisMip + 31) / 32;

        int quantizedSizeInThisMip = GetQuantizedLatentSizeUints(latentCountInThisMip, m_lowResQuantBits);
        m_lowResQuantizedGridOffsets[mip] = quantizedLatentUintCount;
        quantizedLatentUintCount += quantizedSizeInThisMip;
    }

    if (!m_quantizedLatentsMemory.Allocate(m_totalLatentCount))
        return Status::OutOfMemory;

    if (!m_encodedLatentsMemory.Allocate(quantizedLatentUintCount))
        return Status::OutOfMemory;

    if (enableCompression)
    {
        if (!m_baseLatentsMemory.Allocate(m_totalLatentCount))  return Status::OutOfMemory;
        if (!m_gradientMemory.Allocate(m_totalLatentCount))     return Status::OutOfMemory;
        if (!m_moment1Memory.Allocate(m_totalLatentCount))      return Status::OutOfMemory;
        if (!m_moment2Memory.Allocate(m_totalLatentCount))      return Status::OutOfMemory;
        if (!m_gradientMaskMemory.Allocate(totalMaskItemCount)) return Status::OutOfMemory;
    }

    return Status::Ok;
}

void FeatureGrid::Deallocate()
{
    m_quantizedLatentsMemory.Deallocate();
    m_encodedLatentsMemory.Deallocate();
    m_baseLatentsMemory.Deallocate();
    m_gradientMemory.Deallocate();
    m_moment1Memory.Deallocate();
    m_moment2Memory.Deallocate();
    m_gradientMaskMemory.Deallocate();
}

void FeatureGrid::Fill(CudaRandomGen& rng)
{
    const float threshold = 1.f;
    const float scale = 0.5f / sqrtf(float(m_lowResFeatures));

    rng.FillRandomNormalHalf(m_baseLatentsMemory.DevicePtr(),
        uint32_t(m_baseLatentsMemory.Length()),
        scale, 0.f, -threshold, threshold);

    cudaMemcpy(m_quantizedLatentsMemory.DevicePtr(), m_baseLatentsMemory.DevicePtr(), m_baseLatentsMemory.Size(), cudaMemcpyDeviceToDevice);
    cudaMemset(m_gradientMemory.DevicePtr(), 0, m_totalLatentCount * sizeof(float));
    cudaMemset(m_moment1Memory.DevicePtr(), 0, m_totalLatentCount * sizeof(float));
    cudaMemset(m_moment2Memory.DevicePtr(), 0, m_totalLatentCount * sizeof(float));
}

void FeatureGrid::ClearGradientMask()
{
    cudaMemset(m_gradientMaskMemory.DevicePtr(), 0, m_gradientMaskMemory.Size());
}

int FeatureGrid::LodToNeuralLod(int lod) const
{
    return FeatureGridMath::LodToNeuralLod(lod, m_highResGridScale, m_numNeuralMipLevels);
}

half* FeatureGrid::GetBaseLatentsDevicePtr(Grid grid, int neuralLod)
{
    return m_baseLatentsMemory.DevicePtr() ? m_baseLatentsMemory.DevicePtr() + GetLatentOffset(grid, neuralLod) : nullptr;
}

half* FeatureGrid::GetQuantizedLatentsDevicePtr(Grid grid, int neuralLod)
{
    return m_quantizedLatentsMemory.DevicePtr() ? m_quantizedLatentsMemory.DevicePtr() + GetLatentOffset(grid, neuralLod) : nullptr;
}

float* FeatureGrid::GetMoment1DevicePtr(Grid grid, int neuralLod)
{
    return m_moment1Memory.DevicePtr() ? m_moment1Memory.DevicePtr() + GetLatentOffset(grid, neuralLod) : nullptr;
}

float* FeatureGrid::GetMoment2DevicePtr(Grid grid, int neuralLod)
{
    return m_moment2Memory.DevicePtr() ? m_moment2Memory.DevicePtr() + GetLatentOffset(grid, neuralLod) : nullptr;
}

uint32_t* FeatureGrid::GetEncodedLatentsDevicePtr(Grid grid, int neuralLod)
{
    size_t offset = grid == Grid::HighRes ? m_highResQuantizedGridOffsets[neuralLod] : m_lowResQuantizedGridOffsets[neuralLod];
    return m_encodedLatentsMemory.DevicePtr() ? m_encodedLatentsMemory.DevicePtr() + offset : nullptr;
}

uint32_t* FeatureGrid::GetEncodedLatentsHostPtr(Grid grid, int neuralLod)
{
    size_t offset = grid == Grid::HighRes ? m_highResQuantizedGridOffsets[neuralLod] : m_lowResQuantizedGridOffsets[neuralLod];
    return m_encodedLatentsMemory.HostPtr() ? m_encodedLatentsMemory.HostPtr() + offset : nullptr;
}

size_t FeatureGrid::GetQuantizedLatentsSize(Grid grid, int neuralLod)
{
    size_t count = grid == Grid::HighRes ? m_highResLatentCounts[neuralLod] : m_lowResLatentCounts[neuralLod];
    int bits = grid == Grid::HighRes ? m_highResQuantBits : m_lowResQuantBits;
    return GetQuantizedLatentSizeUints(count, bits) * sizeof(uint32_t);
}

uint32_t* FeatureGrid::GetGradientMaskDevicePtr(Grid grid, int neuralLod)
{
    size_t const offset = (grid == Grid::HighRes) ? m_highResMaskOffsets[neuralLod] : m_lowResMaskOffsets[neuralLod];
    return m_gradientMaskMemory.DevicePtr() ? m_gradientMaskMemory.DevicePtr() + offset : nullptr;
}

DeviceAndHostArray<uint32_t>& FeatureGrid::GetEncodedLatentsArray()
{
    return m_encodedLatentsMemory;
}

size_t FeatureGrid::GetLatentOffset(Grid grid, int neuralLod)
{
    return grid == Grid::HighRes ? m_highResLatentOffsets[neuralLod] : m_lowResLatentOffsets[neuralLod];
}

size_t FeatureGrid::GetLatentCount(Grid grid, int neuralLod)
{
    return grid == Grid::HighRes ? m_highResLatentCounts[neuralLod] : m_lowResLatentCounts[neuralLod];
}

size_t FeatureGrid::GetLatentCount(Grid grid) const
{
    return grid == Grid::HighRes ? m_totalHighResLatentCount : m_totalLowResLatentCount;
}

int FeatureGrid::GetNumMipLevels() const 
{
    return m_numNeuralMipLevels;
}


}