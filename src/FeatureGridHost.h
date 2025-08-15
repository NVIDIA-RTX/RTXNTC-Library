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

    Status Initialize(int imageWidth, int imageHeight, int imageMips, int highResGridScale, int highResFeatures, int lowResFeatures,
        int highResQuantBits, int lowResQuantBits, bool enableCompression);
    
    void Deallocate();
    
    void Fill(CudaRandomGen& rng);

    void ClearGradientMask();

    int LodToNeuralLod(int lod) const;
    
    half* GetBaseLatentsDevicePtr(Grid grid, int neuralLod);

    half* GetQuantizedLatentsDevicePtr(Grid grid, int neuralLod);

    float* GetMoment1DevicePtr(Grid grid, int neuralLod);

    float* GetMoment2DevicePtr(Grid grid, int neuralLod);

    template<typename TGrad>
    TGrad* GetGradientsDevicePtr(Grid grid, int neuralLod)
    {
        return m_gradientMemory.DevicePtr() ? (TGrad*)m_gradientMemory.DevicePtr() + GetLatentOffset(grid, neuralLod) : nullptr;
    }

    uint32_t* GetEncodedLatentsDevicePtr(Grid grid, int neuralLod);

    uint32_t* GetEncodedLatentsHostPtr(Grid grid, int neuralLod);

    size_t GetQuantizedLatentsSize(Grid grid, int neuralLod);

    uint32_t* GetGradientMaskDevicePtr(Grid grid, int neuralLod);

    DeviceAndHostArray<uint32_t>& GetEncodedLatentsArray();

    size_t GetLatentOffset(Grid grid, int neuralLod);

    size_t GetLatentCount(Grid grid, int neuralLod);

    size_t GetLatentCount(Grid grid) const;

    int GetNumMipLevels() const;

private:
    
    int m_highResFeatures = 0;
    int m_lowResFeatures = 0;
    int m_highResGridScale = 0;
    int m_highResQuantBits = 0;
    int m_lowResQuantBits = 0;
    std::array<size_t, NTC_MAX_MIPS> m_highResLatentCounts {};
    std::array<size_t, NTC_MAX_MIPS> m_lowResLatentCounts {};
    std::array<size_t, NTC_MAX_MIPS> m_highResLatentOffsets {};
    std::array<size_t, NTC_MAX_MIPS> m_lowResLatentOffsets {};
    std::array<size_t, NTC_MAX_MIPS> m_highResQuantizedGridOffsets {};
    std::array<size_t, NTC_MAX_MIPS> m_lowResQuantizedGridOffsets {};
    std::array<size_t, NTC_MAX_MIPS> m_highResMaskOffsets {};
    std::array<size_t, NTC_MAX_MIPS> m_lowResMaskOffsets {};
    size_t m_totalHighResLatentCount = 0;
    size_t m_totalLowResLatentCount = 0;
    size_t m_totalLatentCount = 0;
    int m_numNeuralMipLevels = 0;
    
    DeviceArray<half> m_baseLatentsMemory;
    DeviceArray<half> m_quantizedLatentsMemory;
    DeviceAndHostArray<uint32_t> m_encodedLatentsMemory;
    DeviceArray<uint32_t> m_gradientMemory; // declared as uint32_t, used as either float or half depending on 'stableTraining'
    DeviceArray<float> m_moment1Memory;
    DeviceArray<float> m_moment2Memory;
    DeviceArray<uint32_t> m_gradientMaskMemory;
};

}