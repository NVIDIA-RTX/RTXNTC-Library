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

#include "FeatureGridMath.h"
#include "tin/tin_matrix.h"
#include <cuda_fp16.hpp>

namespace ntc::cuda
{

class FeatureGrid
{
public:
    __device__ FeatureGrid(int numFeatures, int width, int height, size_t latentStride)
        : m_latentWidth(width)
        , m_latentHeight(height)
        , m_numFeatures(numFeatures)
        , m_latentStride(latentStride)
    {
    }

    __device__ void SetupBilinearFilter(float u, float v, int& x0, int& y0, int& x1, int& y1, half weights[4])
    {
        float x = u * m_latentWidth  - 0.5f;
        float y = v * m_latentHeight - 0.5f;

        int x_b = floor(x);
        int y_b = floor(y);

        float dx = x - x_b;
        float dy = y - y_b;
        float dxn = 1.f - dx;
        float dyn = 1.f - dy;

        weights[0] = half(dxn * dyn);
        weights[1] = half(dx * dyn);
        weights[2] = half(dxn * dy);
        weights[3] = half(dx * dy);

        // Wrap addressing
        if (x_b < 0) x_b += m_latentWidth;
        if (y_b < 0) y_b += m_latentHeight;
        x0 = x_b % m_latentWidth;
        y0 = y_b % m_latentHeight;
        x1 = (x_b + 1) % m_latentWidth;
        y1 = (y_b + 1) % m_latentHeight;
    }

    __device__ void Sample(float u, float v, const half* __restrict__ features,
        tin::HArray<NTC_MLP_INPUT_CHANNELS>& outputArray, int arrayOffset)
    {
        int x0, y0, x1, y1;
        half weights[4];
        SetupBilinearFilter(u, v, x0, y0, x1, y1, weights);

        size_t a00 = (size_t(y0) * size_t(m_latentWidth) + size_t(x0)) * FeatureGridMath::FeaturesPerGroup;
        size_t a01 = (size_t(y0) * size_t(m_latentWidth) + size_t(x1)) * FeatureGridMath::FeaturesPerGroup;
        size_t a10 = (size_t(y1) * size_t(m_latentWidth) + size_t(x0)) * FeatureGridMath::FeaturesPerGroup;
        size_t a11 = (size_t(y1) * size_t(m_latentWidth) + size_t(x1)) * FeatureGridMath::FeaturesPerGroup;
        
        #pragma unroll
        for (int featureIndex = 0; featureIndex < NTC_MLP_FEATURES; featureIndex += 2)
        {
            if (featureIndex >= m_numFeatures)
                break;

            half2 x00 = *(half2*)(features + a00);
            half2 x01 = *(half2*)(features + a01);
            half2 x10 = *(half2*)(features + a10);
            half2 x11 = *(half2*)(features + a11);

            half2 d;
            d.x = x00.x * weights[0] + x01.x * weights[1] + x10.x * weights[2] + x11.x * weights[3];
            d.y = x00.y * weights[0] + x01.y * weights[1] + x10.y * weights[2] + x11.y * weights[3];

            // Convert from [0,1] to [-1,1], that works better as a network input
            d.x = d.x * half(2.0f) - half(1.0f);
            d.y = d.y * half(2.0f) - half(1.0f);
            
            outputArray.set_packed_element(d, (arrayOffset + featureIndex) / 2);

            a00 += m_latentStride;
            a01 += m_latentStride;
            a10 += m_latentStride;
            a11 += m_latentStride;
        }
    }

    __device__ void MarkGradientMask(int x, int y, uint32_t* gradientMask, size_t maskOffsetInBits)
    {
        size_t bitIndex = y * m_latentWidth + x + maskOffsetInBits;
        size_t wordIndex = bitIndex >> 5;
        uint32_t wordMask = 1u << (bitIndex & 31);

        atomicOr(gradientMask + wordIndex, wordMask);
    }

    template<typename GRID_GRAD_TYPE>
    __device__ void SampleBackward(float u, float v, 
        const tin::HArray<NTC_MLP_INPUT_CHANNELS>& outputGradients, int arrayOffset,
        GRID_GRAD_TYPE* __restrict__ gradients, uint32_t* gradientMask, size_t maskOffsetInBits)
    {
        int x0, y0, x1, y1;
        half weights[4];
        SetupBilinearFilter(u, v, x0, y0, x1, y1, weights);

        MarkGradientMask(x0, y0, gradientMask, maskOffsetInBits);
        MarkGradientMask(x1, y0, gradientMask, maskOffsetInBits);
        MarkGradientMask(x0, y1, gradientMask, maskOffsetInBits);
        MarkGradientMask(x1, y1, gradientMask, maskOffsetInBits);
        
        size_t a00 = (size_t(y0) * size_t(m_latentWidth) + size_t(x0)) * FeatureGridMath::FeaturesPerGroup;
        size_t a01 = (size_t(y0) * size_t(m_latentWidth) + size_t(x1)) * FeatureGridMath::FeaturesPerGroup;
        size_t a10 = (size_t(y1) * size_t(m_latentWidth) + size_t(x0)) * FeatureGridMath::FeaturesPerGroup;
        size_t a11 = (size_t(y1) * size_t(m_latentWidth) + size_t(x1)) * FeatureGridMath::FeaturesPerGroup;

        #pragma unroll
        for (int featureIndex = 0; featureIndex < NTC_MLP_FEATURES; featureIndex += 2)
        {
            if (featureIndex >= m_numFeatures)
                break;

            half2 outputGrad = outputGradients.get_packed_element((arrayOffset + featureIndex) / 2);
            outputGrad.x *= half(2.0f);
            outputGrad.y *= half(2.0f);
            
            if (std::is_same<GRID_GRAD_TYPE, float>::value)
            {
                tin::_atomic_addf((float*)&gradients[a00 + 0], float(outputGrad.x * weights[0]));
                tin::_atomic_addf((float*)&gradients[a00 + 1], float(outputGrad.y * weights[0]));
                tin::_atomic_addf((float*)&gradients[a01 + 0], float(outputGrad.x * weights[1]));
                tin::_atomic_addf((float*)&gradients[a01 + 1], float(outputGrad.y * weights[1]));
                tin::_atomic_addf((float*)&gradients[a10 + 0], float(outputGrad.x * weights[2]));
                tin::_atomic_addf((float*)&gradients[a10 + 1], float(outputGrad.y * weights[2]));
                tin::_atomic_addf((float*)&gradients[a11 + 0], float(outputGrad.x * weights[3]));
                tin::_atomic_addf((float*)&gradients[a11 + 1], float(outputGrad.y * weights[3]));
            }
            else
            {
                tin::_atomic_addh2((half2*)&gradients[a00], half2{outputGrad.x * weights[0], outputGrad.y * weights[0]});
                tin::_atomic_addh2((half2*)&gradients[a01], half2{outputGrad.x * weights[1], outputGrad.y * weights[1]});
                tin::_atomic_addh2((half2*)&gradients[a10], half2{outputGrad.x * weights[2], outputGrad.y * weights[2]});
                tin::_atomic_addh2((half2*)&gradients[a11], half2{outputGrad.x * weights[3], outputGrad.y * weights[3]});
            }

            a00 += m_latentStride;
            a01 += m_latentStride;
            a10 += m_latentStride;
            a11 += m_latentStride;
        }
    }

private:
    int m_latentWidth;
    int m_latentHeight;
    int m_numFeatures;
    size_t m_latentStride;
};

} // namespace ntc::cuda