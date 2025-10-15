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

#include "CudaRandomGen.h"
#include "CudaUtils.h"
#include <ctime>
#include <cooperative_groups.h>

namespace ntc
{

void CudaRandomGen::RandomizeSeed()
{
    m_seed = uint32_t(time(nullptr));
}

__global__ void FillRandomUintKernel(uint32_t* buffer, uint32_t length, uint32_t seed)
{
    const uint32_t globalIdx = cooperative_groups::this_grid().thread_rank();

    if (globalIdx >= length)
        return;

    HashBasedRNG rng(globalIdx, seed);
    buffer[globalIdx] = rng.NextUint();
}

void CudaRandomGen::FillRandomUint(uint32_t* buffer, uint32_t length)
{
    uint32_t threadBlockSize = 128;
    uint32_t gridSize = (length + threadBlockSize - 1) / threadBlockSize;

    FillRandomUintKernel <<< gridSize, threadBlockSize >>> (buffer, length, m_seed);

    ++m_seed;
}

__global__ void FillRandomNormalHalfKernel(
    half* buffer,
    uint32_t length,
    uint32_t seed,
    float scale,
    float bias,
    float min_value,
    float max_value)
{
    const uint32_t globalIdx = cooperative_groups::this_grid().thread_rank();
    const uint32_t firstElement = globalIdx * 2;

    if (firstElement >= length)
        return;

    HashBasedRNG rng(globalIdx, seed);

    // Generate normal distributed numbers (x, y) using the Box-Muller transform.
    // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform

    const float u = rng.NextFloat();
    const float v = rng.NextFloat();

    const float r = sqrtf(std::max(0.f, -2.f * logf(std::max(u, 1e-20f)))); // Clamp u to avoid Inf or NaN outputs
    const float theta = 2.f * float(M_PI) * v;

    float x = r * cosf(theta);
    float y = r * sinf(theta);

    // Transform x and y to the specified distribution parameters
    x = std::min(std::max(x * scale + bias, min_value), max_value);
    y = std::min(std::max(y * scale + bias, min_value), max_value);

    if (isinf(x) || isnan(x)) x = 0;
    if (isinf(y) || isnan(y)) y = 0;
    
    buffer[firstElement] = x;

    if (firstElement + 1 < length)
        buffer[firstElement + 1] = y;
}

void CudaRandomGen::FillRandomNormalHalf(
    half* buffer,
    uint32_t length,
    float scale,
    float bias,
    float min_value,
    float max_value)
{
    uint32_t threadBlockSize = 128;
    uint32_t gridSize = DivRoundUp(length, threadBlockSize);
    FillRandomNormalHalfKernel <<< gridSize, threadBlockSize >>> (buffer, length, m_seed, scale, bias, min_value, max_value);

    ++m_seed;
}

}