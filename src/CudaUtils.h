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

#include <cuda_fp16.h>
#include <cstdint>

inline __device__ uint32_t GetBit(uint32_t mask, uint32_t bit)
{
    return (mask >> bit) & 1;
}

inline __device__ bool IsFloatSpecial(float f)
{
    uint32_t u = __float_as_uint(f);
    // Test if the number is an IEEE 754 Inf or NaN pattern (all 1's exponent)
    return (u & 0x7f800000) == 0x7f800000;
}

inline __device__ bool IsHalfSpecial(half f)
{
    uint16_t u = __half_as_ushort(f);
    // Test if the number is an IEEE 754 Inf or NaN pattern (all 1's exponent)
    return (u & 0x7c00) == 0x7c00;
}

inline __device__ float SimpleRandomGenerator(uint32_t& seed)
{
    seed = seed * 1664525 + 1013904223;
    return float(seed >> 8) * 0x1p-24f;
}

inline __device__ float frac(float x)
{
    return x - floorf(x);
}

inline __device__ __host__ uint32_t DivRoundUp(size_t i, uint32_t multiple)
{
    return uint32_t((i + (multiple - 1)) / multiple);
}

inline __device__ __host__ dim3 DivRoundUp(dim3 i, dim3 multiple)
{
    return dim3(
        DivRoundUp(i.x, multiple.x),
        DivRoundUp(i.y, multiple.y),
        DivRoundUp(i.z, multiple.z));
}

// Hybrid Log-Gamma encoding and decoding functions.
// https://en.wikipedia.org/wiki/Hybrid_log%E2%80%93gamma
class HLG
{
public:
    static inline __device__ float Encode(float e)
    {
        float sign = (e < 0.f) ? -1.f : 1.f;
        e = fabsf(e);
        return (e < 1.f)
            ? 0.5f * sqrtf(e) * sign
            : (a * logf(e - b) + c) * sign;
    }

    static inline __device__ float Decode(float h)
    {
        float sign = (h < 0.f) ? -1.f : 1.f;
        h = fabsf(h);
        return (h < 0.5f)
            ? 4.f * h * h * sign
            : (expf((h - c) / a) + b) * sign;
    }

private:
    static constexpr float a = 0.17883277f;
    static constexpr float b = 0.28466892f;
    static constexpr float c = 0.55991073f;
};

// sRGB encoding and decoding functions.
class sRGB
{
public:
    static inline __device__ float Encode(float linear)
    {
        return (linear <= 0.0031308f)
            ? linear * 12.92f
            : 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
    }

    static inline __device__ float Decode(float encoded)
    {
        return (encoded <= 0.04045f)
            ? encoded / 12.92f
            : powf((encoded + 0.055f) / 1.055f, 2.4f);
    }
};


class HashBasedRNG
{
public:
    inline __device__ HashBasedRNG(uint32_t linearIndex, uint32_t offset)
    {
        m_index = 1;
        m_seed = JenkinsHash(linearIndex) + offset;
    }

    inline __device__ uint32_t NextUint()
    {
        uint32_t v = Murmur3Hash();
        ++m_index;
        return v;
    }

    inline __device__ float NextFloat()
    {
        uint32_t v = Murmur3Hash();
        ++m_index;
        const uint32_t one = __float_as_uint(1.f);
        const uint32_t mask = (1 << 23) - 1;
        return __uint_as_float((mask & v) | one) - 1.f;
    }

private:

    uint32_t m_seed;
    uint32_t m_index;

    static inline __device__ uint32_t JenkinsHash(uint32_t a)
    {
        // http://burtleburtle.net/bob/hash/integer.html
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }

    inline __device__ uint32_t Murmur3Hash()
    {
#define ROT32(x, y) ((x << y) | (x >> (32 - y)))

        // https://en.wikipedia.org/wiki/MurmurHash
        uint32_t c1 = 0xcc9e2d51;
        uint32_t c2 = 0x1b873593;
        uint32_t r1 = 15;
        uint32_t r2 = 13;
        uint32_t m = 5;
        uint32_t n = 0xe6546b64;

        uint32_t hash = m_seed;
        uint32_t k = m_index;
        k *= c1;
        k = ROT32(k, r1);
        k *= c2;

        hash ^= k;
        hash = ROT32(hash, r2) * m + n;

        hash ^= 4;
        hash ^= (hash >> 16);
        hash *= 0x85ebca6b;
        hash ^= (hash >> 13);
        hash *= 0xc2b2ae35;
        hash ^= (hash >> 16);

#undef ROT32

        return hash;
    }
};
