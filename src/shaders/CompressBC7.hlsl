/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#define OUTPUT_FORMAT uint4
#include "BlockCompressCommon.hlsli"

#define BC7_SINGLE_MODE (USE_MODE_BUFFER)

static const uint NO_PARTITIONS = 0xffff;
static const int ENDPOINT_ITERATIONS = 2;

// There are other options to store arrays of endpoints, like using 'half' or packing into a uint.
// But those options do not improve performance by any significant amount in most cases.
typedef float3 StoredEndpoint3;
typedef float4 StoredEndpoint4;

// The c_Partitions#_Subset# arrays are bitmasks defining which pixels in a block
// belong to a specific subset in each partitioning scheme.
// These arrays are derived from the BC7 spec, Table.P2 and Table.P3:
// https://registry.khronos.org/OpenGL/extensions/ARB/ARB_texture_compression_bptc.txt
// See tools/gen_bc7_tables.py

static const uint c_Partitions2_Subset0[64] = {
    0x3333, 0x7777, 0x1111, 0x1337, 0x377f, 0x0113, 0x0137, 0x137f,
    0x37ff, 0x0013, 0x017f, 0x17ff, 0x0017, 0x00ff, 0x000f, 0x0fff,
    0x08ef, 0xff71, 0x8eff, 0xf731, 0xff73, 0x8cef, 0xceff, 0x7331,
    0xf773, 0xceef, 0x9999, 0xc993, 0xe817, 0xf00f, 0x8e71, 0xc663,
    0x5555, 0x0f0f, 0xa5a5, 0xcc33, 0xc3c3, 0xaa55, 0x6969, 0x5aa5,
    0x8c31, 0xec37, 0xcdb3, 0xc423, 0x9669, 0x3cc3, 0x6699, 0xf99f,
    0xfd8d, 0xfb1b, 0xb1bf, 0xd8df, 0x36c9, 0x6c93, 0xc639, 0x9c63,
    0x6cc9, 0x6339, 0x7e81, 0x18e7, 0x330f, 0xf033, 0x88bb, 0x11dd
};

static const uint c_Partitions2_Subset1[64] = {
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80,
    0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
    0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
    0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22
};

static const uint c_Partitions3_Subset0[64] = {
    0x0133, 0x0037, 0x006f, 0x1331, 0x00ff, 0x3333, 0x0033, 0x0033,
    0x00ff, 0x000f, 0x000f, 0x3333, 0x1111, 0x1111, 0x0013, 0x8c63,
    0x0137, 0xc631, 0x000f, 0x0333, 0x1111, 0x0077, 0x113f, 0x88cf,
    0xf311, 0x0033, 0x9009, 0x009f, 0x3443, 0x0699, 0x3113, 0x00ef,
    0x007f, 0x3331, 0x1333, 0x9999, 0xf00f, 0x9249, 0x9429, 0x30c3,
    0x3c03, 0x0055, 0x00ff, 0x0303, 0x3333, 0x0909, 0x5005, 0x000f,
    0x0555, 0x1111, 0x0707, 0x000f, 0x1111, 0x7007, 0x0999, 0x00ff,
    0x0099, 0x3333, 0x3003, 0x0fff, 0x7777, 0x0101, 0x0005, 0x8421
};

static const uint c_Partitions3_Subset1[64] = {
    0x08cc, 0x8cc8, 0xcc80, 0xec00, 0x3300, 0x00cc, 0xff00, 0xcccc,
    0x0f00, 0x0ff0, 0x00f0, 0x4444, 0x6666, 0x2222, 0x136c, 0x008c,
    0x36c8, 0x08ce, 0x3330, 0xf000, 0x00ee, 0x8888, 0x22c0, 0x4430,
    0x0c22, 0x0344, 0x6996, 0x9960, 0x0330, 0x0066, 0xc22c, 0x8c00,
    0x1300, 0xc400, 0x004c, 0x2222, 0x00f0, 0x2492, 0x2942, 0xc30c,
    0xc03c, 0x00aa, 0xaa00, 0x3030, 0xc0c0, 0x9090, 0xa00a, 0xaaa0,
    0x0aaa, 0xe0e0, 0x7070, 0x6660, 0x0ee0, 0x0770, 0x0666, 0x6600,
    0x0066, 0x0cc0, 0x0330, 0x6000, 0x8080, 0x1010, 0x000a, 0x08ce
};

static const uint c_Partitions3_Subset2[64] = {
    0xf600, 0x7300, 0x3310, 0x00ce, 0xcc00, 0xcc00, 0x00cc, 0x3300,
    0xf000, 0xf000, 0xff00, 0x8888, 0x8888, 0xcccc, 0xec80, 0x7310,
    0xc800, 0x3100, 0xccc0, 0x0ccc, 0xee00, 0x7700, 0xcc00, 0x3300,
    0x00cc, 0xfc88, 0x0660, 0x6600, 0xc88c, 0xf900, 0x0cc0, 0x7310,
    0xec80, 0x08ce, 0xec80, 0x4444, 0x0f00, 0x4924, 0x4294, 0x0c30,
    0x03c0, 0xff00, 0x5500, 0xcccc, 0x0c0c, 0x6666, 0x0ff0, 0x5550,
    0xf000, 0x0e0e, 0x8888, 0x9990, 0xe00e, 0x8888, 0xf000, 0x9900,
    0xff00, 0xc00c, 0xcccc, 0x9000, 0x0808, 0xeeee, 0xfff0, 0x7310
};

// The c_Partitions#_Anchor# arrays specify the index of the anchor pixel
// for a specific subset in each partitioning scheme.
// They are direct copies from Tables A2, A3a, A3b in the spec referenced above.

static const uint c_Partitions2_Anchor1[64] = {
    15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,
    15, 2, 8, 2, 2, 8, 8,15,
     2, 8, 2, 2, 8, 8, 2, 2,
    15,15, 6, 8, 2, 8,15,15,
     2, 8, 2, 2, 2,15,15, 6,
     6, 2, 6, 8,15,15, 2, 2,
    15,15,15,15,15, 2, 2,15
};

static const uint c_Partitions3_Anchor1[64] = {
     3, 3,15,15, 8, 3,15,15,
     8, 8, 6, 6, 6, 5, 3, 3,
     3, 3, 8,15, 3, 3, 6,10,
     5, 8, 8, 6, 8, 5,15,15,
     8,15, 3, 5, 6,10, 8,15,
    15, 3,15, 5,15,15,15,15,
     3,15, 5, 5, 5, 8, 5,10,
     5,10, 8,13,15,12, 3, 3,
};

static const uint c_Partitions3_Anchor2[64] = {
    15, 8, 8, 3,15,15, 3, 8,
    15,15,15,15,15,15,15, 8,
    15, 8,15, 3,15, 8,15, 8,
     3,15, 6,10,15,15,10, 8,
    15, 3,15,10,10, 8, 9,10,
     6,15, 8,15, 3, 6, 6, 8,
    15, 3,15,15,15,15,15,15,
    15,15,15,15, 3,15,15, 8
};


bool GetBit64(uint2 words, int index)
{
    uint w = index >= 32 ? words.y : words.x;
    return ((w >> (index & 31)) & 1) != 0;
}

int Quantize(float value, int prec)
{
    float scale = (1u << prec) - 1;
    return int(round(saturate(value) * scale));
}

int3 Quantize(float3 value, int prec)
{
    return int3(
        Quantize(value.x, prec),
        Quantize(value.y, prec),
        Quantize(value.z, prec));
}

int4 Quantize(float4 value, int prec)
{
    return int4(
        Quantize(value.x, prec),
        Quantize(value.y, prec),
        Quantize(value.z, prec),
        Quantize(value.w, prec));
}

int ExpandTo8Bits(int q, int bits)
{
    return (q << (8 - bits)) | (q >> (2 * bits - 8));
}

template<int C>
vector<int,C> BC7_ExpandEndpointTo8Bits(vector<int,C> ep, int endpointBits)
{
    for (int ch = 0; ch < C; ++ch)
    {
        ep[ch] = ExpandTo8Bits(ep[ch], endpointBits);
    }
    return ep;
}

template<int C>
vector<float,C> BC7_DequantizeEndpoint(vector<int,C> ep, int endpointBits)
{
    return vector<float, C>(BC7_ExpandEndpointTo8Bits(ep, endpointBits)) / 255.0;
}

template<typename Tpix>
void BC7_MinMax(const Tpix pixels[PIXELS_PER_BLOCK], uint partitionMask, out Tpix mins, out Tpix maxs)
{
    mins = FLT_MAX;
    maxs = -FLT_MAX;
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        if ((partitionMask & (1u << idx)) != 0)
        {
            mins = min(mins, pixels[idx]);
            maxs = max(maxs, pixels[idx]);
        }
    }
}

template<typename Tpix>
void BC7_OptimizeEndpoints(const Tpix pixels[PIXELS_PER_BLOCK], uint partitionMask,
    inout Tpix ep0, inout Tpix ep1, int indexBits)
{
    [loop]
    for (int iter = 0; iter < ENDPOINT_ITERATIONS; ++iter)
    {
        Tpix axis = ep1 - ep0;
        float mul = dot(axis, axis);
        if (mul > 0.0f) mul = 1.0 / mul;

        const float scale = (1u << indexBits) - 1;
        const float iscale = 1.0 / scale;

        float alpha2_sum = 0;
        float beta2_sum = 0;
        float alphabeta_sum = 0;
        Tpix alphax_sum = 0;
        Tpix betax_sum = 0;

        [unroll]
        for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
        {
            if ((partitionMask & (1u << idx)) != 0)
            {
                Tpix color = pixels[idx];
                float f_ind = saturate(dot((color - ep0), axis) * mul);
                float beta = round(f_ind * scale) * iscale;
                float alpha = 1.0 - beta;

                alpha2_sum += alpha * alpha;
                beta2_sum += beta * beta;
                alphabeta_sum += alpha * beta;
                alphax_sum += alpha * color;
                betax_sum += beta * color;
            }
        }

        float denom = alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum;

        if (abs(denom) > 0.0001f)
        {
            float factor = 1.0f / denom;
            Tpix a = saturate((alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor);
            Tpix b = saturate((betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor);

            // Preserve 0 and 1 cases
            ep0 = select(or(ep0 == 0.0, ep0 == 1.0), ep0, a);
            ep1 = select(or(ep1 == 0.0, ep1 == 1.0), ep1, b);
        }
        else
            break;
    }
}

// Determines if some of the RGB components of the two endpoints need to be swapped (hi <-> lo)
// to better fit the colors on a line, in case there is negative covariance between color components.
void BC7_SwapEndpointComponents_RGB(const float3 pixels[PIXELS_PER_BLOCK], uint partitionMask,
    inout float3 ep0, inout float3 ep1)
{
    // Compute the average color in the partition.
    float3 sum_color = 0;
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        if ((partitionMask & (1u << idx)) != 0)
        {
            sum_color += pixels[idx];   
        }
    }
    float3 center = sum_color / countbits(partitionMask);

    // Compute the matrix of covariances.
    float cov_rg = 0, cov_rb = 0;
    float cov_gb = 0;

    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        if ((partitionMask & (1u << idx)) != 0)
        {
            const float3 pix = pixels[idx] - center;
            cov_rg += pix.r * pix.g;
            cov_rb += pix.r * pix.b;
            cov_gb += pix.g * pix.b;
        }
    }

    // Pick one axis with nonzero extent as the reference, and swap the endpoints
    // on the other axes if the covariance between them and the reference axis is negative.
    if (ep0.r != ep1.r)
    {
        if (cov_rg < 0) swap(ep0.g, ep1.g);
        if (cov_rb < 0) swap(ep0.b, ep1.b);
    }
    else if (ep0.g != ep1.g)
    {
        if (cov_gb < 0) swap(ep0.b, ep1.b);
    }
}

// A version of BC7_SwapEndpointComponents_RGB but for 4 color channels.
void BC7_SwapEndpointComponents_RGBA(const float4 pixels[PIXELS_PER_BLOCK], uint partitionMask,
    inout float4 ep0, inout float4 ep1)
{
    float4 sum_color = 0;
    
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        if ((partitionMask & (1u << idx)) != 0)
        {
            sum_color += pixels[idx];
        }
    }

    float4 center = sum_color / countbits(partitionMask);

    float cov_rg = 0, cov_rb = 0, cov_ra = 0;
    float cov_gb = 0, cov_ga = 0;
    float cov_ba = 0;

    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        if ((partitionMask & (1u << idx)) != 0)
        {
            const float4 pix = pixels[idx] - center;
            cov_rg += pix.r * pix.g;
            cov_rb += pix.r * pix.b;
            cov_ra += pix.r * pix.a;
            cov_gb += pix.g * pix.b;
            cov_ga += pix.g * pix.a;
            cov_ba += pix.b * pix.a;
        }
    }

    if (ep0.r != ep1.r)
    {
        if (cov_rg < 0) swap(ep0.g, ep1.g);
        if (cov_rb < 0) swap(ep0.b, ep1.b);
        if (cov_ra < 0) swap(ep0.a, ep1.a);
    }
    else if (ep0.g != ep1.g)
    {
        if (cov_gb < 0) swap(ep0.b, ep1.b);
        if (cov_ga < 0) swap(ep0.a, ep1.a);
    }
    else if (ep0.b != ep1.b)
    {
        if (cov_ba < 0) swap(ep0.a, ep1.a);
    }
}

template<typename Tpix>
float BC7_EstimateError(const Tpix pixels[PIXELS_PER_BLOCK], uint partitionMask, Tpix ep0, Tpix ep1, int indexBits)
{
    Tpix axis = ep1 - ep0;
    float mul = dot(axis, axis);
    if (mul > 0.0f) mul = 1.0 / mul;
    
    const float maxIndex = (1u << indexBits) - 1;
    const float indexScale = 1.f / maxIndex;
    
    float error = 0;

    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        if ((partitionMask & (1u << idx)) != 0)
        {
            const Tpix p = pixels[idx] - ep0;

            float f_ind = dot(p, axis) * mul;
            f_ind = round(f_ind * maxIndex) * indexScale;

            const Tpix diff = f_ind * axis - p;
            error += dot(diff, diff);
        }
    }

    return error;
}

template<int C>
float BC7_ComputeError(const vector<float,C> pixels[PIXELS_PER_BLOCK], const int indices[PIXELS_PER_BLOCK],
    uint partitionMask, vector<int,C> ep0, vector<int,C> ep1, int indexBits, int endpointBits)
{
#if BC7_SINGLE_MODE
    return 0;
#else
    ep0 = BC7_ExpandEndpointTo8Bits<C>(ep0, endpointBits);
    ep1 = BC7_ExpandEndpointTo8Bits<C>(ep1, endpointBits);

    const float maxIndex = (1u << indexBits) - 1;
    const float indexScale = 64.f / maxIndex;

    float error = 0;

    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        if ((partitionMask & (1u << idx)) != 0)
        {
            const int quantizedIndex = int(round(indices[idx] * indexScale)); // 0-64
            const vector<int,C> reconstructed = (ep0 * (64 - quantizedIndex) + ep1 * quantizedIndex + 32) >> 6;
            const vector<float,C> reconstructedNorm = vector<float,C>(reconstructed) / 255.0;
            const vector<float,C> diff = reconstructedNorm - pixels[idx];
            error += dot(diff, diff);
        }
    }

    return error;
#endif
}

int3 BC7_RGB_QuantizeEndpoint(float3 ep, int bits, bool sharedLsb)
{
    int3 qep = Quantize(ep, bits);
    if (sharedLsb)
    {
        int lsb = (qep.x & 1) + (qep.y & 1) + (qep.z & 1) >= 2 ? 1 : 0;
        qep = (qep & 0xFE) | lsb;
    }
    return qep;
}

// This is the version for Mode 1 where the P-bits are shared between ep0 and ep1
void BC7_RGB_QuantizeEndpoints_SharedLSB(float3 ep0, float3 ep1, out int3 qep0, out int3 qep1, int bits)
{
    qep0 = Quantize(ep0, bits);
    qep1 = Quantize(ep1, bits);
    
    int lsb = (qep0.x & 1) + (qep0.y & 1) + (qep0.z & 1) + (qep1.x & 1) + (qep1.y & 1) + (qep1.z & 1) >= 4 ? 1 : 0;
    qep0 = (qep0 & 0xFE) | lsb;
    qep1 = (qep1 & 0xFE) | lsb;
}

int4 BC7_RGBA_QuantizeEndpoint(float4 ep, int bits, bool sharedLsb)
{
    int4 qep = Quantize(ep, bits);
    if (sharedLsb)
    {
        // Set the shared LSB to 1 if the majority of endpoint components are odd, i.e. their LSB is 1.
        // Treat the alpha=0 and alpha=1 cases specially to preserve these values.
        bool const majorityOdd = (qep.x & 1) + (qep.y & 1) + (qep.z & 1) + (qep.w & 1) >= 2;
        bool const alphaZero = (qep.w == 0);
        bool const alphaOne = (ep.a == 1.0);

        int lsb = (majorityOdd && !alphaZero) || alphaOne ? 1 : 0;
        qep = (qep & 0xFE) | lsb;
    }
    return qep;
}

template<int C>
bool BC7_CalculateIndices(const vector<float,C> pixels[PIXELS_PER_BLOCK], uint partitionMask, uint anchor, uint indexBits, uint endpointBits,
    vector<int,C> qep0, vector<int,C> qep1, inout int quant_ind[PIXELS_PER_BLOCK])
{
    vector<float,C> ep0 = BC7_DequantizeEndpoint(qep0, endpointBits);
    vector<float,C> ep1 = BC7_DequantizeEndpoint(qep1, endpointBits);

    vector<float,C> axis = ep1 - ep0;
    float mul = dot(axis, axis);
    if (mul > 0.0f) mul = 1.0f / mul;
    
    bool need_flip = false;

    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        if ((partitionMask & (1u << idx)) != 0)
        {
            float f_ind = dot((pixels[idx] - ep0), axis) * mul;
        
            quant_ind[idx] = Quantize(f_ind, indexBits);

            if (idx == anchor)
            {
                // Need to flip if anchor pixel's high bit is 1 because that cannot be encoded
                const uint highBit = 1u << (indexBits - 1);
                need_flip = (quant_ind[idx] & highBit) != 0;
            }
        }
    }

    if (need_flip)
    {
        const uint mask = (1u << indexBits) - 1;
        [unroll]
        for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
        {
            if ((partitionMask & (1u << idx)) != 0)
            {
                quant_ind[idx] ^= mask;
            }
        }
    }

    return need_flip;
}

// A version of BC7_CalculateIndices that is specialized for a case when anchor pixel is at position 0.
// This allows us to calculate and optionally flip the indices in one loop instead of two.
template<int C>
bool BC7_CalculateIndices_Anchor0(const vector<float,C> pixels[PIXELS_PER_BLOCK], uint partitionMask, uint indexBits, uint endpointBits,
    vector<int,C> qep0, vector<int,C> qep1, out int quant_ind[PIXELS_PER_BLOCK])
{
    vector<float,C> ep0 = BC7_DequantizeEndpoint(qep0, endpointBits);
    vector<float,C> ep1 = BC7_DequantizeEndpoint(qep1, endpointBits);

    vector<float,C> axis = ep1 - ep0;
    float mul = dot(axis, axis);
    if (mul > 0.0f) mul = 1.0f / mul;

    bool need_flip = false;
    const uint mask = (1u << indexBits) - 1;

    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        float f_ind = dot((pixels[idx] - ep0), axis) * mul;
    
        quant_ind[idx] = Quantize(f_ind, indexBits);

        if (idx == 0)
        {
            // Need to flip if pixel 0's high bit is 1 because that cannot be encoded
            const uint highBit = 1u << (indexBits - 1);
            need_flip = (quant_ind[idx] & highBit) != 0;
        }

        if (need_flip)
            quant_ind[idx] ^= mask;
    }

    return need_flip;
}

void d_SetBits(inout uint4 block, inout int offset, int length, uint value)
{
    value &= (1u << length) - 1;
    if (offset < 32)
    {
        block.x |= value << offset;
        if (offset + length >= 32)
            block.y |= value >> (32 - offset);
    }
    else if (offset < 64)
    {
        block.y |= value << (offset - 32);
        if (offset + length >= 64)
            block.z |= value >> (64 - offset);
    }
    else if (offset < 96)
    {
        block.z |= value << (offset - 64);
        if (offset + length >= 96)
            block.w |= value >> (96 - offset);
    }
    else
        block.w |= value << (offset - 96);
    offset += length;
}

void d_SetBits_Hi64(inout uint4 block, inout int offset, int length, uint value)
{
    value &= (1u << length) - 1;
    if (offset < 96)
    {
        block.z |= value << (offset - 64);
        if (offset + length >= 96)
            block.w |= value >> (96 - offset);
    }
    else
        block.w |= value << (offset - 96);
    offset += length;
}

void BC7_Mode0_Finalize(const float3 pixels[PIXELS_PER_BLOCK], int bestPartId, StoredEndpoint3 bestEndpoints[6], inout float currentError, inout uint4 currentBlock)
{
    const int NUM_ENDPOINTS = 6;
    const int INDEX_BITS = 3;
    const int COLOR_BITS = 5;

    float3 ep[NUM_ENDPOINTS];
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; ++i)
        ep[i] = bestEndpoints[i];

    BC7_OptimizeEndpoints(pixels, c_Partitions3_Subset0[bestPartId], ep[0], ep[1], INDEX_BITS);
    BC7_OptimizeEndpoints(pixels, c_Partitions3_Subset1[bestPartId], ep[2], ep[3], INDEX_BITS);
    BC7_OptimizeEndpoints(pixels, c_Partitions3_Subset2[bestPartId], ep[4], ep[5], INDEX_BITS);

    int3 qep[NUM_ENDPOINTS];
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; ++i)
    {
        qep[i] = BC7_RGB_QuantizeEndpoint(ep[i], COLOR_BITS, true);
    }

    float error;
    int iquant_ind[PIXELS_PER_BLOCK];
    if (BC7_CalculateIndices_Anchor0<3>(pixels, c_Partitions3_Subset0[bestPartId], INDEX_BITS, COLOR_BITS, qep[0], qep[1], iquant_ind))
    {
        swap(qep[0], qep[1]);
    }
    error = BC7_ComputeError(pixels, iquant_ind, c_Partitions3_Subset0[bestPartId], qep[0], qep[1], INDEX_BITS, COLOR_BITS);

    if (BC7_CalculateIndices(pixels, c_Partitions3_Subset1[bestPartId], c_Partitions3_Anchor1[bestPartId], INDEX_BITS, COLOR_BITS, qep[2], qep[3], iquant_ind))
    {
        swap(qep[2], qep[3]);
    }
    error += BC7_ComputeError(pixels, iquant_ind, c_Partitions3_Subset1[bestPartId], qep[2], qep[3], INDEX_BITS, COLOR_BITS);

    if (BC7_CalculateIndices(pixels, c_Partitions3_Subset2[bestPartId], c_Partitions3_Anchor2[bestPartId], INDEX_BITS, COLOR_BITS, qep[4], qep[5], iquant_ind))
    {
        swap(qep[4], qep[5]);
    }
    error += BC7_ComputeError(pixels, iquant_ind, c_Partitions3_Subset2[bestPartId], qep[4], qep[5], INDEX_BITS, COLOR_BITS);

    if (error > currentError)
        return;

    // Encode the block

    uint4 block = 0;
    int base = 0;
    d_SetBits(block, base, 1, 1);
    d_SetBits(block, base, 4, bestPartId);
    // Endpoints
    [unroll]
    for (int c = 0; c < 3; ++c)
    {
        [unroll]
        for (int i = 0; i < NUM_ENDPOINTS; ++i)
        {
            d_SetBits(block, base, COLOR_BITS - 1, uint(qep[i][c]) >> 1);
        }
    }
    // P-bits
    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        d_SetBits(block, base, 1, qep[i].x);
    }
    // Indices
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        bool isAnchor = (idx == 0) ||
            (idx == c_Partitions3_Anchor1[bestPartId]) ||
            (idx == c_Partitions3_Anchor2[bestPartId]);
        int bits = isAnchor ? INDEX_BITS - 1 : INDEX_BITS;
        d_SetBits_Hi64(block, base, bits, iquant_ind[idx]);
    }

    currentError = error;
    currentBlock = block;
}

void BC7_Mode2_Finalize(const float3 pixels[PIXELS_PER_BLOCK], int bestPartId, StoredEndpoint3 bestEndpoints[6], inout float currentError, inout uint4 currentBlock)
{
    const int NUM_ENDPOINTS = 6;
    const int INDEX_BITS = 2;
    const int COLOR_BITS = 5;

    float3 ep[NUM_ENDPOINTS];
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; ++i)
        ep[i] = bestEndpoints[i];

    BC7_OptimizeEndpoints(pixels, c_Partitions3_Subset0[bestPartId], ep[0], ep[1], INDEX_BITS);
    BC7_OptimizeEndpoints(pixels, c_Partitions3_Subset1[bestPartId], ep[2], ep[3], INDEX_BITS);
    BC7_OptimizeEndpoints(pixels, c_Partitions3_Subset2[bestPartId], ep[4], ep[5], INDEX_BITS);

    int3 qep[NUM_ENDPOINTS];
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; ++i)
    {
        qep[i] = BC7_RGB_QuantizeEndpoint(ep[i], COLOR_BITS, false);
    }

    float error;
    int iquant_ind[PIXELS_PER_BLOCK];
    if (BC7_CalculateIndices_Anchor0<3>(pixels, c_Partitions3_Subset0[bestPartId], INDEX_BITS, COLOR_BITS, qep[0], qep[1], iquant_ind))
    {
        swap(qep[0], qep[1]);
    }
    error = BC7_ComputeError(pixels, iquant_ind, c_Partitions3_Subset0[bestPartId], qep[0], qep[1], INDEX_BITS, COLOR_BITS);

    if (BC7_CalculateIndices(pixels, c_Partitions3_Subset1[bestPartId], c_Partitions3_Anchor1[bestPartId], INDEX_BITS, COLOR_BITS, qep[2], qep[3], iquant_ind))
    {
        swap(qep[2], qep[3]);
    }
    error += BC7_ComputeError(pixels, iquant_ind, c_Partitions3_Subset1[bestPartId], qep[2], qep[3], INDEX_BITS, COLOR_BITS);

    if (BC7_CalculateIndices(pixels, c_Partitions3_Subset2[bestPartId], c_Partitions3_Anchor2[bestPartId], INDEX_BITS, COLOR_BITS, qep[4], qep[5], iquant_ind))
    {
        swap(qep[4], qep[5]);
    }
    error += BC7_ComputeError(pixels, iquant_ind, c_Partitions3_Subset2[bestPartId], qep[4], qep[5], INDEX_BITS, COLOR_BITS);

    if (error > currentError)
        return;

    // Encode the block

    uint4 block = 0;
    int base = 0;
    d_SetBits(block, base, 3, 0x4);
    d_SetBits(block, base, 6, bestPartId);
    // Endpoints
    [unroll]
    for (int c = 0; c < 3; ++c)
    {
        [unroll]
        for (int i = 0; i < NUM_ENDPOINTS; ++i)
        {
            d_SetBits(block, base, COLOR_BITS, uint(qep[i][c]));
        }
    }
    // Indices
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        bool isAnchor = (idx == 0) ||
            (idx == c_Partitions3_Anchor1[bestPartId]) ||
            (idx == c_Partitions3_Anchor2[bestPartId]);
        int bits = isAnchor ? INDEX_BITS - 1 : INDEX_BITS;
        d_SetBits_Hi64(block, base, bits, iquant_ind[idx]);
    }

    currentError = error;
    currentBlock = block;
}

void BC7_Mode0_2_Compress(const float3 pixels[PIXELS_PER_BLOCK], uint mode, uint partition, inout float currentError, inout uint4 currentBlock)
{
    const int NUM_ENDPOINTS = 6;
    const int SEARCH_INDEX_BITS = 3; // Mode 0 has 3, Mode 1 has 2

    StoredEndpoint3 bestEndpoints[NUM_ENDPOINTS];
    
#if BC7_SINGLE_MODE
    int partId = int(partition);
    int bestPartId = partId;
#else
    int bestPartId = 0;
    float bestError = FLT_MAX;
    [loop]
    for (int partId = 0; partId < 64; ++partId)
#endif
    {
        float3 ep[NUM_ENDPOINTS];
        BC7_MinMax(pixels, c_Partitions3_Subset0[partId], ep[0], ep[1]);
        BC7_MinMax(pixels, c_Partitions3_Subset1[partId], ep[2], ep[3]);
        BC7_MinMax(pixels, c_Partitions3_Subset2[partId], ep[4], ep[5]);

        BC7_SwapEndpointComponents_RGB(pixels, c_Partitions3_Subset0[partId], ep[0], ep[1]);
        BC7_SwapEndpointComponents_RGB(pixels, c_Partitions3_Subset1[partId], ep[2], ep[3]);
        BC7_SwapEndpointComponents_RGB(pixels, c_Partitions3_Subset2[partId], ep[4], ep[5]);

#if BC7_SINGLE_MODE
        [unroll]
        for (int i = 0; i < NUM_ENDPOINTS; ++i)
            bestEndpoints[i] = StoredEndpoint3(ep[i]);
#else
        float error;
        error  = BC7_EstimateError(pixels, c_Partitions3_Subset0[partId], ep[0], ep[1], SEARCH_INDEX_BITS);
        error += BC7_EstimateError(pixels, c_Partitions3_Subset1[partId], ep[2], ep[3], SEARCH_INDEX_BITS);
        error += BC7_EstimateError(pixels, c_Partitions3_Subset2[partId], ep[4], ep[5], SEARCH_INDEX_BITS);

        if (error < bestError)
        {
            bestPartId = partId;
            bestError = error;
            [unroll]
            for (int i = 0; i < NUM_ENDPOINTS; ++i)
                bestEndpoints[i] = StoredEndpoint3(ep[i]);
        }
#endif
    }

#if BC7_SINGLE_MODE
    if (mode == 0)
        BC7_Mode0_Finalize(pixels, bestPartId, bestEndpoints, currentError, currentBlock);
    else
        BC7_Mode2_Finalize(pixels, bestPartId, bestEndpoints, currentError, currentBlock);
#else
    if (bestPartId < 16)
        BC7_Mode0_Finalize(pixels, bestPartId, bestEndpoints, currentError, currentBlock);
    BC7_Mode2_Finalize(pixels, bestPartId, bestEndpoints, currentError, currentBlock);
#endif
}

void BC7_Mode1_Finalize(const float3 pixels[PIXELS_PER_BLOCK], int bestPartId, StoredEndpoint3 bestEndpoints[4], inout float currentError, inout uint4 currentBlock)
{
    const int NUM_ENDPOINTS = 4;
    const int INDEX_BITS = 3;
    const int COLOR_BITS = 7;

    float3 ep[NUM_ENDPOINTS];
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; ++i)
        ep[i] = bestEndpoints[i];

    BC7_OptimizeEndpoints(pixels, c_Partitions2_Subset0[bestPartId], ep[0], ep[1], INDEX_BITS);
    BC7_OptimizeEndpoints(pixels, c_Partitions2_Subset1[bestPartId], ep[2], ep[3], INDEX_BITS);

    int3 qep[NUM_ENDPOINTS];
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; i += 2)
    {
        BC7_RGB_QuantizeEndpoints_SharedLSB(ep[i], ep[i + 1], qep[i], qep[i + 1], COLOR_BITS);
    }

    float error;
    int iquant_ind[PIXELS_PER_BLOCK];
    if (BC7_CalculateIndices_Anchor0<3>(pixels, c_Partitions2_Subset0[bestPartId], INDEX_BITS, COLOR_BITS, qep[0], qep[1], iquant_ind))
    {
        swap(qep[0], qep[1]);
    }
    error = BC7_ComputeError(pixels, iquant_ind, c_Partitions2_Subset0[bestPartId], qep[0], qep[1], INDEX_BITS, COLOR_BITS);

    if (BC7_CalculateIndices(pixels, c_Partitions2_Subset1[bestPartId], c_Partitions2_Anchor1[bestPartId], INDEX_BITS, COLOR_BITS, qep[2], qep[3], iquant_ind))
    {
        swap(qep[2], qep[3]);
    }
    error += BC7_ComputeError(pixels, iquant_ind, c_Partitions2_Subset1[bestPartId], qep[2], qep[3], INDEX_BITS, COLOR_BITS);

    if (error > currentError)
        return;

    // Encode the block

    uint4 block = 0;
    int base = 0;
    d_SetBits(block, base, 2, 0x2);
    d_SetBits(block, base, 6, bestPartId);
    // Endpoints
    [unroll]
    for (int c = 0; c < 3; ++c)
    {
        [unroll]
        for (int i = 0; i < NUM_ENDPOINTS; ++i)
        {
            d_SetBits(block, base, COLOR_BITS - 1, uint(qep[i][c]) >> 1);
        }
    }
    // P-bits
    d_SetBits(block, base, 1, qep[0].x);
    d_SetBits(block, base, 1, qep[2].x);
    // Indices
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        bool isAnchor = (idx == 0) ||
            (idx == c_Partitions2_Anchor1[bestPartId]);
        int bits = isAnchor ? INDEX_BITS - 1 : INDEX_BITS;
        d_SetBits_Hi64(block, base, bits, iquant_ind[idx]);
    }

    currentError = error;
    currentBlock = block;
}

void BC7_Mode3_Finalize(const float3 pixels[PIXELS_PER_BLOCK], int bestPartId, StoredEndpoint3 bestEndpoints[4], inout float currentError, inout uint4 currentBlock)
{
    const int NUM_ENDPOINTS = 4;
    const int INDEX_BITS = 2;
    const int COLOR_BITS = 8;

    float3 ep[NUM_ENDPOINTS];
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; ++i)
        ep[i] = bestEndpoints[i];

    BC7_OptimizeEndpoints(pixels, c_Partitions2_Subset0[bestPartId], ep[0], ep[1], INDEX_BITS);
    BC7_OptimizeEndpoints(pixels, c_Partitions2_Subset1[bestPartId], ep[2], ep[3], INDEX_BITS);

    int3 qep[NUM_ENDPOINTS];
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; ++i)
    {
        qep[i] = BC7_RGB_QuantizeEndpoint(ep[i], COLOR_BITS, true);
    }

    float error;
    int iquant_ind[PIXELS_PER_BLOCK];
    if (BC7_CalculateIndices_Anchor0<3>(pixels, c_Partitions2_Subset0[bestPartId], INDEX_BITS, COLOR_BITS, qep[0], qep[1], iquant_ind))
    {
        swap(qep[0], qep[1]);
    }
    error = BC7_ComputeError(pixels, iquant_ind, c_Partitions2_Subset0[bestPartId], qep[0], qep[1], INDEX_BITS, COLOR_BITS);

    if (BC7_CalculateIndices(pixels, c_Partitions2_Subset1[bestPartId], c_Partitions2_Anchor1[bestPartId], INDEX_BITS, COLOR_BITS, qep[2], qep[3], iquant_ind))
    {
        swap(qep[2], qep[3]);
    }
    error += BC7_ComputeError(pixels, iquant_ind, c_Partitions2_Subset1[bestPartId], qep[2], qep[3], INDEX_BITS, COLOR_BITS);

    if (error > currentError)
        return;

    // Encode the block

    uint4 block = 0;
    int base = 0;
    d_SetBits(block, base, 4, 0x8);
    d_SetBits(block, base, 6, bestPartId);
    // Endpoints
    [unroll]
    for (int c = 0; c < 3; ++c)
    {
        [unroll]
        for (int i = 0; i < NUM_ENDPOINTS; ++i)
        {
            d_SetBits(block, base, COLOR_BITS - 1, uint(qep[i][c]) >> 1);
        }
    }
    // P-bits
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; ++i)
        d_SetBits(block, base, 1, qep[i].x);
    // Indices
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        bool isAnchor = (idx == 0) ||
            (idx == c_Partitions2_Anchor1[bestPartId]);
        int bits = isAnchor ? INDEX_BITS - 1 : INDEX_BITS;
        d_SetBits_Hi64(block, base, bits, iquant_ind[idx]);
    }

    currentError = error;
    currentBlock = block;
}

void BC7_Mode1_3_Compress(const float3 pixels[PIXELS_PER_BLOCK], uint mode, uint partition, inout float currentError, inout uint4 currentBlock)
{
    const int NUM_ENDPOINTS = 4;
    const int SEARCH_INDEX_BITS = 3; // Mode 1 has 3, Mode 3 has 2

    StoredEndpoint3 bestEndpoints[NUM_ENDPOINTS];
#if BC7_SINGLE_MODE
    int partId = int(partition);
    int bestPartId = partId;
#else
    int bestPartId = 0;
    float bestError = FLT_MAX;
    [loop]
    for (int partId = 0; partId < 64; ++partId)
#endif
    {
        float3 ep[NUM_ENDPOINTS];
        BC7_MinMax(pixels, c_Partitions2_Subset0[partId], ep[0], ep[1]);
        BC7_MinMax(pixels, c_Partitions2_Subset1[partId], ep[2], ep[3]);

        BC7_SwapEndpointComponents_RGB(pixels, c_Partitions2_Subset0[partId], ep[0], ep[1]);
        BC7_SwapEndpointComponents_RGB(pixels, c_Partitions2_Subset1[partId], ep[2], ep[3]);

#if BC7_SINGLE_MODE
        [unroll]
        for (int i = 0; i < NUM_ENDPOINTS; ++i)
            bestEndpoints[i] = StoredEndpoint3(ep[i]);
#else
        float error;
        error  = BC7_EstimateError(pixels, c_Partitions2_Subset0[partId], ep[0], ep[1], SEARCH_INDEX_BITS);
        error += BC7_EstimateError(pixels, c_Partitions2_Subset1[partId], ep[2], ep[3], SEARCH_INDEX_BITS);

        if (error < bestError)
        {
            bestPartId = partId;
            bestError = error;
            [unroll]
            for (int i = 0; i < NUM_ENDPOINTS; ++i)
                bestEndpoints[i] = StoredEndpoint3(ep[i]);
        }
#endif
    }

#if BC7_SINGLE_MODE
    if (mode == 1)
        BC7_Mode1_Finalize(pixels, bestPartId, bestEndpoints, currentError, currentBlock);
    else
        BC7_Mode3_Finalize(pixels, bestPartId, bestEndpoints, currentError, currentBlock);
#else
    BC7_Mode1_Finalize(pixels, bestPartId, bestEndpoints, currentError, currentBlock);
    BC7_Mode3_Finalize(pixels, bestPartId, bestEndpoints, currentError, currentBlock);
#endif
}

template<bool HR_COLOR>
void BC7_Mode4_Compress_Internal(const float4 pixels[PIXELS_PER_BLOCK], uint rotation, inout float currentError, inout uint4 currentBlock)
{
    const int COLOR_INDEX_BITS = HR_COLOR ? 3 : 2;
    const int ALPHA_INDEX_BITS = HR_COLOR ? 2 : 3;
    const int COLOR_BITS = 5;
    const int ALPHA_BITS = 6;

    float3 colorPixels[PIXELS_PER_BLOCK];
    float alphaPixels[PIXELS_PER_BLOCK];
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        colorPixels[idx] = pixels[idx].rgb;
        alphaPixels[idx] = pixels[idx].a;
    }

    float a0, a1;
    BC7_MinMax<float>(alphaPixels, NO_PARTITIONS, a0, a1);
    BC7_OptimizeEndpoints<float>(alphaPixels, NO_PARTITIONS, a0, a1, ALPHA_INDEX_BITS);

    float3 ep0, ep1;
    BC7_MinMax<float3>(colorPixels, NO_PARTITIONS, ep0, ep1);
    BC7_SwapEndpointComponents_RGB(colorPixels, NO_PARTITIONS, ep0, ep1);
    BC7_OptimizeEndpoints<float3>(colorPixels, NO_PARTITIONS, ep0, ep1, COLOR_INDEX_BITS);

    int3 qep0 = Quantize(ep0, COLOR_BITS);
    int3 qep1 = Quantize(ep1, COLOR_BITS);
    int qa0 = Quantize(a0, ALPHA_BITS);
    int qa1 = Quantize(a1, ALPHA_BITS);

    float error;
    int colorIndices[PIXELS_PER_BLOCK];
    if (BC7_CalculateIndices_Anchor0<3>(colorPixels, NO_PARTITIONS, COLOR_INDEX_BITS, COLOR_BITS, qep0, qep1, colorIndices))
    {
        swap(qep0, qep1);
    }
    error = BC7_ComputeError<3>(colorPixels, colorIndices, NO_PARTITIONS, qep0, qep1, COLOR_INDEX_BITS, COLOR_BITS);

    int alphaIndices[PIXELS_PER_BLOCK];
    if (BC7_CalculateIndices_Anchor0<1>(alphaPixels, NO_PARTITIONS, ALPHA_INDEX_BITS, ALPHA_BITS, qa0, qa1, alphaIndices))
    {
        swap(qa0, qa1);
    }
    error += BC7_ComputeError<1>(alphaPixels, alphaIndices, NO_PARTITIONS, qa0, qa1, ALPHA_INDEX_BITS, ALPHA_BITS);

    if (error > currentError)
        return;

    // Encode the block

    uint4 block = 0;
    int base = 0;
    d_SetBits(block, base, 5, 1u << 4);
    d_SetBits(block, base, 2, rotation); // Rotation
    d_SetBits(block, base, 1, HR_COLOR ? 1 : 0); // IdxMode
    // Endpoints
    [unroll]
    for (int c = 0; c < 3; ++c)
    {
        d_SetBits(block, base, COLOR_BITS, uint(qep0[c]));
        d_SetBits(block, base, COLOR_BITS, uint(qep1[c]));
    }
    d_SetBits(block, base, ALPHA_BITS, uint(qa0));
    d_SetBits(block, base, ALPHA_BITS, uint(qa1));
    // Indices
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        if (HR_COLOR)
            d_SetBits(block, base, (idx == 0) ? ALPHA_INDEX_BITS - 1 : ALPHA_INDEX_BITS, alphaIndices[idx]);
        else
            d_SetBits(block, base, (idx == 0) ? COLOR_INDEX_BITS - 1 : COLOR_INDEX_BITS, colorIndices[idx]);
    }
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        if (HR_COLOR)
            d_SetBits_Hi64(block, base, (idx == 0) ? COLOR_INDEX_BITS - 1 : COLOR_INDEX_BITS, colorIndices[idx]);
        else
            d_SetBits_Hi64(block, base, (idx == 0) ? ALPHA_INDEX_BITS - 1 : ALPHA_INDEX_BITS, alphaIndices[idx]);
    }

    currentError = error;
    currentBlock = block;
}

void BC7_Mode5_Compress_Internal(const float4 pixels[PIXELS_PER_BLOCK], uint rotation, inout float currentError, inout uint4 currentBlock)
{
    const int INDEX_BITS = 2;
    const int COLOR_BITS = 7;
    const int ALPHA_BITS = 8;

    float3 colorPixels[PIXELS_PER_BLOCK];
    float alphaPixels[PIXELS_PER_BLOCK];
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        colorPixels[idx] = pixels[idx].rgb;
        alphaPixels[idx] = pixels[idx].a;
    }

    float a0, a1;
    BC7_MinMax<float>(alphaPixels, NO_PARTITIONS, a0, a1);
    BC7_OptimizeEndpoints<float>(alphaPixels, NO_PARTITIONS, a0, a1, INDEX_BITS);

    float3 ep0, ep1;
    BC7_MinMax<float3>(colorPixels, NO_PARTITIONS, ep0, ep1);
    BC7_SwapEndpointComponents_RGB(colorPixels, NO_PARTITIONS, ep0, ep1);
    BC7_OptimizeEndpoints<float3>(colorPixels, NO_PARTITIONS, ep0, ep1, INDEX_BITS);

    int3 qep0 = Quantize(ep0, COLOR_BITS);
    int3 qep1 = Quantize(ep1, COLOR_BITS);
    int qa0 = Quantize(a0, ALPHA_BITS);
    int qa1 = Quantize(a1, ALPHA_BITS);

    float error;
    int colorIndices[PIXELS_PER_BLOCK];
    if (BC7_CalculateIndices_Anchor0<3>(colorPixels, NO_PARTITIONS, INDEX_BITS, COLOR_BITS, qep0, qep1, colorIndices))
    {
        swap(qep0, qep1);
    }
    error = BC7_ComputeError<3>(colorPixels, colorIndices, NO_PARTITIONS, qep0, qep1, INDEX_BITS, COLOR_BITS);

    int alphaIndices[PIXELS_PER_BLOCK];
    if (BC7_CalculateIndices_Anchor0<1>(alphaPixels, NO_PARTITIONS, INDEX_BITS, ALPHA_BITS, qa0, qa1, alphaIndices))
    {
        swap(qa0, qa1);
    }
    error += BC7_ComputeError<1>(alphaPixels, alphaIndices, NO_PARTITIONS, qa0, qa1, INDEX_BITS, ALPHA_BITS);

    if (error > currentError)
        return;

    // Encode the block

    uint4 block = 0;
    int base = 0;
    d_SetBits(block, base, 6, 1u << 5);
    d_SetBits(block, base, 2, rotation); // Rotation
    // Endpoints
    [unroll]
    for (int c = 0; c < 3; ++c)
    {
        d_SetBits(block, base, COLOR_BITS, uint(qep0[c]));
        d_SetBits(block, base, COLOR_BITS, uint(qep1[c]));
    }
    d_SetBits(block, base, ALPHA_BITS, uint(qa0));
    d_SetBits(block, base, ALPHA_BITS, uint(qa1));
    // Indices
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        d_SetBits_Hi64(block, base, (idx == 0) ? INDEX_BITS - 1 : INDEX_BITS, colorIndices[idx]);
    }
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        d_SetBits_Hi64(block, base, (idx == 0) ? INDEX_BITS - 1 : INDEX_BITS, alphaIndices[idx]);
    }

    currentError = error;
    currentBlock = block;
}

// NOTE: The 'pixels' parameter is declared as inout. This function modifies the order of channels in the pixel data,
// to try different rotation modes, but then reorders them back to the original RGBA. Doing this using an 'inout'
// parameter is faster than a regular 'in', presumably because of smaller register pressure.
void BC7_Mode45_Compress(inout float4 pixels[PIXELS_PER_BLOCK], inout float currentError, inout uint4 currentBlock)
{
    [loop]
    for (uint rotation = 0; rotation < 4; ++rotation)
    {
        BC7_Mode4_Compress_Internal<false>(pixels, rotation, currentError, currentBlock);

        BC7_Mode4_Compress_Internal<true>(pixels, rotation, currentError, currentBlock);

        BC7_Mode5_Compress_Internal(pixels, rotation, currentError, currentBlock);
        
        // Rotate the channels for the next rotation scheme, or back to the original.
        // 00 – Block format is Vector(RGB) Scalar(A) - no swapping
        // 01 – Block format is Vector(AGB) Scalar(R) - swap A and R
        // 10 – Block format is Vector(RAB) Scalar(G) - swap A and G
        // 11 - Block format is Vector(RGA) Scalar(B) - swap A and B
        switch(rotation)
        {
            case 0:
                [unroll]
                for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
                {
                    // Source order: RGBA, dest order: AGBR
                    pixels[idx] = pixels[idx].agbr;
                }
                break;
            case 1:
                [unroll]
                for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
                {
                    // Source order: AGBR, dest order: RABG
                    pixels[idx] = pixels[idx].arbg;
                }
                break;
            case 2:
                [unroll]
                for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
                {
                    // Source order: RABG, dest order: RGAB
                    pixels[idx] = pixels[idx].ragb;
                }
                break;
            default:
                [unroll]
                for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
                {
                    // Source order: RGAB, dest order: RGBA
                    pixels[idx] = pixels[idx].rgab;
                }
                break;
        }
    }
}

void BC7_Mode45_Compress_SingleMode(float4 pixels[PIXELS_PER_BLOCK], uint mode, uint partition, inout float currentError, inout uint4 currentBlock)
{
    bool hr_color = (partition & 4) != 0;
    uint rotation = partition & 3;

    // Rotate the channels
    // 00 – Vector(RGB) Scalar(A) - no swapping
    // 01 – Vector(AGB) Scalar(R)
    // 10 – Vector(RAB) Scalar(G)
    // 11 - Vector(RGA) Scalar(B)
    switch(rotation)
    {
        case 1:
            [unroll]
            for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
            {
                pixels[idx] = pixels[idx].agbr;
            }
            break;
        case 2:
            [unroll]
            for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
            {
                pixels[idx] = pixels[idx].rabg;
            }
            break;
        case 3:
            [unroll]
            for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
            {
                pixels[idx] = pixels[idx].rgab;
            }
            break;
    }

    if (mode == 4 && !hr_color)
        BC7_Mode4_Compress_Internal<false>(pixels, rotation, currentError, currentBlock);
    else if (mode == 4 && hr_color)
        BC7_Mode4_Compress_Internal<true>(pixels, rotation, currentError, currentBlock);
    else // mode == 5
        BC7_Mode5_Compress_Internal(pixels, rotation, currentError, currentBlock);
}

void BC7_Mode6_Compress(const float4 pixels[PIXELS_PER_BLOCK], inout float currentError, inout uint4 currentBlock)
{
    const int INDEX_BITS = 4;
    const int COLOR_BITS = 8;

    float4 ep0, ep1;
    BC7_MinMax(pixels, NO_PARTITIONS, ep0, ep1);
    BC7_SwapEndpointComponents_RGBA(pixels, NO_PARTITIONS, ep0, ep1);
    BC7_OptimizeEndpoints(pixels, NO_PARTITIONS, ep0, ep1, INDEX_BITS);
    
    int4 qep0 = BC7_RGBA_QuantizeEndpoint(ep0, COLOR_BITS, true);
    int4 qep1 = BC7_RGBA_QuantizeEndpoint(ep1, COLOR_BITS, true);
    
    int iquant_ind[PIXELS_PER_BLOCK];
    if (BC7_CalculateIndices_Anchor0<4>(pixels, NO_PARTITIONS, INDEX_BITS, COLOR_BITS, qep0, qep1, iquant_ind))
    {
        swap(qep0, qep1);
    }

    float error = BC7_ComputeError(pixels, iquant_ind, NO_PARTITIONS, qep0, qep1, INDEX_BITS, COLOR_BITS);
    if (error > currentError)
        return;

    // Encode the block

    uint4 block = 0;
    int base = 0;
    d_SetBits(block, base, 7, 1u << 6);
    // Endpoints
    [unroll]
    for (int c = 0; c < 4; ++c)
    {
        d_SetBits(block, base, COLOR_BITS - 1, uint(qep0[c]) >> 1);
        d_SetBits(block, base, COLOR_BITS - 1, uint(qep1[c]) >> 1);
    }
    // P-bits
    d_SetBits(block, base, 1, qep0.x);
    d_SetBits(block, base, 1, qep1.x);
    // Indices
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        d_SetBits_Hi64(block, base, (idx == 0) ? INDEX_BITS - 1 : INDEX_BITS, iquant_ind[idx]);
    }
    
    currentError = error;
    currentBlock = block;
}

void BC7_Mode7_Compress(const float4 pixels[PIXELS_PER_BLOCK], uint partition, inout float currentError, inout uint4 currentBlock)
{
    const int NUM_ENDPOINTS = 4;
    const int INDEX_BITS = 2;
    const int COLOR_BITS = 6;


#if BC7_SINGLE_MODE
    int partId = int(partition);
    int bestPartId = partId;
#else
    int bestPartId = 0;
    float bestError = FLT_MAX;
    StoredEndpoint4 bestEndpoints[NUM_ENDPOINTS];

    [loop]
    for (int partId = 0; partId < 64; ++partId)
    {
#endif
        float4 ep[NUM_ENDPOINTS];
        BC7_MinMax(pixels, c_Partitions2_Subset0[partId], ep[0], ep[1]);
        BC7_MinMax(pixels, c_Partitions2_Subset1[partId], ep[2], ep[3]);

        if (false) // Disabled because overall quality improvement is very small
        {
            BC7_SwapEndpointComponents_RGBA(pixels, c_Partitions2_Subset0[partId], ep[0], ep[1]);
            BC7_SwapEndpointComponents_RGBA(pixels, c_Partitions2_Subset1[partId], ep[2], ep[3]);
        }

#if !BC7_SINGLE_MODE
        float error;
        error = BC7_EstimateError(pixels, c_Partitions2_Subset0[partId], ep[0], ep[1], INDEX_BITS);
        error += BC7_EstimateError(pixels, c_Partitions2_Subset1[partId], ep[2], ep[3], INDEX_BITS);

        if (error < bestError)
        {
            bestPartId = partId;
            bestError = error;
            [unroll]
            for (int i = 0; i < NUM_ENDPOINTS; ++i)
                bestEndpoints[i] = StoredEndpoint4(ep[i]);
        }
    }

    float4 ep[NUM_ENDPOINTS];
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; ++i)
        ep[i] = bestEndpoints[i];
#endif

    BC7_OptimizeEndpoints(pixels, c_Partitions2_Subset0[bestPartId], ep[0], ep[1], INDEX_BITS);
    BC7_OptimizeEndpoints(pixels, c_Partitions2_Subset1[bestPartId], ep[2], ep[3], INDEX_BITS);
    
    int4 qep[NUM_ENDPOINTS];
    [unroll]
    for (int i = 0; i < NUM_ENDPOINTS; ++i)
    {
        qep[i] = BC7_RGBA_QuantizeEndpoint(ep[i], COLOR_BITS, true);
    }

    float error;
    int iquant_ind[PIXELS_PER_BLOCK];
    if (BC7_CalculateIndices_Anchor0<4>(pixels, c_Partitions2_Subset0[bestPartId], INDEX_BITS, COLOR_BITS, qep[0], qep[1], iquant_ind))
    {
        swap(qep[0], qep[1]);
    }
    error = BC7_ComputeError(pixels, iquant_ind, c_Partitions2_Subset0[bestPartId], qep[0], qep[1], INDEX_BITS, COLOR_BITS);

    if (BC7_CalculateIndices(pixels, c_Partitions2_Subset1[bestPartId], c_Partitions2_Anchor1[bestPartId], INDEX_BITS, COLOR_BITS, qep[2], qep[3], iquant_ind))
    {
        swap(qep[2], qep[3]);
    }
    error += BC7_ComputeError(pixels, iquant_ind, c_Partitions2_Subset1[bestPartId], qep[2], qep[3], INDEX_BITS, COLOR_BITS);

    if (error > currentError)
        return;

    // Encode the block

    uint4 block = 0;
    int base = 0;
    d_SetBits(block, base, 8, 0x80);
    d_SetBits(block, base, 6, bestPartId);
    // Endpoints
    [unroll]
    for (int c = 0; c < 4; ++c)
    {
        [unroll]
        for (int i = 0; i < NUM_ENDPOINTS; ++i)
        {
            d_SetBits(block, base, COLOR_BITS - 1, uint(qep[i][c]) >> 1);
        }
    }
    // P-bits
    d_SetBits(block, base, 1, qep[0].x);
    d_SetBits(block, base, 1, qep[1].x);
    d_SetBits(block, base, 1, qep[2].x);
    d_SetBits(block, base, 1, qep[3].x);
    // Indices
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        bool isAnchor = (idx == 0) ||
            (idx == c_Partitions2_Anchor1[bestPartId]);
        int bits = isAnchor ? INDEX_BITS - 1 : INDEX_BITS;
        d_SetBits_Hi64(block, base, bits, iquant_ind[idx]);
    }

    currentError = error;
    currentBlock = block;
}

#if BC7_SINGLE_MODE
bool LoadModeAndPartition(uint2 blockPos, out uint mode, out uint partition)
{
    blockPos.x += g_Const.modeMapOffsetX;
    blockPos.y += g_Const.modeMapOffsetY;

    mode = 0;
    partition = 0;

    if (blockPos.x >= g_Const.modeMapWidthInBlocks || blockPos.y >= g_Const.modeMapHeightInBlocks)
        return false;

    const uint firstBitIndex = (blockPos.y * g_Const.modeMapWidthInBlocks + blockPos.x) * BLOCK_COMPRESS_BC7_MP_BITS;
    const uint lowWordOffset = g_Const.modeBufferByteOffset + (firstBitIndex >> 3) & ~3;
    const uint bitOffset = firstBitIndex & 31;

    const uint lowWord = t_Modes.Load(lowWordOffset);
    uint highWord = 0;
    if (bitOffset + BLOCK_COMPRESS_BC7_MP_BITS > 32) 
        highWord = t_Modes.Load(lowWordOffset + 4);

    uint modeAndPartition = (lowWord >> bitOffset) | (highWord << (32 - bitOffset));
    modeAndPartition &= (1 << BLOCK_COMPRESS_BC7_MP_BITS) - 1;
    
    // Zero value is reserved to indicate integration errors
    if (modeAndPartition == 0)
        return false;

    if (modeAndPartition == BLOCK_COMPRESS_BC7_MODE0_PART0_VALUE)
        modeAndPartition = 0;

    mode = (modeAndPartition >> 6) & 7;
    partition = modeAndPartition & 0x3F;

    return true;
}

#define BC7_MODES 8
groupshared uint s_ActiveThreadCount;
groupshared uint s_ModeCounts[BC7_MODES];
groupshared uint s_TasksPerMode[BC7_MODES][BLOCK_COMPRESS_BC7_CS_GROUP_WIDTH * BLOCK_COMPRESS_BC7_CS_GROUP_HEIGHT];

void ReorderThreadsByMode(uint2 threadIdx, inout uint2 blockPos, inout uint mode, inout uint partition)
{
    // Pack the block position and partition into a single 32-bit uint.
    // 12 bits should be enough for x and y because max size of a graphics texture is 16k, or 4k blocks.
    // Partition index is 6 bits.
    const uint positionMask = 0xFFF;
    const uint partitionMask = 0x3F;
    uint task = (partition << 24) | ((blockPos.y & positionMask) << 12) | (blockPos.x & positionMask);

    uint taskIndex;
    InterlockedAdd(s_ModeCounts[mode], 1, taskIndex);
    s_TasksPerMode[mode][taskIndex] = task;

    // Enumerate all active threads in the group.
    // Note that some threads may be inactive because the entire group is not needed for this image tile,
    // or some threads may have been killed due to invalid mode/partition data.
    const uint activeThreadsInWave = WaveActiveCountBits(true);
    uint firstLinearIndexInWave = 0;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(s_ActiveThreadCount, activeThreadsInWave, firstLinearIndexInWave);
    }
    firstLinearIndexInWave = WaveReadLaneFirst(firstLinearIndexInWave);
    const uint activeThreadIdxInWave = WavePrefixCountBits(true);
    uint linearThreadIdx = firstLinearIndexInWave + activeThreadIdxInWave;

    GroupMemoryBarrierWithGroupSync();

    for (uint m = 0; m < BC7_MODES; ++m)
    {
        uint modeCount = s_ModeCounts[m];
        if (linearThreadIdx < modeCount)
        {
            uint pos = s_TasksPerMode[m][linearThreadIdx];
            blockPos.x = pos & positionMask;
            blockPos.y = (pos >> 12) & positionMask;
            partition = (pos >> 24) & partitionMask;
            mode = m;
            break;
        }
        linearThreadIdx -= modeCount;
    }
}
#endif

void LoadColorsForBlock(uint2 blockPos, out float4 colors4[PIXELS_PER_BLOCK], out float3 colors3[PIXELS_PER_BLOCK],
    out bool opaqueBlock)
{
    opaqueBlock = true;
    [unroll]
    for (int idx = 0; idx < PIXELS_PER_BLOCK; ++idx)
    {
        const float4 color = t_Input[BCn_GetPixelPos(blockPos, idx) + uint2(g_Const.srcLeft, g_Const.srcTop)];
        colors4[idx] = color.rgba;
        colors3[idx] = color.rgb;
        if (color.a < 1.0)
            opaqueBlock = false;
    }
}

[numthreads(BLOCK_COMPRESS_BC7_CS_GROUP_WIDTH, BLOCK_COMPRESS_BC7_CS_GROUP_HEIGHT, 1)]
void main(uint2 threadIdx : SV_GroupThreadID, uint2 globalIdx : SV_DispatchThreadID)
{
#if BC7_SINGLE_MODE
    // Clear mode counts while all threads are still active
    if (threadIdx.y == 0)
    {
        if (threadIdx.x < BC7_MODES)
        {
            s_ModeCounts[threadIdx.x] = 0;
        }
        if (threadIdx.x == 0)
        {
            s_ActiveThreadCount = 0;
        }
    }
    GroupMemoryBarrierWithGroupSync();
#endif

    if (globalIdx.x >= g_Const.widthInBlocks || globalIdx.y >= g_Const.heightInBlocks)
        return;

    float3 colors3[PIXELS_PER_BLOCK];
    float4 colors4[PIXELS_PER_BLOCK];
    bool opaqueBlock = true;

    uint2 blockPos = globalIdx;
    float error = FLT_MAX;
    uint4 block = 0;

#if BC7_SINGLE_MODE
    uint mode, partition;
    if (!LoadModeAndPartition(blockPos, mode, partition))
    {
        // Integration error - write a default block containing 4 colored 2x2 squares (black, red, green, blue)
        block.x = 0x00f80004;
        block.y = 0x0001f000;
        block.z = 0x3ff0001f;
        block.w = 0x1f9e0000;
        u_Output[blockPos] = block;
        return;
    }

    ReorderThreadsByMode(threadIdx, blockPos, mode, partition);

    LoadColorsForBlock(blockPos, colors4, colors3, opaqueBlock);
    
    switch(mode)
    {
        case 0:
        case 2:
            BC7_Mode0_2_Compress(colors3, mode, partition, error, block);
            break;
        case 1:
        case 3:
            BC7_Mode1_3_Compress(colors3, mode, partition, error, block);
            break;
        case 4:
        case 5:
            BC7_Mode45_Compress_SingleMode(colors4, mode, partition, error, block);
            break;
        case 6:
            BC7_Mode6_Compress(colors4, error, block);
            break;
        case 7:
            BC7_Mode7_Compress(colors4, partition, error, block);
            break;
    }
#else
    LoadColorsForBlock(blockPos, colors4, colors3, opaqueBlock);
    if (opaqueBlock)
    {
        BC7_Mode0_2_Compress(colors3, 0, 0, error, block);
        BC7_Mode1_3_Compress(colors3, 0, 0, error, block);
    }
    BC7_Mode45_Compress(colors4, error, block);
    BC7_Mode6_Compress(colors4, error, block);
    BC7_Mode7_Compress(colors4, 0, error, block);
#endif

    u_Output[blockPos] = block;
}