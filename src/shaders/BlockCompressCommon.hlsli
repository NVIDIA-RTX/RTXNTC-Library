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

#ifndef BLOCK_COMPRESS_COMMON_HLSLI
#define BLOCK_COMPRESS_COMMON_HLSLI

#include "libntc/shaders/BlockCompressConstants.h"
#include "libntc/shaders/Bindings.h"
#include "BindingHelpers.hlsli"

#ifdef __cplusplus
static const NtcBlockCompressConstants g_Const;
#else
NTC_DECLARE_CBUFFER(ConstantBuffer<NtcBlockCompressConstants> g_Const, NTC_BINDING_BC_CONSTANT_BUFFER, 0);
#endif
NTC_DECLARE_SRV(Texture2D<float4> t_Input,                NTC_BINDING_BC_INPUT_TEXTURE, 0);
NTC_DECLARE_SRV(ByteAddressBuffer t_Modes,                NTC_BINDING_BC_MODE_BUFFER, 0); // Optional - BC7 only
NTC_DECLARE_UAV(RWTexture2D<OUTPUT_FORMAT> u_Output,      NTC_BINDING_BC_OUTPUT_TEXTURE, 0);

#define PIXELS_PER_BLOCK 16

#define FOREACH_PIXEL(name) [unroll] for (int name = 0; name < PIXELS_PER_BLOCK; ++name)

#define REFINEMENT_PASSES 4

static const float FLT_MAX = 3.402823466e+38F;

// Encoding table for 5-bit color components (R, B) in BC1.
// Each value contains two color endpoints that, when used with index = 2 or 3, produce the 8-bit original color component.
// The values are derived from the OMatch5 table in NVTT3 (nvdxt_gpu.cu)
static const uint c_ColorMatch5bit[256] =
{
    0x0000,0x0000,0x0100,0x0100,0x0001,0x0001,0x0001,0x0101,0x0101,0x0101,0x0201,0x0400,0x0102,0x0102,0x0102,0x0202,
    0x0202,0x0202,0x0302,0x0501,0x0203,0x0203,0x0004,0x0303,0x0303,0x0303,0x0403,0x0403,0x0403,0x0503,0x0304,0x0304,
    0x0205,0x0404,0x0404,0x0504,0x0504,0x0405,0x0405,0x0405,0x0306,0x0505,0x0505,0x0605,0x0804,0x0506,0x0506,0x0506,
    0x0606,0x0606,0x0606,0x0706,0x0905,0x0607,0x0607,0x0408,0x0707,0x0707,0x0707,0x0807,0x0807,0x0807,0x0907,0x0708,
    0x0708,0x0609,0x0808,0x0808,0x0908,0x0908,0x0809,0x0809,0x0809,0x070a,0x0909,0x0909,0x0a09,0x0c08,0x090a,0x090a,
    0x090a,0x0a0a,0x0a0a,0x0a0a,0x0b0a,0x0d09,0x0a0b,0x0a0b,0x080c,0x0b0b,0x0b0b,0x0b0b,0x0c0b,0x0c0b,0x0c0b,0x0d0b,
    0x0b0c,0x0b0c,0x0a0d,0x0c0c,0x0c0c,0x0d0c,0x0d0c,0x0c0d,0x0c0d,0x0c0d,0x0b0e,0x0d0d,0x0d0d,0x0e0d,0x100c,0x0d0e,
    0x0d0e,0x0d0e,0x0e0e,0x0e0e,0x0e0e,0x0f0e,0x110d,0x0e0f,0x0e0f,0x0c10,0x0f0f,0x0f0f,0x0f0f,0x100f,0x100f,0x100f,
    0x110f,0x0f10,0x0f10,0x0e11,0x1010,0x1010,0x1110,0x1110,0x1011,0x1011,0x1011,0x0f12,0x1111,0x1111,0x1211,0x1410,
    0x1112,0x1112,0x1112,0x1212,0x1212,0x1212,0x1312,0x1511,0x1213,0x1213,0x1014,0x1313,0x1313,0x1313,0x1413,0x1413,
    0x1413,0x1513,0x1314,0x1314,0x1215,0x1414,0x1414,0x1514,0x1514,0x1415,0x1415,0x1415,0x1316,0x1515,0x1515,0x1615,
    0x1814,0x1516,0x1516,0x1516,0x1616,0x1616,0x1616,0x1716,0x1915,0x1617,0x1617,0x1418,0x1717,0x1717,0x1717,0x1817,
    0x1817,0x1817,0x1917,0x1718,0x1718,0x1619,0x1818,0x1818,0x1918,0x1918,0x1819,0x1819,0x1819,0x171a,0x1919,0x1919,
    0x1a19,0x1c18,0x191a,0x191a,0x191a,0x1a1a,0x1a1a,0x1a1a,0x1b1a,0x1d19,0x1a1b,0x1a1b,0x181c,0x1b1b,0x1b1b,0x1b1b,
    0x1c1b,0x1c1b,0x1c1b,0x1d1b,0x1b1c,0x1b1c,0x1a1d,0x1c1c,0x1c1c,0x1d1c,0x1d1c,0x1c1d,0x1c1d,0x1c1d,0x1b1e,0x1d1d,
    0x1d1d,0x1e1d,0x1e1d,0x1d1e,0x1d1e,0x1d1e,0x1e1e,0x1e1e,0x1e1e,0x1f1e,0x1f1e,0x1e1f,0x1e1f,0x1e1f,0x1f1f,0x1f1f
};

// Same as c_ColorMatch5bit but for the 6-bit (Green) color component.
static const uint c_ColorMatch6bit[256] =
{
    0x0000,0x0100,0x0001,0x0101,0x0101,0x0201,0x0102,0x0202,0x0202,0x0302,0x0203,0x0303,0x0303,0x0403,0x0304,0x0404,
    0x0404,0x0504,0x0405,0x0505,0x0505,0x0605,0x0506,0x1100,0x0606,0x0706,0x0607,0x1002,0x0707,0x0807,0x0708,0x1103,
    0x0808,0x0908,0x0809,0x1005,0x0909,0x0a09,0x090a,0x1106,0x0a0a,0x0b0a,0x0a0b,0x1008,0x0b0b,0x0c0b,0x0b0c,0x1109,
    0x0c0c,0x0d0c,0x0c0d,0x100b,0x0d0d,0x0e0d,0x0d0e,0x110c,0x0e0e,0x0f0e,0x0e0f,0x100e,0x0f0f,0x100f,0x0e10,0x0f10,
    0x0e11,0x1010,0x1110,0x1011,0x0f12,0x1111,0x1211,0x1112,0x0e14,0x1212,0x1312,0x1213,0x0f15,0x1313,0x1413,0x1314,
    0x0e17,0x1414,0x1514,0x1415,0x0f18,0x1515,0x1615,0x1516,0x0e1a,0x1616,0x1716,0x1617,0x0f1b,0x1717,0x1817,0x1718,
    0x2113,0x1818,0x1918,0x1819,0x2015,0x1919,0x1a19,0x191a,0x2116,0x1a1a,0x1b1a,0x1a1b,0x2018,0x1b1b,0x1c1b,0x1b1c,
    0x2119,0x1c1c,0x1d1c,0x1c1d,0x201b,0x1d1d,0x1e1d,0x1d1e,0x211c,0x1e1e,0x1f1e,0x1e1f,0x201e,0x1f1f,0x201f,0x1e20,
    0x1f20,0x1e21,0x2020,0x2120,0x2021,0x1f22,0x2121,0x2221,0x2122,0x1e24,0x2222,0x2322,0x2223,0x1f25,0x2323,0x2423,
    0x2324,0x1e27,0x2424,0x2524,0x2425,0x1f28,0x2525,0x2625,0x2526,0x1e2a,0x2626,0x2726,0x2627,0x1f2b,0x2727,0x2827,
    0x2728,0x3123,0x2828,0x2928,0x2829,0x3025,0x2929,0x2a29,0x292a,0x3126,0x2a2a,0x2b2a,0x2a2b,0x3028,0x2b2b,0x2c2b,
    0x2b2c,0x3129,0x2c2c,0x2d2c,0x2c2d,0x302b,0x2d2d,0x2e2d,0x2d2e,0x312c,0x2e2e,0x2f2e,0x2e2f,0x302e,0x2f2f,0x302f,
    0x2e30,0x2f30,0x2e31,0x3030,0x3130,0x3031,0x2f32,0x3131,0x3231,0x3132,0x2e34,0x3232,0x3332,0x3233,0x2f35,0x3333,
    0x3433,0x3334,0x2e37,0x3434,0x3534,0x3435,0x2f38,0x3535,0x3635,0x3536,0x2e3a,0x3636,0x3736,0x3637,0x2f3b,0x3737,
    0x3837,0x3738,0x2e3d,0x3838,0x3938,0x3839,0x2f3e,0x3939,0x3a39,0x393a,0x3a3a,0x3a3a,0x3b3a,0x3a3b,0x3b3b,0x3b3b,
    0x3c3b,0x3b3c,0x3c3c,0x3c3c,0x3d3c,0x3c3d,0x3d3d,0x3d3d,0x3e3d,0x3d3e,0x3e3e,0x3e3e,0x3f3e,0x3e3f,0x3f3f,0x3f3f
};

template<typename T>
void swap(inout T a, inout T b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

float colorDistance(float3 c0, float3 c1)
{
    return dot(c0 - c1, c0 - c1);
}

bool BC1_IsTransparentPixel(float alpha)
{
    return (alpha <= g_Const.alphaThreshold);
}

uint2 BC1a_TransparentBlock(float4 pixels[PIXELS_PER_BLOCK], out float blockError)
{
    blockError = 0;
    FOREACH_PIXEL(idx)
    {
        if (!BC1_IsTransparentPixel(pixels[idx].a))
            blockError += dot(pixels[idx], pixels[idx]);
    }

    return uint2(0x00000000u, 0xffffffffu);
}

uint2 BCn_GetPixelPos(uint2 blockPos, uint pixelIdx)
{
    return blockPos * 4 + uint2(pixelIdx & 3, pixelIdx >> 2);
}

uint BC1_SelectBestIndex(float d[4], out float error)
{
    if (min(d[0], d[1]) < min(d[2], d[3]))
    {
        if (d[0] < d[1])
        {
            error = d[0];
            return 0;
        }
        else
        {
            error = d[1];
            return 1;
        }
    }
    else
    {
        if (d[2] < d[3])
        {
            error = d[2];
            return 2;
        }
        else
        {
            error = d[3];
            return 3;
        }
    }
}

uint BC1_ComputeIndex4(const float4 color, const float3 maxColor, const float3 minColor, out float error)
{
    float3 color2 = lerp(maxColor, minColor, 1.0f / 3.0f);
    float3 color3 = lerp(maxColor, minColor, 2.0f / 3.0f);

    float d[4];
    float alphaDistance = (1.0 - color.a) * (1.0 - color.a);
    d[0] = colorDistance(maxColor, color.rgb) + alphaDistance;
    d[1] = colorDistance(minColor, color.rgb) + alphaDistance;
    d[2] = colorDistance(color2, color.rgb) + alphaDistance;
    d[3] = colorDistance(color3, color.rgb) + alphaDistance;

    return BC1_SelectBestIndex(d, error);
}

uint BC1_ComputeIndex3(const float4 color, const float3 maxColor, const float3 minColor, out float error)
{
    if (BC1_IsTransparentPixel(color.a))
    {
        error = 0;
        return 3;
    }

    float3 color2 = lerp(maxColor, minColor, 0.5f);

    float d[4];
    float alphaDistance = (1.0 - color.a) * (1.0 - color.a);
    d[0] = colorDistance(maxColor, color.rgb) + alphaDistance;
    d[1] = colorDistance(minColor, color.rgb) + alphaDistance;
    d[2] = colorDistance(color2, color.rgb) + alphaDistance;
    d[3] = dot(color, color);

    return BC1_SelectBestIndex(d, error);
}

float3 expand565(uint w)
{
    float3 v;
    v.x = float((w >> 11) & 0x1f) * 1.0f / 31.0f;
    v.y = float((w >> 5) & 0x3f) * 1.0f / 63.0f;
    v.z = float(w & 0x1f) * 1.0f / 31.0f;
    return v;
}

float3 roundAndExpand565(float3 v, out uint w)
{
    uint x = uint(round(saturate(v.x) * 31.0f));
    uint y = uint(round(saturate(v.y) * 63.0f));
    uint z = uint(round(saturate(v.z) * 31.0f));

    w = (x << 11) | (y << 5) | z;

    v.x = float(x) * 1.0f / 31.0f;
    v.y = float(y) * 1.0f / 63.0f;
    v.z = float(z) * 1.0f / 31.0f;

    return v;
}

uint2 BC1_EncodeSingleColor(float4 pixels[PIXELS_PER_BLOCK], out float blockError)
{
    uint2 result;

    float3 averageColor = 0;
    FOREACH_PIXEL(idx)
    {
        averageColor += pixels[idx].rgb;
    }
    averageColor /= 16.0;

    int3 intColor = int3(round(averageColor * 255.f));

    uint3 match = uint3(c_ColorMatch5bit[intColor.r], c_ColorMatch6bit[intColor.g], c_ColorMatch5bit[intColor.b]);
    uint color0 = ((match.r & 0x1f) << 11) | ((match.g & 0x3f) << 5) | (match.b & 0x1f);
    uint color1 = (((match.r >> 8) & 0x1f) << 11) | (((match.g >> 8) & 0x3f) << 5) | ((match.b >> 8) & 0x1f);

    if (color0 < color1)
    {
        result.x = (color0 << 16) | color1;
        result.y = 0xffffffff;
    }
    else
    {
        result.x = (color1 << 16) | color0;
        result.y = 0xaaaaaaaa;
    }

    float3 ep0 = expand565(color0);
    float3 ep1 = expand565(color1);
    float4 encodedColor;
    encodedColor.rgb = lerp(ep0, ep1, 1.0 / 3.0);
    encodedColor.a = 1.0;

    blockError = 0;
    FOREACH_PIXEL(idx)
    {
        float4 diff = pixels[idx] - encodedColor;
        blockError += dot(diff, diff);
    }

    return result;
}

uint2 BC1_CompressFast(float4 pixels[PIXELS_PER_BLOCK], out float blockError)
{
    // compute overall min/max/sum
    float3 sums = 0;
    float3 minColor = 1;
    float3 maxColor = 0;
    FOREACH_PIXEL(idx)
    {
        sums += pixels[idx].rgb;
        minColor = min(minColor, pixels[idx].rgb);
        maxColor = max(maxColor, pixels[idx].rgb);
    }

    // Compute the matrix of covariances.
    float3 center = sums / 16.0;
    float cov_rg = 0, cov_rb = 0;
    float cov_gb = 0;
    FOREACH_PIXEL(idx)
    {
        float3 pix = pixels[idx].rgb - center;
        cov_rg += pix.r * pix.g;
        cov_rb += pix.r * pix.b;
        cov_gb += pix.g * pix.b;
    }

    // Pick one axis with nonzero extent as the reference, and swap the endpoints
    // on the other axes if the covariance between them and the reference axis is negative.
    if (minColor.r != maxColor.r)
    {
        if (cov_rg < 0) swap(minColor.g, maxColor.g);
        if (cov_rb < 0) swap(minColor.b, maxColor.b);
    }
    else if (minColor.g != maxColor.g)
    {
        if (cov_gb < 0) swap(minColor.b, maxColor.b);
    }
    
    float3 inset = (maxColor - minColor) / 16.0f;
    maxColor = saturate(maxColor - inset);
    minColor = saturate(minColor + inset);

    uint color0 = 0, color1 = 0;

    for (int pass = 0; pass < REFINEMENT_PASSES; ++pass)
    {
        float alpha2_sum = 0;
        float beta2_sum = 0;
        float alphabeta_sum = 0;
        float3 alphax_sum = 0;
        float3 betax_sum = 0;
        FOREACH_PIXEL(idx)
        {
            // compute 2-bit index
            float pixelError;
            uint index = BC1_ComputeIndex4(pixels[idx], maxColor, minColor, pixelError);
            
            // adjust min/max color
            float beta = float(index & 1);
            if (index & 2) beta = (1 + beta) / 3.0f;
            float alpha = 1.0f - beta;

            alpha2_sum += alpha * alpha;
            beta2_sum += beta * beta;
            alphabeta_sum += alpha * beta;
            alphax_sum += alpha * pixels[idx].rgb;
            betax_sum += beta * pixels[idx].rgb;
        }

        float denom = alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum;
        if (abs(denom) > 0.0001f)
        {
            float factor = 1.0f / denom;
            float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
            float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

            // Preserve 0 and 1 cases
            minColor = select(or(minColor == 0.0, minColor == 1.0), minColor, saturate(b));
            maxColor = select(or(maxColor == 0.0, maxColor == 1.0), maxColor, saturate(a));
        }

        uint newColor0, newColor1;
        maxColor = roundAndExpand565(maxColor, newColor0);
        minColor = roundAndExpand565(minColor, newColor1);
        
        if (newColor0 == color0 && newColor1 == color1)
            break;

        color0 = newColor0;
        color1 = newColor1;
    }

    // make sure that color0 > color1, otherwise the block is treated as alpha
    if (color0 == color1)
    {
        if (color0 == 0)
        {
            color0 = 1;
            maxColor = expand565(color0);
        }
        else
        {
            color1 = color0 - 1;
            minColor = expand565(color1);
        }
    }
    else if (color0 < color1)
    {
        swap(maxColor, minColor);
        swap(color0, color1);
    }

    // re-compute indices with adjusted min/max color
    uint indices = 0;
    blockError = 0;
    FOREACH_PIXEL(idx)
    {
        float pixelError;
        uint index = BC1_ComputeIndex4(pixels[idx], maxColor, minColor, pixelError);
        indices |= index << (2 * idx);
        blockError += pixelError;
    }
    
    // save the result
    uint2 result;
    result.x = (color1 << 16) | color0;
    result.y = indices;
    return result;
}

uint2 BC1a_CompressFast(float4 pixels[PIXELS_PER_BLOCK], out float blockError)
{
    // Compute overall min/max/sum excluding transparent pixels
    float3 sums = 0;
    float3 minColor = 1;
    float3 maxColor = 0;
    int opaqueCount = 0;
    FOREACH_PIXEL(idx)
    {
        if (!BC1_IsTransparentPixel(pixels[idx].a))
        {
            sums += pixels[idx].rgb;
            minColor = min(minColor, pixels[idx].rgb);
            maxColor = max(maxColor, pixels[idx].rgb);
            opaqueCount += 1;
        }
    }
    
    // Compute the matrix of covariances.
    float3 center = (opaqueCount > 0) ? sums / opaqueCount : 0;
    float cov_rg = 0, cov_rb = 0;
    float cov_gb = 0;
    FOREACH_PIXEL(idx)
    {
        if (!BC1_IsTransparentPixel(pixels[idx].a))
        {
            float3 pix = pixels[idx].rgb - center;
            cov_rg += pix.r * pix.g;
            cov_rb += pix.r * pix.b;
            cov_gb += pix.g * pix.b;
        }
    }

    // Pick one axis with nonzero extent as the reference, and swap the endpoints
    // on the other axes if the covariance between them and the reference axis is negative.
    if (minColor.r != maxColor.r)
    {
        if (cov_rg < 0) swap(minColor.g, maxColor.g);
        if (cov_rb < 0) swap(minColor.b, maxColor.b);
    }
    else if (minColor.g != maxColor.g)
    {
        if (cov_gb < 0) swap(minColor.b, maxColor.b);
    }

    float3 inset = (maxColor - minColor) / opaqueCount;
    maxColor = saturate(maxColor - inset);
    minColor = saturate(minColor + inset);

    float alpha2_sum = 0;
    float beta2_sum = 0;
    float alphabeta_sum = 0;
    float3 alphax_sum = 0;
    float3 betax_sum = 0;
    FOREACH_PIXEL(idx)
    {
        // compute 2-bit index
        float pixelError;
        uint index = BC1_ComputeIndex3(pixels[idx], maxColor, minColor, pixelError);

        if (index != 3)
        {  
            // adjust min/max color
            float beta = float(index & 1); // 0->0, 1->1
            if (index & 2) beta = 0.5f; // 2->0.5
            float alpha = 1.0f - beta;

            alpha2_sum += alpha * alpha;
            beta2_sum += beta * beta;
            alphabeta_sum += alpha * beta;
            alphax_sum += alpha * pixels[idx].rgb;
            betax_sum += beta * pixels[idx].rgb;
        }
    }

    float denom = alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum;
    if (abs(denom) > 0.0001f)
    {
        float factor = 1.0f / denom;
        float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
        float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

        // Preserve 0 and 1 cases
        minColor = select(or(minColor == 0.0, minColor == 1.0), minColor, saturate(a));
        maxColor = select(or(maxColor == 0.0, maxColor == 1.0), maxColor, saturate(b));
    }

    // get c16 color0/1
    uint color0, color1;
    maxColor = roundAndExpand565(maxColor, color0);
    minColor = roundAndExpand565(minColor, color1);
    if (color0 > color1)
    {
        swap(maxColor, minColor);
        swap(color0, color1);
    }

    // re-compute indices with adjusted min/max color
    uint indices = 0;
    blockError = 0;
    FOREACH_PIXEL(idx)
    {
        float pixelError;
        uint index = BC1_ComputeIndex3(pixels[idx], maxColor, minColor, pixelError);
        indices |= index << (2 * idx);

        if (index != 3 || !BC1_IsTransparentPixel(pixels[idx].a))
            blockError += pixelError;
    }

    // save the result
    uint2 result;
    result.x = (color1 << 16) | color0;
    result.y = indices;
    return result;
}

uint2 BC2_EncodeAlpha(float alphas[PIXELS_PER_BLOCK])
{
    uint2 block = 0;
    FOREACH_PIXEL(idx)
    {
        uint value = uint(round(saturate(alphas[idx]) * 15.0));
        if (idx < 8)
            block.x |= value << (idx * 4);
        else
            block.y |= value << ((idx - 8) * 4);
    }
    
    return block;
}

void BC4_setAlpha0(inout uint2 block, uint value)
{
    block.x = (block.x & 0xffffff00) | (value & 0xff);
}

void BC4_setAlpha1(inout uint2 block, uint value)
{
    block.x = (block.x & 0xffff00ff) | ((value & 0xff) << 8);
}

uint BC4_getAlpha0(uint2 block)
{
    return block.x & 0xff;
}

uint BC4_getAlpha1(uint2 block)
{
    return (block.x >> 8)  & 0xff;
}

void BC4_clearIndices(inout uint2 block)
{
    block.x &= 0xffff;
    block.y = 0;
}

void BC4_setIndex(inout uint2 block, uint pixelIdx, uint paletteIdx)
{
    uint offset = 16 + pixelIdx * 3;
    paletteIdx &= 7;
    // Insert 3 bits into a 64-bit word (block.x, block.y)
    // Note: the conditions are necessary because shift operators wrap the shift amount over 32
    if (offset < 32)      block.x |= paletteIdx << offset;
    if (61 - offset < 32) block.y |= (paletteIdx << 29) >> (61 - offset);
}

uint BC4_getIndex(uint2 block, uint pixelIdx)
{
    uint offset = 16 + pixelIdx * 3;
    uint paletteIdx = 0;
    // Extract 3 bits from a 64-bit word (block.x, block.y)
    // Note: the conditions are necessary because shift operators wrap the shift amount over 32
    if (offset < 32)      paletteIdx = (block.x >> offset);
    if (61 - offset < 32) paletteIdx |= ((block.y << (61 - offset)) >> 29);
    return paletteIdx & 7;
}

bool BC4_sameIndices(uint2 block0, uint2 block1)
{
    return (block0.x & 0xffff0000) == (block1.x & 0xffff0000)
        && block0.y == block1.y;
}

void BC4_EvaluatePalette8(uint alpha0, uint alpha1, out uint alphas[8])
{
    int bias = 3;

    // 8-alpha block:  derive the other six alphas.
    // Bit code 000 = alpha0, 001 = alpha1, others are interpolated.
    alphas[0] = alpha0;
    alphas[1] = alpha1;
    alphas[2] = (6 * alpha0 + 1 * alpha1 + bias) / 7;    // bit code 010
    alphas[3] = (5 * alpha0 + 2 * alpha1 + bias) / 7;    // bit code 011
    alphas[4] = (4 * alpha0 + 3 * alpha1 + bias) / 7;    // bit code 100
    alphas[5] = (3 * alpha0 + 4 * alpha1 + bias) / 7;    // bit code 101
    alphas[6] = (2 * alpha0 + 5 * alpha1 + bias) / 7;    // bit code 110
    alphas[7] = (1 * alpha0 + 6 * alpha1 + bias) / 7;    // bit code 111
}

void BC4_EvaluatePalette6(uint alpha0, uint alpha1, out uint alphas[8])
{
    int bias = 2;

    // 6-alpha block.
    // Bit code 000 = alpha0, 001 = alpha1, others are interpolated.
    alphas[0] = alpha0;
    alphas[1] = alpha1;
    alphas[2] = (4 * alpha0 + 1 * alpha1 + bias) / 5;    // Bit code 010
    alphas[3] = (3 * alpha0 + 2 * alpha1 + bias) / 5;    // Bit code 011
    alphas[4] = (2 * alpha0 + 3 * alpha1 + bias) / 5;    // Bit code 100
    alphas[5] = (1 * alpha0 + 4 * alpha1 + bias) / 5;    // Bit code 101
    alphas[6] = 0x00;                                    // Bit code 110
    alphas[7] = 0xFF;                                    // Bit code 111
}

void BC4_EvaluatePalette(uint alpha0, uint alpha1, out uint alphas[8])
{
    if (alpha0 > alpha1) {
        BC4_EvaluatePalette8(alpha0, alpha1, alphas);
    }
    else {
        BC4_EvaluatePalette6(alpha0, alpha1, alphas);
    }
}

float BC4_ComputeAlphaIndices(uint values[PIXELS_PER_BLOCK], inout uint2 block)
{
    uint alphas[8];
    BC4_EvaluatePalette(BC4_getAlpha0(block), BC4_getAlpha1(block), alphas);
    BC4_clearIndices(block);
    
    float totalError = 0;
    FOREACH_PIXEL(idx)
    {
        int minDist = 0x7fffffff; // INT_MAX
        uint best = 8;
        [unroll]
        for (uint p = 0; p < 8; p++)
        {
            int dist = int(values[idx]) - int(alphas[p]);
            dist *= dist;
            if (dist < minDist)
            {
                minDist = dist;
                best = p;
            }
        }
        totalError += minDist;
        BC4_setIndex(block, idx, best);
    }
    
    return totalError;
}

void BC4_OptimizeAlpha8(uint values[PIXELS_PER_BLOCK], inout uint2 block, bool zeroAlpha1 = false, bool oneAlpha0 = false)
{
    float alpha2_sum = 0;
    float beta2_sum = 0;
    float alphabeta_sum = 0;
    float alphax_sum = 0;
    float betax_sum = 0;

    FOREACH_PIXEL(idx)
    {
        uint paletteIdx = BC4_getIndex(block, idx);
        float alpha;
        if (paletteIdx < 2) alpha = 1.0f - paletteIdx;
        else alpha = (8.0f - paletteIdx) / 7.0f;
        float beta = 1 - alpha;
        
        alpha2_sum += alpha * alpha;
        beta2_sum += beta * beta;
        alphabeta_sum += alpha * beta;
        alphax_sum += alpha * values[idx];
        betax_sum += beta * values[idx];
    }

    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

    uint alpha0 = oneAlpha0 ? 255 : uint(round(clamp(a, 0.0f, 255.0f)));
    uint alpha1 = zeroAlpha1 ? 0 : uint(round(clamp(b, 0.0f, 255.0f)));

    if (alpha0 < alpha1)
        swap(alpha0, alpha1);

    BC4_setAlpha0(block, alpha0);
    BC4_setAlpha1(block, alpha1);
}

void BC4_OptimizeAlpha6(uint values[PIXELS_PER_BLOCK], inout uint2 block)
{
    float alpha2_sum = 0;
    float beta2_sum = 0;
    float alphabeta_sum = 0;
    float alphax_sum = 0;
    float betax_sum = 0;
    
    FOREACH_PIXEL(idx)
    {
        uint paletteIdx = BC4_getIndex(block, idx);
        float alpha;
        if (paletteIdx < 2)
            alpha = 1.0f - paletteIdx;
        else if (paletteIdx < 6)
            alpha = (6.0f - paletteIdx) / 5.0f;
        else
            continue;

        float beta = 1.0f - alpha;

        alpha2_sum += alpha * alpha;
        beta2_sum += beta * beta;
        alphabeta_sum += alpha * beta;
        alphax_sum += alpha * values[idx];
        betax_sum += beta * values[idx];
    }
    
    const float factor = alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum;
    if (factor == 0.0f) return;

    float a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) / factor;
    float b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) / factor;

    BC4_setAlpha0(block, uint(round(clamp(a, 0.0f, 255.0f))));
    BC4_setAlpha1(block, uint(round(clamp(b, 0.0f, 255.0f))));
}

uint2 BC4_CompressFast(float pixels[PIXELS_PER_BLOCK])
{
    uint values[PIXELS_PER_BLOCK];
    uint mina = 255;
    uint maxa = 0;
    uint mina_no01 = 255;
    uint maxa_no01 = 0;
    FOREACH_PIXEL(idx)
    {
        values[idx] = uint(round(saturate(pixels[idx]) * 255.0));

        // Get min/max alpha.
        mina = min(mina, values[idx]);
        maxa = max(maxa, values[idx]);

        // Get min/max alpha that is not 0 or 1.
        if (values[idx] != 0) mina_no01 = min(mina_no01, values[idx]);
        if (values[idx] != 255) maxa_no01 = max(maxa_no01, values[idx]);
    }
    
    bool zeroAlpha1 = mina == 0;
    bool oneAlpha0 = maxa == 255;

    uint2 block = 0;
    BC4_setAlpha0(block, oneAlpha0 ? 255 : uint(maxa - (maxa - mina) / 34));
    BC4_setAlpha1(block, zeroAlpha1 ? 0 : uint(mina + (maxa - mina) / 34));
    
    float besterror = BC4_ComputeAlphaIndices(values, block);

    uint2 bestblock = block;
    
    for (int i = 0; i < 8; i++)
    {
        BC4_OptimizeAlpha8(values, block, zeroAlpha1, oneAlpha0);
        float error = BC4_ComputeAlphaIndices(values, block);

        if (error >= besterror)
        {
            // No improvement, stop.
            break;
        }

        if (BC4_sameIndices(block, bestblock))
        {
            bestblock = block;
            break;
        }

        bestblock = block;
        besterror = error;
    }

    BC4_setAlpha0(block, mina_no01 + (maxa_no01 - mina_no01) / 34);
    BC4_setAlpha1(block, maxa_no01 - (maxa_no01 - mina_no01) / 34);
    float besterror2 = BC4_ComputeAlphaIndices(values, block);

    uint2 bestblock2 = block;

    for (int i = 0; i < 8; i++)
    {
        BC4_OptimizeAlpha6(values, block);
        float error = BC4_ComputeAlphaIndices(values, block);

        if (error >= besterror2)
        {
            // No improvement, stop.
            break;
        }

        if (BC4_sameIndices(block, bestblock2))
        {
            bestblock2 = block;
            break;
        }

        bestblock2 = block;
        besterror2 = error;
    }

    if (besterror2 < besterror)
    {
        bestblock = bestblock2;
    }

    return bestblock;
}

#endif // BLOCK_COMPRESS_COMMON_HLSLI