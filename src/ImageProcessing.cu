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

#include "CudaUtils.h"
#include "ImageProcessing.h"
#include "tin/tin_common.h"
#include "tin/tin_reducer.h"

namespace ntc::cuda
{

// Computes the top-left pixel position for the bilinear filter footprint needed for image resize
// and the fitler weights for the 4 pixels (x: TL, y: TR, z: BL, w: BR)
// Returns true if the input position is valid.
__device__ inline bool SetupBilinearFilter(
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    bool verticalFlip,
    uint2 dstPos,
    uint2& outSrcPos,
    float4& outSrcWeights)
{
    if (int(dstPos.x) >= dstWidth || int(dstPos.y) >= dstHeight)
        return false;

    if (verticalFlip)
        dstPos.y = uint32_t(dstHeight) - dstPos.y - 1;

    if (srcWidth == dstWidth && srcHeight == dstHeight)
    {
        // Shortcut for 1:1 copies, also to make sure that there is no numerical error.
        outSrcPos = dstPos;
        outSrcWeights = { 1.f, 0.f, 0.f, 0.f };
        return true;
    }

    float2 outputUv;
    outputUv.x = (float(dstPos.x) + 0.5f) / float(dstWidth);
    outputUv.y = (float(dstPos.y) + 0.5f) / float(dstHeight);

    float2 inputPosition;
    inputPosition.x = outputUv.x * float(srcWidth) - 0.5f;
    inputPosition.y = outputUv.y * float(srcHeight) - 0.5f;
    inputPosition.x = std::max(0.f, std::min(float(srcWidth) - 1.f, inputPosition.x));
    inputPosition.y = std::max(0.f, std::min(float(srcHeight) - 1.f, inputPosition.y));

    outSrcPos.x = __float2uint_rd(inputPosition.x);
    outSrcPos.y = __float2uint_rd(inputPosition.y);

    float2 inputFraction;
    inputFraction.x = inputPosition.x - floorf(inputPosition.x);
    inputFraction.y = inputPosition.y - floorf(inputPosition.y);

    outSrcWeights.x = (1.f - inputFraction.x) * (1.f - inputFraction.y);
    outSrcWeights.y = inputFraction.x * (1.f - inputFraction.y);
    outSrcWeights.z = (1.f - inputFraction.x) * inputFraction.y;
    outSrcWeights.w = inputFraction.x * inputFraction.y;

    return true;
}

inline __device__ uint8_t* GetPitchLinearChannelGroupPtr(PitchLinearImageSlice slice, uint32_t x, uint32_t y, uint32_t channel)
{
    return slice.pData 
        + y * uint64_t(slice.rowPitch)
        + x * uint64_t(slice.pixelStride)
        + ((channel + slice.firstChannel) >> slice.logChannelGroupSize) * uint64_t(slice.channelGroupStride);
}

inline __device__ uint32_t GetPitchLinearChannelInGroup(PitchLinearImageSlice slice, uint32_t channel)
{
    return (channel + slice.firstChannel) & ((1u << slice.logChannelGroupSize) - 1u);
}

__device__ float DecodeColorSpace(ColorSpace colorSpace, float value)
{
    switch (colorSpace)
    {
        case ColorSpace::sRGB:
            return sRGB::Decode(value);
        case ColorSpace::HLG:
            return HLG::Decode(value);
        case ColorSpace::Linear:
        default:
            return value;
    }
}

__device__ float EncodeColorSpace(ColorSpace colorSpace, float value)
{
    switch (colorSpace)
    {
        case ColorSpace::sRGB:
            return sRGB::Encode(value);
        case ColorSpace::HLG:
            return HLG::Encode(value);
        case ColorSpace::Linear:
        default:
            return value;
    }
}

__device__ float ReadImageChannel(const uint8_t* src, uint32_t ch, ntc::ChannelFormat format, ColorSpace colorSpace)
{
    float value = 0;
    switch (format)
    {
        case ntc::ChannelFormat::UNORM8:
            value = float(src[ch]) / 255.f;
            break;
        case ntc::ChannelFormat::UNORM16:
            value = float(((uint16_t*)src)[ch]) / 65535.f;
            break;
        case ntc::ChannelFormat::FLOAT16:
            value = ((const half*)src)[ch];
            break;
        case ntc::ChannelFormat::FLOAT32:
            value = ((const float*)src)[ch];
            break;
        case ntc::ChannelFormat::UINT32:
            value = float(((const uint32_t*)src)[ch]);
            break;
        default:
            return 0;
    }
    value = DecodeColorSpace(colorSpace, value);
    return value;
}

__device__ void WriteImageChannel(uint8_t* dst, uint32_t ch, ntc::ChannelFormat format, ColorSpace colorSpace, float value, float dither /* [-0.5, 0.5) */)
{
    value = EncodeColorSpace(colorSpace, value);
    switch (format)
    {
        case ntc::ChannelFormat::UNORM8:
            value += dither / 255.f;
            dst[ch] = uint8_t(std::max(0.f, std::min(255.f, roundf(value * 255.f))));
            break;
        case ntc::ChannelFormat::UNORM16:
            value += dither / 65535.f;
            ((uint16_t*)dst)[ch] = uint16_t(std::max(0.f, std::min(65535.f, roundf(value * 65535.f))));
            break;
        case ntc::ChannelFormat::FLOAT16:
            ((half*)dst)[ch] = std::max(-65504.f, std::min(65504.f, value));
            break;
        case ntc::ChannelFormat::FLOAT32:
            ((float*)dst)[ch] = value;
            break;
        case ntc::ChannelFormat::UINT32:
            ((uint32_t*)dst)[ch] = uint32_t(std::max(value, 0.f));
            break;
    }
}

__device__ float GetDefaultChannelValue(int channel)
{
    // Default values: alpha channel at 1.0, other channels at 0.0
    return channel == 3 ? 1.f : 0.f;
}

__constant__ ColorSpace g_ChannelColorSpaces[NTC_MAX_CHANNELS];

__global__ void ResizeMultichannelImageKernel(
    PitchLinearImageSlice src,
    PitchLinearImageSlice dst)
{
    uint2 outputPixel;
    outputPixel.x = blockIdx.x * blockDim.x + threadIdx.x;
    outputPixel.y = blockIdx.y * blockDim.y + threadIdx.y;

    uint2 inputPixel;
    float4 inputWeights;
    if (!SetupBilinearFilter(src.width, src.height, dst.width, dst.height, false, outputPixel, inputPixel, inputWeights))
        return;

    for (uint32_t ch = 0; ch < uint32_t(dst.channels); ++ch)
    {
        uint8_t const* const srcData = GetPitchLinearChannelGroupPtr(src, inputPixel.x, inputPixel.y, ch);
        uint32_t const srcChannelInGroup = GetPitchLinearChannelInGroup(src, ch);

        ColorSpace colorSpace = g_ChannelColorSpaces[ch];
        float a = inputWeights.x > 0.f ? ReadImageChannel(srcData, srcChannelInGroup, ntc::ChannelFormat::FLOAT16, colorSpace) : 0.f;
        float b = inputWeights.y > 0.f ? ReadImageChannel(srcData + src.pixelStride, srcChannelInGroup, ntc::ChannelFormat::FLOAT16, colorSpace) : 0.f;
        float c = inputWeights.z > 0.f ? ReadImageChannel(srcData + src.rowPitch, srcChannelInGroup, ntc::ChannelFormat::FLOAT16, colorSpace) : 0.f;
        float d = inputWeights.w > 0.f ? ReadImageChannel(srcData + src.rowPitch + src.pixelStride, srcChannelInGroup, ntc::ChannelFormat::FLOAT16, colorSpace) : 0.f;
        
        float value = a * inputWeights.x + b * inputWeights.y + c * inputWeights.z + d * inputWeights.w;
        
        uint8_t* const dstData = GetPitchLinearChannelGroupPtr(dst, outputPixel.x, outputPixel.y, ch);
        uint32_t const dstChannelInGroup = GetPitchLinearChannelInGroup(dst, ch);

        WriteImageChannel(dstData, dstChannelInGroup, ntc::ChannelFormat::FLOAT16, colorSpace, value, 0.f);
    }
}

void ResizeMultichannelImage(
    PitchLinearImageSlice src,
    PitchLinearImageSlice dst,
    std::array<ColorSpace, NTC_MAX_CHANNELS> const& channelColorSpaces)
{
    dim3 threadBlockSize(16, 16, 1);
    dim3 gridSize = DivRoundUp(dim3(dst.width, dst.height, 1), threadBlockSize);

    cudaMemcpyToSymbol(g_ChannelColorSpaces, channelColorSpaces.data(), sizeof(ColorSpace) * NTC_MAX_CHANNELS);

    ResizeMultichannelImageKernel <<< gridSize, threadBlockSize >>> (src, dst);
}

__global__ void CopyImageKernel(
    PitchLinearImageSlice src,
    PitchLinearImageSlice dst,
    bool useDithering,
    bool verticalFlip)
{
    uint2 outputPixel;
    outputPixel.x = blockIdx.x * blockDim.x + threadIdx.x;
    outputPixel.y = blockIdx.y * blockDim.y + threadIdx.y;

    uint2 inputPixel;
    float4 inputWeights;
    if (!SetupBilinearFilter(src.width, src.height, dst.width, dst.height, verticalFlip, outputPixel, inputPixel, inputWeights))
        return;

    HashBasedRNG rng(outputPixel.x + outputPixel.y * dst.width, 0);

    for (uint32_t ch = 0; ch < uint32_t(dst.channels); ++ch)
    {
        float value;
        if (ch < uint32_t(src.channels))
        {
            uint8_t const* const srcData = GetPitchLinearChannelGroupPtr(src, inputPixel.x, inputPixel.y, ch);
            uint32_t const srcChannelInGroup = GetPitchLinearChannelInGroup(src, ch);
            ColorSpace const srcColorSpace = src.channelColorSpaces[ch];

            const float a = inputWeights.x > 0.f ? ReadImageChannel(srcData, srcChannelInGroup, src.format, srcColorSpace) : 0.f;
            const float b = inputWeights.y > 0.f ? ReadImageChannel(srcData + src.pixelStride, srcChannelInGroup, src.format, srcColorSpace) : 0.f;
            const float c = inputWeights.z > 0.f ? ReadImageChannel(srcData + src.rowPitch, srcChannelInGroup, src.format, srcColorSpace) : 0.f;
            const float d = inputWeights.w > 0.f ? ReadImageChannel(srcData + src.rowPitch + src.pixelStride, srcChannelInGroup, src.format, srcColorSpace) : 0.f;

            value = a * inputWeights.x + b * inputWeights.y + c * inputWeights.z + d * inputWeights.w;
        }
        else
        {
            value = GetDefaultChannelValue(ch);
        }

        uint8_t* const dstData = GetPitchLinearChannelGroupPtr(dst, outputPixel.x, outputPixel.y, ch);
        uint32_t const dstChannelInGroup = GetPitchLinearChannelInGroup(dst, ch);
        float const dither = useDithering ? rng.NextFloat() - 0.5f : 0.f;
        ColorSpace const dstColorSpace = dst.channelColorSpaces[ch];

        WriteImageChannel(dstData, dstChannelInGroup, dst.format, dstColorSpace, value, dither);
    }
}

void CopyImage(
    PitchLinearImageSlice src,
    PitchLinearImageSlice dst,
    bool useDithering,
    bool verticalFlip)
{
    dim3 threadBlockSize(16, 16, 1);
    dim3 gridSize = DivRoundUp(dim3(dst.width, dst.height, 1), threadBlockSize);
    
    CopyImageKernel <<< gridSize, threadBlockSize >>> (src, dst, useDithering, verticalFlip);
}

__device__ inline void WritePixelToSurface(
    cudaSurfaceObject_t surface,
    uint2 position,
    int pixelStride,
    uint4 data)
{
    switch (pixelStride)
    {
        case 1:
            surf2Dwrite<uint8_t>(data.x, surface, position.x * pixelStride, position.y);
            break;
        case 2:
            surf2Dwrite<uint16_t>(data.x, surface, position.x * pixelStride, position.y);
            break;
        case 4:
            surf2Dwrite<uint32_t>(data.x, surface, position.x * pixelStride, position.y);
            break;
        case 8:
            surf2Dwrite<uint2>(make_uint2(data.x, data.y), surface, position.x * pixelStride, position.y);
            break;
        case 16:
            surf2Dwrite<uint4>(data, surface, position.x * pixelStride, position.y);
            break;
        default:
            // unsupported - should not happen - do nothing to prevent a MisalignedAddress exception
            break;
    }
}

__device__ inline uint4 ReadPixelFromSurface(
    cudaSurfaceObject_t surface,
    uint2 position,
    int pixelStride)
{
    uint4 result{};

    position.x *= pixelStride;

    switch (pixelStride)
    {
        case 1:
            result.x = surf2Dread<uint8_t>(surface, position.x, position.y);
            break;
        case 2:
            result.x = surf2Dread<uint16_t>(surface, position.x, position.y);
            break;
        case 4:
            result.x = surf2Dread<uint32_t>(surface, position.x, position.y);
            break;
        case 8:
        {
            uint2 data2 = surf2Dread<uint2>(surface, position.x, position.y);
            result.x = data2.x;
            result.y = data2.y;
            break;
        }
        case 16:
            result = surf2Dread<uint4>(surface, position.x, position.y);
            break;
        default:
            // unsupported - should not happen - do nothing to prevent a MisalignedAddress exception
            break;
    }

    return result;
}

__device__ inline uint32_t GetFormatBitsPerChannel(ChannelFormat format)
{
    switch (format)
    {
        case ChannelFormat::UNORM8:
            return 8;
        case ChannelFormat::UNORM16:
        case ChannelFormat::FLOAT16:
            return 16;
        case ChannelFormat::FLOAT32:
        case ChannelFormat::UINT32:
            return 32;
        default:
            return 0;
    }
}

__device__ inline void EncodeSurfaceChannel(
    uint4& data,
    uint32_t channel,
    uint32_t bitsPerChannel,
    float value,
    ChannelFormat format,
    ColorSpace colorSpace,
    float dither /* [-0.5, 0.5) */
)
{
    value = EncodeColorSpace(colorSpace, value);

    // Convert the value to all supported formats because surf2Dwrite doesn't do any format conversions

    uint32_t encoded = 0;
    switch (format)
    {
        case ChannelFormat::UNORM8:
            value += dither / 255.f;
            encoded = uint32_t(std::max(0.f, std::min(255.f, roundf(value * 255.f))));
            break;
        case ChannelFormat::UNORM16:
            value += dither / 65535.f;
            encoded = uint32_t(std::max(0.f, std::min(65535.f, roundf(value * 65535.f))));
            break;
        case ChannelFormat::FLOAT16:
            encoded = __half_as_ushort(half(std::max(-65504.f, std::min(65504.f, value))));
            break;
        case ChannelFormat::FLOAT32:
            encoded = __float_as_uint(value);
            break;
        case ChannelFormat::UINT32:
            encoded = uint32_t(std::max(value, 0.f));
            break;
    }

    // Insert the encoded value into the output uint4 at the right offset

    const uint32_t offset = channel * bitsPerChannel;
    encoded = encoded << (offset & 31);

    switch (offset >> 5)
    {
        case 0:  data.x |= encoded; break;
        case 1:  data.y |= encoded; break;
        case 2:  data.z |= encoded; break;
        default: data.w |= encoded; break;
    }
}

__device__ inline float DecodeSurfaceChannel(
    uint4 data,
    uint32_t channel,
    uint32_t bitsPerChannel,
    ChannelFormat pixelFormat,
    ColorSpace colorSpace)
{
    // Extract the encoded value from the input uint4 at the right offset

    const uint32_t offset = channel * bitsPerChannel;
    uint32_t encoded;

    switch (offset >> 5)
    {
        case 0:  encoded = data.x; break;
        case 1:  encoded = data.y; break;
        case 2:  encoded = data.z; break;
        default: encoded = data.w; break;
    }

    encoded = (encoded >> (offset & 31)) & ((1 << bitsPerChannel) - 1);

    // Convert the value from all supported formats

    float value = 0;
    switch (pixelFormat)
    {
        case ChannelFormat::UNORM8:
            value = float(encoded) / 255.f;
            break;
        case ChannelFormat::UNORM16:
            value = float(encoded) / 65535.f;
            break;
        case ChannelFormat::FLOAT16:
            value = __ushort_as_half(uint16_t(encoded));
            break;
        case ChannelFormat::FLOAT32:
            value = __uint_as_float(encoded);
            break;
        case ChannelFormat::UINT32:
            value = float(encoded);
            break;
        default:
            return 0.f;
    }

    value = DecodeColorSpace(colorSpace, value);
    return value;
}

__global__ void CopyImageToSurfaceKernel(
    PitchLinearImageSlice src,
    SurfaceInfo dst,
    bool useDithering,
    bool verticalFlip)
{
    uint2 outputPixel;
    outputPixel.x = blockIdx.x * blockDim.x + threadIdx.x;
    outputPixel.y = blockIdx.y * blockDim.y + threadIdx.y;

    uint2 inputPixel;
    float4 inputWeights;
    if (!SetupBilinearFilter(src.width, src.height, dst.width, dst.height, verticalFlip, outputPixel, inputPixel, inputWeights))
        return;
    
    const uint32_t bits = GetFormatBitsPerChannel(dst.format);

    HashBasedRNG rng(outputPixel.x + outputPixel.y * dst.width, 0);
    
    uint4 data {};

    for (int ch = 0; ch < dst.channels; ++ch)
    {
        float value;
        
        if (ch < src.channels)
        { 
            uint8_t const* const srcData = GetPitchLinearChannelGroupPtr(src, inputPixel.x, inputPixel.y, ch);
            uint32_t const srcChannelInGroup = GetPitchLinearChannelInGroup(src, ch);
            ColorSpace const srcColorSpace = src.channelColorSpaces[ch];

            float4 inputValues;
            inputValues.x = inputWeights.x > 0.f ? ReadImageChannel(srcData, srcChannelInGroup, src.format, srcColorSpace) : 0.f;
            inputValues.y = inputWeights.y > 0.f ? ReadImageChannel(srcData + src.pixelStride, srcChannelInGroup, src.format, srcColorSpace) : 0.f;
            inputValues.z = inputWeights.z > 0.f ? ReadImageChannel(srcData + src.rowPitch, srcChannelInGroup, src.format, srcColorSpace) : 0.f;
            inputValues.w = inputWeights.w > 0.f ? ReadImageChannel(srcData + src.rowPitch + src.pixelStride, srcChannelInGroup, src.format, srcColorSpace) : 0.f;

            value =
                inputValues.x * inputWeights.x +
                inputValues.y * inputWeights.y +
                inputValues.z * inputWeights.z +
                inputValues.w * inputWeights.x;
        }
        else
        {
            value = GetDefaultChannelValue(ch);
        }

        float const dither = useDithering ? rng.NextFloat() - 0.5f : 0.f;
        ColorSpace const dstColorSpace = (ch < 3) ? dst.rgbColorSpace : dst.alphaColorSpace;

        EncodeSurfaceChannel(data, ch, bits, value, dst.format, dstColorSpace, dither);
    }

    // Write out the necessary amount of data for this pixel

    WritePixelToSurface(dst.surface, outputPixel, dst.pixelStride, data);
}

void CopyImageToSurface(
    PitchLinearImageSlice src,
    SurfaceInfo dst,
    bool useDithering,
    bool verticalFlip)
{
    dim3 threadBlockSize(16, 16, 1);
    dim3 gridSize = DivRoundUp(dim3(dst.width, dst.height, 1), threadBlockSize);

    CopyImageToSurfaceKernel <<< gridSize, threadBlockSize >>> (src, dst, useDithering, verticalFlip);
}

__global__ void CopySurfaceToImageKernel(
    SurfaceInfo src,
    PitchLinearImageSlice dst,
    bool verticalFlip)
{
    uint2 outputPixel;
    outputPixel.x = blockIdx.x * blockDim.x + threadIdx.x;
    outputPixel.y = blockIdx.y * blockDim.y + threadIdx.y;

    uint2 inputPixel;
    float4 inputWeights;
    if (!SetupBilinearFilter(src.width, src.height, dst.width, dst.height, verticalFlip, outputPixel, inputPixel, inputWeights))
        return;
    
    // Read the necessary amount of data for this pixel

    uint4 srcPixelValues[4]{};
#pragma unroll
    for (int sample = 0; sample < 4; ++sample)
    {
        uint2 srcPos;
        srcPos.x = inputPixel.x + (sample & 1);
        srcPos.y = inputPixel.y + (sample >> 1);

        if (int(srcPos.x) >= src.width || int(srcPos.y) >= src.height)
            continue;

        srcPixelValues[sample] = ReadPixelFromSurface(src.surface, srcPos, src.pixelStride);
    }
    
    const uint32_t bits = GetFormatBitsPerChannel(src.format);

    for (int ch = 0; ch < dst.channels; ++ch)
    {
        float value;

        if (ch < src.channels)
        {
            // Decode the channel from the 4 source pixels
            ColorSpace const srcColorSpace = (ch < 3) ? src.rgbColorSpace : src.alphaColorSpace;

            float srcChannelValues[4]{};
            #pragma unroll
            for (int sample = 0; sample < 4; ++sample)
            {
                srcChannelValues[sample] = DecodeSurfaceChannel(srcPixelValues[sample], ch, bits, src.format, srcColorSpace);
            }

            // Apply the bilinear filter

            value = 
                srcChannelValues[0] * inputWeights.x + 
                srcChannelValues[1] * inputWeights.y +
                srcChannelValues[2] * inputWeights.z +
                srcChannelValues[3] * inputWeights.w;
        }
        else
        {
            value = GetDefaultChannelValue(ch);
        }

        // Write the output
        
        uint8_t* const dstData = GetPitchLinearChannelGroupPtr(dst, outputPixel.x, outputPixel.y, ch);
        uint32_t const dstChannelInGroup = GetPitchLinearChannelInGroup(dst, ch);
        ColorSpace const dstColorSpace = dst.channelColorSpaces[ch];

        WriteImageChannel(dstData, dstChannelInGroup, dst.format, dstColorSpace, value, /* dither = */ 0.f);
    }
}

void CopySurfaceToImage(
    SurfaceInfo src,
    PitchLinearImageSlice dst,
    bool verticalFlip)
{
    dim3 threadBlockSize(16, 16, 1);
    dim3 gridSize = DivRoundUp(dim3(dst.width, dst.height, 1), threadBlockSize);
    
    CopySurfaceToImageKernel <<< gridSize, threadBlockSize >>> (src, dst, verticalFlip);
}


static constexpr uint32_t ReduceGroupWidth = 16;
static constexpr uint32_t ReduceGroupHeight = 16;

struct HalfAsInt
{
    static constexpr float Scale = 0x80000000u / 0x10000u;

    static __device__ int Encode(float value, bool roundUp)
    {
        constexpr float MaxHalf = 65504.f;
        value = std::max(-MaxHalf, std::min(MaxHalf, value));
        
        float const scaledValue = value * Scale;
        int mappedValue = roundUp ? int(ceilf(scaledValue)) : int(floorf(scaledValue));
        
        return mappedValue;
    }

    static float Decode(int value)
    {
        return float(value) / float(Scale);
    }
};

__global__ void InitMinMaxChannelValuesKernel(int* __restrict__ mins, int* __restrict__ maxs)
{
    mins[threadIdx.x] = INT_MAX;
    maxs[threadIdx.x] = INT_MIN;
}

__global__ void ComputeMinMaxChannelValuesKernel(
    PitchLinearImageSlice image,
    int* __restrict__ mins,
    int* __restrict__ maxs)
{
    int2 inputPixel;
    inputPixel.x = blockIdx.x * blockDim.x + threadIdx.x;
    inputPixel.y = blockIdx.y * blockDim.y + threadIdx.y;
    
    inputPixel.x = std::min(inputPixel.x, image.width - 1);
    inputPixel.y = std::min(inputPixel.y, image.height - 1);

    constexpr uint32_t WarpsPerGroup = (ReduceGroupWidth * ReduceGroupHeight) / tin::WarpSize;
    using Reducer = tin::Reducer<float, WarpsPerGroup>;

    __shared__ float smem[Reducer::sharedmem_size()];

    uint32_t const linearIndex = threadIdx.x + threadIdx.y * ReduceGroupWidth;

    for (int ch = 0; ch < image.channels; ++ch)
    {
        uint8_t const* const srcData = GetPitchLinearChannelGroupPtr(image, inputPixel.x, inputPixel.y, ch);
        uint32_t const srcChannelInGroup = GetPitchLinearChannelInGroup(image, ch);

        float const value = ReadImageChannel(srcData, srcChannelInGroup, ntc::ChannelFormat::FLOAT16, ColorSpace::Linear);

        float const min = Reducer::min(smem, value);
        float const max = Reducer::max(smem, value);

        if (linearIndex == 0)
        {
            atomicMin(mins + ch, HalfAsInt::Encode(min, false));
            atomicMax(maxs + ch, HalfAsInt::Encode(max, true));
        }
    }
}

cudaError_t ComputeMinMaxChannelValues(
    PitchLinearImageSlice image,
    int* scratchMemory,
    float outMinimums[NTC_MAX_CHANNELS],
    float outMaximums[NTC_MAX_CHANNELS])
{
    assert(scratchMemory);

    dim3 threadBlockSize(ReduceGroupWidth, ReduceGroupHeight, 1);
    dim3 gridSize = DivRoundUp(dim3(image.width, image.height, 1), threadBlockSize);
    
    InitMinMaxChannelValuesKernel <<< dim3(1), dim3(NTC_MAX_CHANNELS) >>> (scratchMemory, scratchMemory + NTC_MAX_CHANNELS);

    ComputeMinMaxChannelValuesKernel <<< gridSize, threadBlockSize >>> (image, scratchMemory, scratchMemory + NTC_MAX_CHANNELS);

    uint32_t tmp[NTC_MAX_CHANNELS * 2];
    cudaError_t err = cudaMemcpy(tmp, scratchMemory, sizeof(int) * NTC_MAX_CHANNELS * 2, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;

    for (int ch = 0; ch < image.channels; ++ch)
    {
        outMinimums[ch] = HalfAsInt::Decode(tmp[ch]);
        outMaximums[ch] = HalfAsInt::Decode(tmp[ch + NTC_MAX_CHANNELS]);
    }

    return cudaSuccess;
}

} // namespace ntc::cuda