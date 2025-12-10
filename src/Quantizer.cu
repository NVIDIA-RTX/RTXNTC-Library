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

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include "CudaUtils.h"
#include "FeatureGridMath.h"
#include "MlpDesc.h"
#include "WeightLayout.h"
#include "RegressionCommon.h"
#include "tin/tin_matrix_host.h"
#include "tin/tin_activation.h"
#include "tin/tin_mlp.h"
#include <libntc/ntc.h>
#include <cuda_fp8.h>


namespace ntc::cuda
{

namespace th = tin::host;

template<typename T>
__device__ T* GetPtrAtByteOffset(void* basePtr, size_t offset)
{
    return basePtr ? reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(basePtr) + offset) : nullptr;
}

template<typename T>
__device__ T const* GetPtrAtByteOffset(void const* basePtr, size_t offset)
{
    return basePtr ? reinterpret_cast<T const*>(reinterpret_cast<uint8_t const*>(basePtr) + offset) : nullptr;
}


struct AddressParams
{
    th::HMatrixB weightMatrix;
    int rows;
    int col = 0;
    int layerIndex = 0;
    bool inputLayer = false;
    bool outputLayer = false;

    __device__ AddressParams(int rows, int cols)
        : weightMatrix(rows, cols)
        , rows(rows)
    { }
};

static __device__ AddressParams GetColumnAddressParams(int threadIdx)
{
    // All quantizer kernels are used with a single thread block where each thread processes one column (output channel),
    // with the columns from all layers concatenated together.
    // Determine which layer and column in the layer this thread is responsible for.

    int columnOffset = 0;
    int layerIndex;
    int layerInputs;
    int layerOutputs;
    for (layerIndex = 0; layerIndex < NTC_MLP_LAYERS; ++layerIndex)
    {
        layerInputs = MlpDesc::GetLayerInputChannels(layerIndex);
        layerOutputs = MlpDesc::GetLayerOutputChannels(layerIndex);
        if (threadIdx < columnOffset + layerOutputs)
            break;  
        columnOffset += layerOutputs;
    }

    AddressParams params(layerInputs, layerOutputs);
    params.col = threadIdx - columnOffset;
    params.layerIndex = layerIndex;
    params.inputLayer = layerIndex == 0;
    params.outputLayer = layerIndex == (NTC_MLP_LAYERS - 1);
    return params;
}
extern __constant__ ChannelInfo g_ChannelInfo[NTC_MAX_CHANNELS];

__device__ void QuantizeColumnInt8(
    AddressParams params,
    half2* __restrict__ fp16WeightsForLayer,
    half* __restrict__ fp16BiasForLayer,
    int8_t* __restrict__ int8WeightsForLayer,
    float* __restrict__ scaleForLayer,
    int32_t* __restrict__ biasForLayer)
{
    float elemMin = std::numeric_limits<float>::max();
    float elemMax = std::numeric_limits<float>::min();

    for (int r = 0; r < params.rows; r += 2)
    {
        int elemOffset = params.weightMatrix.get_packed_offset(r, params.col);

        float2 elem = __half22float2(fp16WeightsForLayer[elemOffset]);

        elemMin = std::min(elemMin, elem.x);
        elemMax = std::max(elemMax, elem.x);
        elemMin = std::min(elemMin, elem.y);
        elemMax = std::max(elemMax, elem.y);
    }
    float limit = std::max(fabs(elemMax), fabs(elemMin));
    float ilimit = __frcp_rn(limit);

    // Quantize each column
    const float levels = 256;
    const float scale = (levels - 1) / 2;
    const float iscale = 1 / scale;
    const float qmin = -levels / 2 + 1;
    const float qmax =  levels / 2 - 1;

    int integerWeightSum = 0;

    for (int r = 0; r < params.rows; r += 2)
    {
        int elemOffset = params.weightMatrix.get_packed_offset(r, params.col);

        half2 helem = fp16WeightsForLayer[elemOffset];

        float2 elem = __half22float2(helem);
        elem.x = round(elem.x * scale * ilimit);
        elem.x = std::max(std::min(elem.x, qmax), qmin);
        int8_t qx = int8_t(elem.x);
        elem.x = elem.x * limit * iscale;

        elem.y = round(elem.y * (scale / limit));
        elem.y = std::max(std::min(elem.y, qmax), qmin);
        int8_t qy = int8_t(elem.y);
        elem.y = elem.y * limit * iscale;
        half2 res = __float22half2_rn(elem);

        fp16WeightsForLayer[elemOffset] = res;

        if (int8WeightsForLayer)
        {
            int addr = params.col * params.rows + r;
            int8WeightsForLayer[addr + 0] = qx;
            int8WeightsForLayer[addr + 1] = qy;
        }
        
        integerWeightSum += qx + qy;
    }

    if (scaleForLayer || biasForLayer)
    {
        float layerScale = limit * iscale;
        float layerBias = fp16BiasForLayer[params.col];

        const float activationScale = tin::ActHGELUClamp::step;
        const int activationBias = tin::ActHGELUClamp::bias;

        if (params.inputLayer)
        {
            layerScale /= tin::InputQuant::scale;
        }
        else
        {
            layerScale *= activationScale;
            layerBias  -= float(integerWeightSum * activationBias) * layerScale;

            if (params.outputLayer)
            {
                layerScale *= g_ChannelInfo[params.col].optimalToLinearScale;
                layerBias  = layerBias * g_ChannelInfo[params.col].optimalToLinearScale + g_ChannelInfo[params.col].optimalToLinearBias;
            }
        }

        // Convert the float scale and bias from the (float(output) * scale + bias) form to the
        // (float(output + int(bias/scale)) * scale) form.
        // Special case when scale is zero, which can happen on the output layer if an image channel is constant.
        // See also LoadWeightsFromStream(...) in TextureSetMetadata.cpp which implements the same logic.
        int integerLayerBias = 0;
        if (layerScale == 0.f)
        {
            // Zero scale: use a predefined constant scale and express the constant bias using this scale.
            float const constantScale = 65536.f;
            layerScale = 1.f / constantScale;
            integerLayerBias = int(roundf(layerBias * constantScale));

            // Zero out the weights to produce the correct result.
            if (int8WeightsForLayer)
            {
                for (int r = 0; r < params.rows; r += 2)
                {
                    int addr = params.col * params.rows + r;
                    int8WeightsForLayer[addr + 0] = 0;
                    int8WeightsForLayer[addr + 1] = 0;
                }
            }
        }
        else
        {
            // Nonzero scale: simple conversion.
            integerLayerBias = int(roundf(layerBias / layerScale));
        }

        if (scaleForLayer) scaleForLayer[params.col] = layerScale;
        if (biasForLayer) biasForLayer[params.col] = integerLayerBias;
    }
}

__device__ void QuantizeColumnFP8(
    AddressParams params,
    half2* __restrict__ fp16WeightsForLayer,
    half* __restrict__ fp16BiasForLayer,
    int8_t* __restrict__ fp8WeightsForLayer,
    half* __restrict__ biasForLayer)
{
    for (int r = 0; r < params.rows; r += 2)
    {
        int elemOffset = params.weightMatrix.get_packed_offset(r, params.col);

        half2 helem = fp16WeightsForLayer[elemOffset];
        half2 res;

        if (fp8WeightsForLayer)
        {
            // When we need to actually convert the weights, use CUDA FP8 math
            __nv_fp8x2_e4m3 qelem = __nv_fp8x2_e4m3(__half2(helem));
            int8_t qx = int8_t(qelem.__x & 0xff);
            int8_t qy = int8_t(qelem.__x >> 8);
            res = half2(qelem);

            int addr = params.col * params.rows + r;
            fp8WeightsForLayer[addr + 0] = qx;
            fp8WeightsForLayer[addr + 1] = qy;
        }
        else
        {
            // When we don't need the FP8 weights, use the round function because it's faster on pre-SM8.9 GPUs
            res.x = tin::RoundHalfToFloatE4M3(helem.x);
            res.y = tin::RoundHalfToFloatE4M3(helem.y);
        }
        
        fp16WeightsForLayer[elemOffset] = res;
    }

    // FP8 weights are not used for the last layer, and therefore don't have scales.
    // Bias is not quantized, just copied over.
    if (biasForLayer)
    {
        biasForLayer[params.col] = fp16BiasForLayer[params.col];
    }
}

__global__ void QuantizeNetworkInt8Kernel(
    WeightLayout const fp16WeightLayout,
    WeightLayout const quantizedWeightLayout,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ int8Data)
{
    using namespace cooperative_groups;
    auto block = cooperative_groups::this_thread_block();

    int i = block.thread_rank();
    AddressParams params = GetColumnAddressParams(i);

    QuantizeColumnInt8(params, 
        GetPtrAtByteOffset<half2>(halfWeights, fp16WeightLayout.weights[params.layerIndex].offset),
        GetPtrAtByteOffset<half>(halfWeights, fp16WeightLayout.biases[params.layerIndex].offset),
        GetPtrAtByteOffset<int8_t>(int8Data, quantizedWeightLayout.weights[params.layerIndex].offset),
        GetPtrAtByteOffset<float>(int8Data, quantizedWeightLayout.scales[params.layerIndex].offset),
        GetPtrAtByteOffset<int32_t>(int8Data, quantizedWeightLayout.biases[params.layerIndex].offset));
}

__global__ void QuantizeNetworkFP8Kernel(
    WeightLayout const fp16WeightLayout,
    WeightLayout const quantizedWeightLayout,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ fp8Data)
{
    using namespace cooperative_groups;
    auto block = cooperative_groups::this_thread_block();

    int i = block.thread_rank();
    AddressParams params = GetColumnAddressParams(i);

    if (params.outputLayer)
    {
        QuantizeColumnInt8(params,
            GetPtrAtByteOffset<half2>(halfWeights, fp16WeightLayout.weights[params.layerIndex].offset),
            GetPtrAtByteOffset<half>(halfWeights, fp16WeightLayout.biases[params.layerIndex].offset),
            GetPtrAtByteOffset<int8_t>(fp8Data, quantizedWeightLayout.weights[params.layerIndex].offset),
            GetPtrAtByteOffset<float>(fp8Data, quantizedWeightLayout.scales[params.layerIndex].offset),
            GetPtrAtByteOffset<int32_t>(fp8Data, quantizedWeightLayout.biases[params.layerIndex].offset));
    }
    else
    {
        QuantizeColumnFP8(params,
            GetPtrAtByteOffset<half2>(halfWeights, fp16WeightLayout.weights[params.layerIndex].offset),
            GetPtrAtByteOffset<half>(halfWeights, fp16WeightLayout.biases[params.layerIndex].offset),
            GetPtrAtByteOffset<int8_t>(fp8Data, quantizedWeightLayout.weights[params.layerIndex].offset),
            GetPtrAtByteOffset<half>(fp8Data, quantizedWeightLayout.biases[params.layerIndex].offset));
    }
}

void QuantizeNetwork(
    WeightLayout const& fp16WeightLayout,
    WeightLayout const& quantizedWeightLayout,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ outputData,
    bool useFP8)
{
    int threadBlockSize = MlpDesc::GetTotalOutputCount();
    int gridSize = 1;

    if (useFP8)
    {
        QuantizeNetworkFP8Kernel <<< gridSize, threadBlockSize >>> (fp16WeightLayout, quantizedWeightLayout, halfWeights, outputData);
    }
    else
    {
        QuantizeNetworkInt8Kernel <<< gridSize, threadBlockSize >>> (fp16WeightLayout, quantizedWeightLayout, halfWeights, outputData);
    }
}

__device__ void UnquantizeColumnInt8(
    AddressParams params,
    half2* __restrict__ fp16WeightsForLayer,
    half* __restrict__ fp16BiasForLayer,
    int8_t const* __restrict__ int8WeightsForLayer,
    float const* __restrict__ scaleForLayer,
    int const* __restrict__ biasForLayer)
{
    float layerScale = scaleForLayer[params.col];
    int const integerLayerBias = biasForLayer[params.col];
    float layerBias = float(integerLayerBias) * layerScale;

    // This function reverses the effect of QuantizeNetworkInt8Kernel, except the optimalToLinear scale and bias
    
    // Undo the layerScale multiplication
    if (params.inputLayer)
    {
        layerScale *= tin::InputQuant::scale;
    }
    else
    {
        layerScale *= tin::ActHGELUClamp::invStep;
    }

    // Go over all weights in the column and multiply them by scale.
    // Also accumulate the sum of integer weights to undo the bias change.
    int integerWeightSum = 0;
    for (int r = 0; r < params.rows; r += 2)
    {
        // Read two int8 weights in colum major layout
        int addr = params.col * params.rows + r;
        int8_t qx = int8WeightsForLayer[addr + 0];
        int8_t qy = int8WeightsForLayer[addr + 1];

        float2 elem;
        elem.x = float(qx) * layerScale;
        elem.y = float(qy) * layerScale;

        // Write two fp16 weights in MMA layout
        int elemOffset = params.weightMatrix.get_packed_offset(r, params.col);
        fp16WeightsForLayer[elemOffset] = __float22half2_rn(elem);
    
        integerWeightSum += qx + qy;
    }

    // Undo the bias change
    if (!params.inputLayer)
    {
        const float activationScale = tin::ActHGELUClamp::step;
        const int activationBias = tin::ActHGELUClamp::bias;

        // Note: multiplying by activationScale here because that term was removed from layerScale earlier
        layerBias += float(integerWeightSum * activationBias) * layerScale * activationScale;
    }

    // Write the fp16 bias
    fp16BiasForLayer[params.col] = layerBias;
}

__device__ void UnquantizeColumnFP8(
    AddressParams params,
    half2* __restrict__ fp16WeightsForLayer,
    half* __restrict__ fp16BiasForLayer,
    int8_t const* __restrict__ fp8WeightsForLayer,
    half const* __restrict__ biasForLayer)
{
    // This function reverses the effect of QuantizeNetworkFP8Kernel, except the optimalToLinear scale and bias
    
    for (int r = 0; r < params.rows; r += 2)
    {
        // Read two fp8 weights in colum major layout
        int addr = params.col * params.rows + r;
        
        __nv_fp8x2_e4m3 qelem;
        qelem.__x = *reinterpret_cast<uint16_t const*>(fp8WeightsForLayer + addr);

        // Write two fp16 weights in MMA layout
        int elemOffset = params.weightMatrix.get_packed_offset(r, params.col);
        fp16WeightsForLayer[elemOffset] = half2(qelem);
    }

    // Write the fp16 bias
    float layerBias = biasForLayer ? float(biasForLayer[params.col]) : 0.f;
    fp16BiasForLayer[params.col] = layerBias;
}

__global__ void ConvertNetworkFromInt8ToFP16Kernel(
    WeightLayout const fp16WeightLayout,
    WeightLayout const quantizedWeightLayout,
    half* __restrict__ halfWeights,
    int8_t const* __restrict__ int8Data)
{
    using namespace cooperative_groups;
    auto block = cooperative_groups::this_thread_block();
    
    int i = block.thread_rank();
    AddressParams params = GetColumnAddressParams(i);

    UnquantizeColumnInt8(params,
        GetPtrAtByteOffset<half2>(halfWeights, fp16WeightLayout.weights[params.layerIndex].offset),
        GetPtrAtByteOffset<half>(halfWeights, fp16WeightLayout.biases[params.layerIndex].offset),
        GetPtrAtByteOffset<int8_t>(int8Data, quantizedWeightLayout.weights[params.layerIndex].offset),
        GetPtrAtByteOffset<float>(int8Data, quantizedWeightLayout.scales[params.layerIndex].offset),
        GetPtrAtByteOffset<int32_t>(int8Data, quantizedWeightLayout.biases[params.layerIndex].offset));
}

__global__ void ConvertNetworkFromFP8ToFP16Kernel(
    WeightLayout const fp16WeightLayout,
    WeightLayout const quantizedWeightLayout,
    half* __restrict__ halfWeights,
    int8_t const* __restrict__ fp8Data)
{
    using namespace cooperative_groups;
    auto block = cooperative_groups::this_thread_block();
    
    int i = block.thread_rank();
    AddressParams params = GetColumnAddressParams(i);

    if (params.outputLayer)
    {
        UnquantizeColumnInt8(params,
            GetPtrAtByteOffset<half2>(halfWeights, fp16WeightLayout.weights[params.layerIndex].offset),
            GetPtrAtByteOffset<half>(halfWeights, fp16WeightLayout.biases[params.layerIndex].offset),
            GetPtrAtByteOffset<int8_t>(fp8Data, quantizedWeightLayout.weights[params.layerIndex].offset),
            GetPtrAtByteOffset<float>(fp8Data, quantizedWeightLayout.scales[params.layerIndex].offset),
            GetPtrAtByteOffset<int32_t>(fp8Data, quantizedWeightLayout.biases[params.layerIndex].offset));
    }
    else
    {
        UnquantizeColumnFP8(params,
            GetPtrAtByteOffset<half2>(halfWeights, fp16WeightLayout.weights[params.layerIndex].offset),
            GetPtrAtByteOffset<half>(halfWeights, fp16WeightLayout.biases[params.layerIndex].offset),
            GetPtrAtByteOffset<int8_t>(fp8Data, quantizedWeightLayout.weights[params.layerIndex].offset),
            GetPtrAtByteOffset<half>(fp8Data, quantizedWeightLayout.biases[params.layerIndex].offset));
    }
}

void ConvertNetworkFromQuantizedToFp16(
    WeightLayout const& fp16WeightLayout,
    WeightLayout const& quantizedWeightLayout,
    half* __restrict__ halfWeights,
    int8_t* __restrict__ inputData,
    bool useFP8)
{
    int threadBlockSize = MlpDesc::GetTotalOutputCount();
    int gridSize = 1;

    if (useFP8)
    {
        ConvertNetworkFromFP8ToFP16Kernel <<< gridSize, threadBlockSize >>> (fp16WeightLayout, quantizedWeightLayout, halfWeights, inputData);
    }
    else
    {
        ConvertNetworkFromInt8ToFP16Kernel <<< gridSize, threadBlockSize >>> (fp16WeightLayout, quantizedWeightLayout, halfWeights, inputData);
    }
}

__device__ uint32_t QuantizeValue(float value, int bits)
{
    float const quantizationStep = 1.f / float((1 << bits) - 1);
    value = std::max(0.f, std::min(1.f, value));
    float quantizedValue = roundf(value / quantizationStep);
    return uint32_t(quantizedValue);
}

__device__ float UnquantizeValue(uint32_t quantized, int bits)
{
    float const quantizationStep = 1.f / float((1 << bits) - 1);
    quantized &= (1 << bits) - 1;
    return float(quantized) * quantizationStep;
}

__global__ void PackLatentsKernel(
    int width,
    int height,
    int numLayers,
    size_t latentStride,
    const half* __restrict__ w_in,
    uint16_t* __restrict__ w_out)
{
    using namespace cooperative_groups;

    grid_group gg = this_grid();
    dim3 globalIdx = gg.thread_index();

    static_assert(FeatureGridMath::FeaturesPerGroup == 2, "Expecting 2 features per group");
    static_assert(FeatureGridMath::FeaturesPerLayer == 4, "Expecting 4 features per layer");

    if (globalIdx.x >= width || globalIdx.y >= height || globalIdx.z >= numLayers)
        return;
    
    size_t const srcOffset = globalIdx.z * latentStride * 2
                           + (globalIdx.y * width + globalIdx.x) * FeatureGridMath::FeaturesPerGroup;
    
    uint32_t packed;
    packed = QuantizeValue(float(w_in[srcOffset + 0]), 4);
    packed |= QuantizeValue(float(w_in[srcOffset + 1]), 4) << 4;
    packed |= QuantizeValue(float(w_in[srcOffset + latentStride + 0]), 4) << 8;
    packed |= QuantizeValue(float(w_in[srcOffset + latentStride + 1]), 4) << 12;
    
    size_t const dstOffset = (globalIdx.z * height + globalIdx.y) * width + globalIdx.x;
    w_out[dstOffset] = packed;
}

void PackLatents(
    int width,
    int height,
    int numLayers,
    size_t latentStride,
    const half* __restrict__ w_in,
    uint16_t* __restrict__ w_out)
{
    dim3 blockSize(8, 8, 1);
    dim3 gridSize = DivRoundUp(dim3(width, height, numLayers), blockSize);

    PackLatentsKernel <<< gridSize, blockSize >>> (width, height, numLayers, latentStride, w_in, w_out);
}

__global__ void UnpackLatentsKernel(
    int width,
    int height,
    int numLayers,
    size_t latentStride,
    const uint16_t* __restrict__ w_in,
    half* __restrict__ w_out)
{
    using namespace cooperative_groups;

    grid_group gg = this_grid();
    dim3 globalIdx = gg.thread_index();

    static_assert(FeatureGridMath::FeaturesPerGroup == 2, "Expecting 2 features per group");
    static_assert(FeatureGridMath::FeaturesPerLayer == 4, "Expecting 4 features per layer");

    if (globalIdx.x >= width || globalIdx.y >= height || globalIdx.z >= numLayers)
        return;
        
    size_t const srcOffset = (globalIdx.z * height + globalIdx.y) * width + globalIdx.x;
    uint32_t const packed = w_in[srcOffset];

    size_t const dstOffset = globalIdx.z * latentStride * 2
                           + (globalIdx.y * width + globalIdx.x) * FeatureGridMath::FeaturesPerGroup;

    w_out[dstOffset + 0] = half(UnquantizeValue(packed, 4));
    w_out[dstOffset + 1] = half(UnquantizeValue(packed >> 4, 4));
    w_out[dstOffset + latentStride + 0] = half(UnquantizeValue(packed >> 8, 4));
    w_out[dstOffset + latentStride + 1] = half(UnquantizeValue(packed >> 12, 4));
}

void UnpackLatents(
    int width,
    int height,
    int numLayers,
    size_t latentStride,
    const uint16_t* __restrict__ w_in,
    half* __restrict__ w_out)
{
    dim3 blockSize(8, 8, 1);
    dim3 gridSize = DivRoundUp(dim3(width, height, numLayers), blockSize);

    UnpackLatentsKernel <<< gridSize, blockSize >>> (width, height, numLayers, latentStride, w_in, w_out);
}

} // namespace ntc::cuda