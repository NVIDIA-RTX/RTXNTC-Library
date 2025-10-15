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

// Custom header from a pre-rel Agility SDK goes first to avoid it being disabled by '#ifndef __d3d12_h__' in the beginning
#if NTC_WITH_DX12
#include <directx/d3d12.h>
#endif

#include "CoopVecWeightConverter.h"
#include "Context.h"
#include "GraphicsResources.h"
#include "StdTypes.h"
#include "MlpDesc.h"
#include <cassert>

#if NTC_WITH_DX12
#include <dxgi1_4.h>
#include <wrl.h>
#endif

namespace ntc
{

static const uint32_t g_NvidiaVendorID = 0x10DE;

static DataType GetDataTypeForWeights(InferenceWeightType weightType)
{
    switch (weightType)
    {
    case InferenceWeightType::GenericInt8:
        return DataType::Int8;
    case InferenceWeightType::GenericFP8:
    case InferenceWeightType::CoopVecFP8:
        return DataType::FP8;
    default:
        return DataType::None;
    }
}

static size_t GetDataTypeSize(DataType type)
{
    switch (type)
    {
    case DataType::None:
        return 0;
    case DataType::Int8:
        return sizeof(uint8_t);
    case DataType::Int32:
        return sizeof(int32_t);
    case DataType::FP8:
        return sizeof(uint8_t);
    case DataType::FP16:
        return sizeof(uint16_t);
    case DataType::FP32:
        return sizeof(float);
    default:
        assert(!"Unsupported data type");
        return 0;
    }
}

#if NTC_WITH_VULKAN
static VkConvertCooperativeVectorMatrixInfoNV GetVkConvertLayerDesc(DataType type, int inputChannels, int outputChannels,
    size_t* pDstSize, uint64_t srcData = 0, uint64_t dstData = 0)
{
    size_t const componentSize = GetDataTypeSize(type);

    VkConvertCooperativeVectorMatrixInfoNV layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;
    layoutInfo.srcSize = inputChannels * outputChannels * componentSize;
    layoutInfo.srcData.deviceAddress = srcData;
    layoutInfo.pDstSize = pDstSize;
    layoutInfo.dstData.deviceAddress = dstData;
    switch (type)
    {
    case DataType::Int8:
        layoutInfo.srcComponentType = VK_COMPONENT_TYPE_SINT8_KHR;
        break;
    case DataType::FP8:
        layoutInfo.srcComponentType = VK_COMPONENT_TYPE_FLOAT_E4M3_NV;
        break;
    case DataType::FP16:
        layoutInfo.srcComponentType = VK_COMPONENT_TYPE_FLOAT16_KHR;
        layoutInfo.srcSize *= sizeof(uint16_t);
        break;
    default:
        assert(!"Unsupported weight type for cooperative vector conversion");
    }
    layoutInfo.dstComponentType = layoutInfo.srcComponentType;
    layoutInfo.numRows = outputChannels;
    layoutInfo.numColumns = inputChannels;
    layoutInfo.srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
    layoutInfo.srcStride = inputChannels * componentSize;
    layoutInfo.dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV;
    layoutInfo.dstStride = 0;
    return layoutInfo;
}
#endif

#if NTC_WITH_DX12
static D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO GetDX12ConvertLayerDestInfo(DataType type, int inputChannels, int outputChannels)
{
    D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO info{};
    info.DestLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_MUL_OPTIMAL;
    info.NumRows = outputChannels;
    info.NumColumns = inputChannels;
    info.DestStride = inputChannels;
    switch (type)
    {
    case DataType::Int8:
        info.DestDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8;
        break;
    case DataType::FP8:
        info.DestDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT_E4M3;
        break;
    case DataType::FP16:
        info.DestDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        break;
    default:
        assert(!"Unsupported weight type for cooperative vector conversion");
    }
    return info;
}
static D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO GetDX12ConvertLayerDesc(DataType type, int inputChannels, int outputChannels,
    size_t dstSize, uint64_t srcData, uint64_t dstData)
{
    uint32_t const componentSize = uint32_t(GetDataTypeSize(type));

    D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO info{};
    info.DestInfo = GetDX12ConvertLayerDestInfo(type, inputChannels, outputChannels);
    info.DestInfo.DestSize = UINT(dstSize);
    info.SrcInfo.SrcSize = inputChannels * outputChannels * componentSize;
    info.SrcInfo.SrcDataType = info.DestInfo.DestDataType;
    info.SrcInfo.SrcLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
    info.SrcInfo.SrcStride = inputChannels * componentSize;
    info.DataDesc.SrcVA = srcData;
    info.DataDesc.DestVA = dstData;
    return info;
}
static D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO GetDX12CopyScaleBiasDesc(size_t size, uint64_t srcData, uint64_t dstData)
{
    assert(size % sizeof(float) == 0);

    D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO info{};
    info.DestInfo.DestSize = UINT(size);
    info.DestInfo.DestLayout = D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
    info.DestInfo.DestStride = UINT(size);
    info.DestInfo.NumRows = 1;
    info.DestInfo.NumColumns = UINT(size) / sizeof(float);
    // Use FP32 to define the scale and bias copy for comatibility with Intel GPUs
    info.DestInfo.DestDataType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32;
    info.SrcInfo.SrcSize = info.DestInfo.DestSize;
    info.SrcInfo.SrcDataType = info.DestInfo.DestDataType;
    info.SrcInfo.SrcLayout = info.DestInfo.DestLayout;
    info.SrcInfo.SrcStride = info.DestInfo.DestStride;
    info.DataDesc.SrcVA = srcData;
    info.DataDesc.DestVA = dstData;
    return info;
}
#endif

bool CoopVecWeightConverter::IsConversionSupported(GraphicsResources const* resources, InferenceWeightType weightType)
{
    if (!resources)
        return false;

    switch (weightType)
    {
    case InferenceWeightType::CoopVecFP8: return resources->IsCoopVecSupported();
    default: return false;
    }
}

bool CoopVecWeightConverter::GetWeightLayout(GraphicsResources const* resources, MlpDesc const& mlpDesc,
    InferenceWeightType weightType, WeightLayout& outLayout)
{
    outLayout.weights[0].type = outLayout.weights[1].type = outLayout.weights[2].type = GetDataTypeForWeights(weightType);
    outLayout.weights[3].type = (outLayout.weights[0].type == DataType::FP8) ? DataType::Int8 : outLayout.weights[0].type;

    if (!IsCoopVecWeightType(weightType))
    {
        for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
        {
            outLayout.weights[layer].size =
                mlpDesc.GetLayerInputChannels(layer) *
                mlpDesc.GetLayerOutputChannels(layer) *
                GetDataTypeSize(outLayout.weights[layer].type);
        }
    }
    else
    {
        // Handle all coopvec layouts below

        if (!IsConversionSupported(resources, weightType))
            return false;

        if (resources->GetGraphicsApi() == GraphicsAPI::Vulkan)
        {
#if NTC_WITH_VULKAN
            VkDevice const vkDevice = resources->GetVulkanDevice();

            assert(vkDevice);
            assert(resources->pfn_vkConvertCooperativeVectorMatrixNV);

            // Compute converted weight sizes for all layers
            for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
            {
                VkConvertCooperativeVectorMatrixInfoNV convertInfo = GetVkConvertLayerDesc(
                    outLayout.weights[layer].type, mlpDesc.GetLayerInputChannels(layer),
                    mlpDesc.GetLayerOutputChannels(layer), &outLayout.weights[layer].size);

                VkResult res = resources->pfn_vkConvertCooperativeVectorMatrixNV(vkDevice, &convertInfo);

                if (res != VK_SUCCESS)
                    return false;
            }

#else // !NTC_WITH_VULKAN
            return false;
#endif
        }
        if (resources->GetGraphicsApi() == GraphicsAPI::D3D12)
        {
#if NTC_WITH_DX12
            ID3D12Device* d3dDevice = resources->GetD3D12Device();

            Microsoft::WRL::ComPtr<ID3D12DevicePreview> devicePreview;
            if (d3dDevice->QueryInterface(IID_PPV_ARGS(&devicePreview)) != S_OK)
                return false;

            // Compute converted weight sizes for all layers
            for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
            {
                // Compute converted size for input layer
                D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO convertInfo = GetDX12ConvertLayerDestInfo(
                    outLayout.weights[layer].type, mlpDesc.GetLayerInputChannels(layer),
                    mlpDesc.GetLayerOutputChannels(layer));

                devicePreview->GetLinearAlgebraMatrixConversionDestinationInfo(&convertInfo);

                // The GetLinearAlgebraMatrixConversionDestinationInfo function doesn't have a way to signal failure,
                // so check the returned size for sanity instead.
                // Note: if the function fails, it might mean that the D3D12CooperativeVectorExperiment experimental
                // feature is not enabled on the device.
                if (convertInfo.DestSize == 0)
                    return false;

                outLayout.weights[layer].size = convertInfo.DestSize;
            }

#else // !NTC_WITH_DX12
            return false;
#endif
        }
    }

    // Calculate the offsets for all layers' weights and total weight size
    for (int layer = 1; layer < NTC_MLP_LAYERS; ++layer)
    {
        outLayout.weights[layer].offset = outLayout.weights[layer - 1].offset + outLayout.weights[layer - 1].size;
    }
    outLayout.combinedWeights.offset = 0;
    outLayout.combinedWeights.size = outLayout.weights[NTC_MLP_LAYERS - 1].offset + outLayout.weights[NTC_MLP_LAYERS - 1].size;

    // Calculate the sizes for the scale and bias vectors
    size_t totalScaleSize = 0;
    size_t totalBiasSize = 0;
    for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
    {
        DataType scaleType = DataType::None;
        DataType biasType = DataType::None;

        switch (weightType)
        {
        case InferenceWeightType::GenericInt8:
            scaleType = DataType::FP32;
            biasType = DataType::Int32;
            break;

        case InferenceWeightType::GenericFP8:
        case InferenceWeightType::CoopVecFP8:
            if (layer == NTC_MLP_LAYERS - 1)
            {
                scaleType = DataType::FP32;
                biasType = DataType::Int32;
            }
            else
            {
                // FP8 mode uses FP16 bias vectors, no scale
                biasType = DataType::FP16;
            }
            break;
        }

        outLayout.scales[layer].type = scaleType;
        outLayout.biases[layer].type = biasType;
        outLayout.scales[layer].size = GetDataTypeSize(scaleType) * mlpDesc.GetLayerOutputChannels(layer);
        outLayout.biases[layer].size = GetDataTypeSize(biasType) * mlpDesc.GetLayerOutputChannels(layer);
        totalScaleSize += outLayout.scales[layer].size;
        totalBiasSize += outLayout.biases[layer].size;
    }

    // Allocate the scale and bias vector block
    outLayout.combinedScaleBias.offset = outLayout.combinedWeights.offset + outLayout.combinedWeights.size;
    outLayout.combinedScaleBias.size = totalScaleSize + totalBiasSize;

    // Calculate the offsets for the scale and bias vectors following the rules for each format.
    // See the comment in the beginning of TextureSet.cpp
    switch (weightType)
    {
    case InferenceWeightType::GenericInt8:
        // All scale vectors one after another, then all bias vectors
        outLayout.scales[0].offset = outLayout.combinedScaleBias.offset;
        outLayout.biases[0].offset = outLayout.combinedScaleBias.offset + totalScaleSize;
        for (int layer = 1; layer < NTC_MLP_LAYERS; ++layer)
        {
            outLayout.scales[layer].offset = outLayout.scales[layer - 1].offset + outLayout.scales[layer - 1].size;
            outLayout.biases[layer].offset = outLayout.biases[layer - 1].offset + outLayout.biases[layer - 1].size;
        }
        break;

    case InferenceWeightType::GenericFP8:
    case InferenceWeightType::CoopVecFP8:
        // Bias vectors for layers 0-2, then scale for layer 3, bias for layer 3
        outLayout.biases[0].offset = outLayout.combinedScaleBias.offset;
        outLayout.biases[1].offset = outLayout.biases[0].offset + outLayout.biases[0].size;
        outLayout.biases[2].offset = outLayout.biases[1].offset + outLayout.biases[1].size;
        outLayout.scales[3].offset = outLayout.biases[2].offset + outLayout.biases[2].size;
        outLayout.biases[3].offset = outLayout.scales[3].offset + outLayout.scales[3].size;
        break;
    }

    outLayout.bufferSize = outLayout.combinedWeights.size + outLayout.combinedScaleBias.size;

    return true;
}

void CoopVecWeightConverter::ConvertWeights(GraphicsResources const* resources, MlpDesc const& mlpDesc,
    WeightLayout const& srcLayout, void* srcBuffer, uint64_t srcOffset,
    WeightLayout const& dstLayout, void* dstBuffer, uint64_t dstOffset,
    void* commandListOrBuffer)
{
    assert(srcBuffer);
    assert(dstBuffer);
    assert(commandListOrBuffer);
    
#if NTC_WITH_VULKAN
    if (resources->GetGraphicsApi() == GraphicsAPI::Vulkan)
    {
        assert(resources->pfn_vkCmdConvertCooperativeVectorMatrixNV);
        assert(resources->pfn_vkCmdCopyBuffer);
        assert(resources->pfn_vkGetBufferDeviceAddress);

        VkDevice vkDevice = resources->GetVulkanDevice();
        VkCommandBuffer vkCmdBuf = static_cast<VkCommandBuffer>(commandListOrBuffer);
        VkBuffer vkSrcBuffer = static_cast<VkBuffer>(srcBuffer);
        VkBuffer vkDstBuffer = static_cast<VkBuffer>(dstBuffer);

        // Obtain the device addresses of the buffers for the conversion functions
        VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
        bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bufferDeviceAddressInfo.buffer = vkSrcBuffer;
        VkDeviceAddress const srcBufferVA = resources->pfn_vkGetBufferDeviceAddress(vkDevice, &bufferDeviceAddressInfo);
        bufferDeviceAddressInfo.buffer = vkDstBuffer;
        VkDeviceAddress const dstBufferVA = resources->pfn_vkGetBufferDeviceAddress(vkDevice, &bufferDeviceAddressInfo);

        // Fill the array with conversion parameters for all layers
        VkConvertCooperativeVectorMatrixInfoNV convertInfos[NTC_MLP_LAYERS];

        for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
        {
            int const inputChannels = mlpDesc.GetLayerInputChannels(layer);
            int const outputChannels = mlpDesc.GetLayerOutputChannels(layer);

            size_t dstLayerSize = dstLayout.weights[layer].size;
            convertInfos[layer] = GetVkConvertLayerDesc(dstLayout.weights[layer].type,
                inputChannels, outputChannels, &dstLayerSize,
                srcBufferVA + srcOffset + srcLayout.weights[layer].offset,
                dstBufferVA + dstOffset + dstLayout.weights[layer].offset);
        }
    
        // Perform all conversions in one call
        resources->pfn_vkCmdConvertCooperativeVectorMatrixNV(vkCmdBuf, NTC_MLP_LAYERS, convertInfos);

        // Copy the scale and bias data
        VkBufferCopy region{};
        region.srcOffset = srcOffset + srcLayout.combinedScaleBias.offset;
        region.dstOffset = dstOffset + dstLayout.combinedScaleBias.offset;
        region.size = dstLayout.combinedScaleBias.size;

        resources->pfn_vkCmdCopyBuffer(vkCmdBuf, vkSrcBuffer, vkDstBuffer, 1, &region);
    }
#endif
#if NTC_WITH_DX12
    if (resources->GetGraphicsApi() == GraphicsAPI::D3D12)
    {
        ID3D12Device* d3dDevice = resources->GetD3D12Device();
        ID3D12GraphicsCommandList* d3dCmdList = static_cast<ID3D12GraphicsCommandList*>(commandListOrBuffer);
        ID3D12Resource* d3dSrcBuffer = static_cast<ID3D12Resource*>(srcBuffer);
        ID3D12Resource* d3dDstBuffer = static_cast<ID3D12Resource*>(dstBuffer);

        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandListPreview> commandListPreview;
        if (d3dCmdList->QueryInterface(IID_PPV_ARGS(&commandListPreview)) != S_OK)
            return;

        D3D12_GPU_VIRTUAL_ADDRESS const srcBufferVA = d3dSrcBuffer->GetGPUVirtualAddress();
        D3D12_GPU_VIRTUAL_ADDRESS const dstBufferVA = d3dDstBuffer->GetGPUVirtualAddress();

        D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO convertInfos[NTC_MLP_LAYERS + 1];

        for (int layer = 0; layer < NTC_MLP_LAYERS; ++layer)
        {
            int const inputChannels = mlpDesc.GetLayerInputChannels(layer);
            int const outputChannels = mlpDesc.GetLayerOutputChannels(layer);

            convertInfos[layer] = GetDX12ConvertLayerDesc(
                dstLayout.weights[layer].type, inputChannels, outputChannels,
                dstLayout.weights[layer].size,
                srcBufferVA + srcOffset + srcLayout.weights[layer].offset,
                dstBufferVA + dstOffset + dstLayout.weights[layer].offset);
        }
        
        // Copy the scale and bias data.
        // D3D's CopyBufferRegion requires resource states incompatible with the conversion ops.
        // Use a degenerate form of a matrix conversion to copy the extra data and avoid placing a barrier.
        convertInfos[4] = GetDX12CopyScaleBiasDesc(dstLayout.combinedScaleBias.size,
            srcBufferVA + srcOffset + srcLayout.combinedScaleBias.offset,
            dstBufferVA + dstOffset + dstLayout.combinedScaleBias.offset);
        
        commandListPreview->ConvertLinearAlgebraMatrix(convertInfos, NTC_MLP_LAYERS + 1);
    }
#endif
}

bool CoopVecWeightConverter::IsCoopVecWeightType(InferenceWeightType weightType)
{
    return weightType == InferenceWeightType::CoopVecFP8;
}

InferenceWeightType CoopVecWeightConverter::GetGenericWeightType(InferenceWeightType weightType)
{
    switch (weightType)
    {
    case InferenceWeightType::CoopVecFP8:
        return InferenceWeightType::GenericFP8;
    default:
        // For all other types, return the same type
        return weightType;
    }
}

#if NTC_WITH_DX12
void CoopVecWeightConverter::IsDX12CoopVecSupported(GraphicsResources const* resources, bool& outSupported)
{
    outSupported = false;

    // Get the general cooperative vector support tier
    D3D12_FEATURE_DATA_D3D12_OPTIONS_EXPERIMENTAL experimentalOptions{};
    if (resources->GetD3D12Device()->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL,
        &experimentalOptions, UINT(sizeof experimentalOptions)) != S_OK)
        return;

    if (experimentalOptions.CooperativeVectorTier < D3D12_COOPERATIVE_VECTOR_TIER_1_0)
        return;

    // Get the format count
    D3D12_FEATURE_DATA_COOPERATIVE_VECTOR coopVecData{};
    if (resources->GetD3D12Device()->CheckFeatureSupport(D3D12_FEATURE_COOPERATIVE_VECTOR,
        &coopVecData, UINT(sizeof coopVecData)) != S_OK)
        return;
    
    // Get the supported format list
    Vector<D3D12_COOPERATIVE_VECTOR_PROPERTIES_MUL> properties(coopVecData.MatrixVectorMulAddPropCount,
        resources->GetAllocator());
    coopVecData.pMatrixVectorMulAddProperties = properties.data();
    coopVecData.OuterProductAccumulatePropCount = 0;
    coopVecData.VectorAccumulatePropCount = 0;
    if (resources->GetD3D12Device()->CheckFeatureSupport(D3D12_FEATURE_COOPERATIVE_VECTOR,
        &coopVecData, UINT(sizeof coopVecData)) != S_OK)
        return;

    // Go over the properties and see if there are formats that we need in the list
    bool int8LayerSupported = false;
    bool fp8LayersSupported = false;
    for (const auto& prop : properties)
    {
        if (prop.InputType == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32 &&
            prop.InputInterpretation == D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8 &&
            prop.MatrixInterpretation == D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8 &&
            prop.OutputType == D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32)
            int8LayerSupported = true;

        if (prop.InputType == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16 &&
            prop.InputInterpretation == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT_E4M3 &&
            prop.MatrixInterpretation == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT_E4M3 &&
            prop.OutputType == D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16)
            fp8LayersSupported = true;
    }

    outSupported = int8LayerSupported && fp8LayersSupported;
}
#endif

#if NTC_WITH_VULKAN
void CoopVecWeightConverter::IsVkCoopVecSupported(GraphicsResources const* resources,
    bool& outSupported)
{
    outSupported = false;

    VkPhysicalDevice vkPhysicalDevice = resources->GetVulkanPhysicalDevice();

    if (!resources->pfn_vkGetPhysicalDeviceCooperativeVectorPropertiesNV ||
        !resources->pfn_vkConvertCooperativeVectorMatrixNV || 
        !resources->pfn_vkCmdConvertCooperativeVectorMatrixNV ||
        !vkPhysicalDevice)
        return;

    // Verify that the physical device we're on was made by NVIDIA.
    // When running on the integrated GPU in an Optimus system, the vkGetPhysicalDeviceCooperativeVectorPropertiesNV
    // function pointer is nonzero, but calling the function results in a crash.
    if (resources->GetVulkanPhysicalDeviceProperties().vendorID != g_NvidiaVendorID)
        return;
    
    // Get the property count
    uint32_t propertyCount = 0;
    if (resources->pfn_vkGetPhysicalDeviceCooperativeVectorPropertiesNV(vkPhysicalDevice,
        &propertyCount, nullptr) != VK_SUCCESS)
        return;
        
    Vector<VkCooperativeVectorPropertiesNV> properties(propertyCount, resources->GetAllocator());
    // Init the sType fields
    for (auto& prop : properties)
        prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV;

    // Get the actual properties
    if (resources->pfn_vkGetPhysicalDeviceCooperativeVectorPropertiesNV(vkPhysicalDevice,
        &propertyCount, properties.data()) != VK_SUCCESS)
        return;

    // Go over the properties and see if there are formats that we need in the list
    bool int8LayerSupported = false;
    bool fp8LayersSupported = false;
    for (const auto& prop : properties)
    {
        if (prop.sType == VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV &&
            prop.inputType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            prop.inputInterpretation == VK_COMPONENT_TYPE_SINT8_KHR &&
            prop.matrixInterpretation == VK_COMPONENT_TYPE_SINT8_KHR &&
            prop.resultType == VK_COMPONENT_TYPE_SINT32_KHR)
            int8LayerSupported = true;

        if (prop.sType == VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV &&
            prop.inputType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            prop.inputInterpretation == VK_COMPONENT_TYPE_FLOAT_E4M3_NV &&
            prop.matrixInterpretation == VK_COMPONENT_TYPE_FLOAT_E4M3_NV &&
            prop.resultType == VK_COMPONENT_TYPE_FLOAT16_KHR)
            fp8LayersSupported = true;
    }

    outSupported = int8LayerSupported && fp8LayersSupported;
}
#endif

}