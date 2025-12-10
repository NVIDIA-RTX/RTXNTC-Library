/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "GraphicsResources.h"
#include "CoopVecWeightConverter.h"
#include <libntc/ntc.h>

namespace ntc
{

GraphicsResources::GraphicsResources(ContextParameters const& params)
    : m_allocator(params.pAllocator)
    , m_graphicsApi(params.graphicsApi)
{
    if (params.graphicsApi == GraphicsAPI::D3D12)
    {
#if NTC_WITH_DX12
        m_d3d12Device = static_cast<ID3D12Device*>(params.d3d12Device);

        if (params.enableCooperativeVector)
        {
            CoopVecWeightConverter::IsDX12CoopVecSupported(this, m_coopVecSupported);
        }
#endif
    }
    else if (params.graphicsApi == GraphicsAPI::Vulkan)
    {
#if NTC_WITH_VULKAN
        m_vulkanInstance = static_cast<VkInstance>(params.vkInstance);
        m_vulkanPhysicalDevice = static_cast<VkPhysicalDevice>(params.vkPhysicalDevice);
        m_vulkanDevice = static_cast<VkDevice>(params.vkDevice);
        m_vulkanLoader = new (m_allocator->Allocate(sizeof(VulkanDynamicLoader))) VulkanDynamicLoader();
        
        PFN_vkGetInstanceProcAddr const vkGetInstanceProcAddr =
            m_vulkanLoader->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        if (vkGetInstanceProcAddr)
        {
            #define LOAD_INSTANCE_FN(name) pfn_##name = (PFN_##name)vkGetInstanceProcAddr(m_vulkanInstance, #name)

            LOAD_INSTANCE_FN(vkGetPhysicalDeviceCooperativeVectorPropertiesNV);
            
            #undef LOAD_INSTANCE_FN
        }

        PFN_vkGetDeviceProcAddr const vkGetDeviceProcAddr =
            m_vulkanLoader->getProcAddress<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
        if (vkGetDeviceProcAddr)
        {
            #define LOAD_DEVICE_FN(name) pfn_##name = (PFN_##name)vkGetDeviceProcAddr(m_vulkanDevice, #name)

            LOAD_DEVICE_FN(vkCmdConvertCooperativeVectorMatrixNV);
            LOAD_DEVICE_FN(vkCmdCopyBuffer);
            LOAD_DEVICE_FN(vkCmdDecompressMemoryNV);
            LOAD_DEVICE_FN(vkConvertCooperativeVectorMatrixNV);
            LOAD_DEVICE_FN(vkGetBufferDeviceAddress);

            #undef LOAD_DEVICE_FN
        }

        PFN_vkGetPhysicalDeviceProperties const vkGetPhysicalDeviceProperties =
            m_vulkanLoader->getProcAddress<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
        if (m_vulkanPhysicalDevice && vkGetPhysicalDeviceProperties)
        {
            vkGetPhysicalDeviceProperties(m_vulkanPhysicalDevice, &m_vulkanPhysicalDeviceProperties);
        }

        if (params.enableCooperativeVector)
        {
            CoopVecWeightConverter::IsVkCoopVecSupported(this, m_coopVecSupported);
        }
#endif
    }

    if (!params.enableCooperativeVector)
        m_coopVecSupported = false;
}

GraphicsResources::~GraphicsResources()
{
#if NTC_WITH_VULKAN
    if (m_vulkanLoader)
    {
        m_vulkanLoader->~DynamicLoader();
        m_allocator->Deallocate(m_vulkanLoader, sizeof(VulkanDynamicLoader));
        m_vulkanLoader = nullptr;
    }
#endif
}
  
}