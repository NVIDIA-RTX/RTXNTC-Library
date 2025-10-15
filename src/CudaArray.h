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

#include <libntc/ntc.h>
#include <cuda_runtime_api.h>
#include <algorithm>

namespace ntc
{

template <typename T>
class HostArray {
public:
    HostArray(IAllocator* allocator)
        : m_allocator(allocator)
    { }

    HostArray(HostArray&& other) noexcept
    {
        *this = std::move(other);
    }
    
    ~HostArray()
    {
        Deallocate();
    }

    [[nodiscard]] bool Allocate(size_t length)
    {
        if (m_hostMemory)
            return false;

        m_length = length;
        m_hostMemory = (T*)m_allocator->Allocate(Size());
        return m_hostMemory != nullptr;
    }

    void Deallocate()
    {
        if (!m_hostMemory)
            return;

        m_allocator->Deallocate(m_hostMemory, Size());
        m_hostMemory = nullptr;
    }

    T* HostPtr()
    {
        return m_hostMemory;
    }

    const T* HostPtr() const
    {
        return m_hostMemory;
    }
    
    [[nodiscard]] T* HostPtrOffset(size_t offsetInElements)
    {
        return m_hostMemory ? m_hostMemory + offsetInElements : nullptr;
    }

    [[nodiscard]] const T* HostPtrOffset(size_t offsetInElements) const
    {
        return m_hostMemory ? m_hostMemory + offsetInElements : nullptr;
    }
    
    size_t Length() const
    {
        return m_length;
    }

    size_t Size() const
    {
        return m_length * sizeof(T);
    }

    HostArray<T>& operator=(HostArray<T>&& other) noexcept
    {
        m_length = other.m_length;
        m_hostMemory = other.m_hostMemory;
        other.m_length = 0;
        other.m_hostMemory = nullptr;
        return *this;
    }
    
protected:
    IAllocator* m_allocator;
    size_t m_length = 0;
    T* m_hostMemory = nullptr;
};

template <typename T>
class DeviceArray
{
public:
    DeviceArray() = default;

    DeviceArray(DeviceArray&& other) noexcept
    {
        *this = std::move(other);
    }
    
    ~DeviceArray()
    {
        Deallocate();
    }

    [[nodiscard]] bool Allocate(size_t length)
    {
        if (m_deviceMemory)
            return false;

        m_length = length;
        return cudaMalloc((void**)&m_deviceMemory, length * sizeof(T)) == cudaSuccess;
    }

    bool Deallocate()
    {
        if (!m_deviceMemory)
            return true;

        cudaError_t err = cudaFree(m_deviceMemory);
        m_deviceMemory = nullptr;
        return err == cudaSuccess;
    }

    [[nodiscard]] T* DevicePtr()
    {
        return m_deviceMemory;
    }

    [[nodiscard]] const T* DevicePtr() const
    {
        return m_deviceMemory;
    }

    [[nodiscard]] T* DevicePtrOffset(size_t offsetInElements)
    {
        return m_deviceMemory ? m_deviceMemory + offsetInElements : nullptr;
    }

    [[nodiscard]] const T* DevicePtrOffset(size_t offsetInElements) const
    {
        return m_deviceMemory ? m_deviceMemory + offsetInElements : nullptr;
    }

    size_t Length() const
    {
        return m_length;
    }

    size_t Size() const
    {
        return m_length * sizeof(T);
    }

    DeviceArray<T>& operator=(DeviceArray<T>&& other) noexcept
    {
        m_length = other.m_length;
        m_deviceMemory = other.m_deviceMemory;
        other.m_length = 0;
        other.m_deviceMemory = nullptr;
        return *this;
    }

protected:
    size_t m_length = 0;
    T* m_deviceMemory = nullptr;
};

template <typename T>
class DeviceAndHostArray : public HostArray<T>, public DeviceArray<T> {
public:

    DeviceAndHostArray(IAllocator* allocator)
        : HostArray<T>(allocator)
    { }

    DeviceAndHostArray(DeviceAndHostArray&& other) noexcept
    {
        *this = std::move(other);
    }
    
    ~DeviceAndHostArray()
    {
        Deallocate();
    }

    [[nodiscard]] bool Allocate(size_t length)
    {
        return HostArray<T>::Allocate(length) && DeviceArray<T>::Allocate(length);
    }

    bool Deallocate()
    {
        HostArray<T>::Deallocate();
        return DeviceArray<T>::Deallocate();
    }
    
    // Copies the entire array or a part of it from CPU memory to GPU memory.
    // If length == 0 (default), the entire array is copied.
    cudaError_t CopyToDevice(size_t length = 0)
    {
        if (!this->m_deviceMemory)
            return cudaErrorInvalidDevicePointer;

        if (!this->m_hostMemory)
            return cudaErrorInvalidHostPointer;

        if (length == 0)
            length = HostArray<T>::m_length;
        else if (length > HostArray<T>::m_length)
            return cudaErrorInvalidValue;

        return cudaMemcpy(this->m_deviceMemory, this->m_hostMemory, length * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Copies the entire array or a part of it from GPU memory to CPU memory.
    // If length == 0 (default), the entire array is copied.
    cudaError_t CopyToHost(size_t length = 0)
    {
        if (!this->m_deviceMemory)
            return cudaErrorInvalidDevicePointer;

        if (!this->m_hostMemory)
            return cudaErrorInvalidHostPointer;
            
        if (length == 0)
            length = HostArray<T>::m_length;
        else if (length > HostArray<T>::m_length)
            return cudaErrorInvalidValue;

        return cudaMemcpy(this->m_hostMemory, this->m_deviceMemory, length * sizeof(T), cudaMemcpyDeviceToHost);
    }

    size_t Length() const
    {
        return HostArray<T>::Length();
    }

    size_t Size() const
    {
        return HostArray<T>::Size();
    }

    DeviceAndHostArray<T>& operator=(DeviceAndHostArray<T>&& other) noexcept
    {
        DeviceArray<T>::operator=(other);
        HostArray<T>::operator=(other);
        return *this;
    }
};

}