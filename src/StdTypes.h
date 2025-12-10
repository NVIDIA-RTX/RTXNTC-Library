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
#include <string>
#include <vector>
#include <queue>
#include <memory>

namespace ntc
{

template<typename T>
class Allocator
{
public:
    using value_type = T;

    Allocator(IAllocator* allocator) noexcept
        : m_allocator(allocator)
    { }

    template <class U> Allocator(const Allocator<U>& other) noexcept
        : m_allocator(other.m_allocator)
    { }

    T* allocate(std::size_t n)
    {
        return (T*)m_allocator->Allocate(sizeof(T) * n);
    }

    void deallocate(T* p, std::size_t n)
    {
        m_allocator->Deallocate(p, sizeof(T) * n);
    }

    bool operator==(const Allocator<T>& other) const noexcept
    {
        return &m_allocator == &other.m_allocator;
    }

    bool operator!=(const Allocator<T>& other) const noexcept
    {
        return (&m_allocator != &other.m_allocator);
    }
    
private:
    IAllocator* m_allocator;

    // Declare other template specializations as friends to access their m_allocator
    template <class U> friend class Allocator;
};

class String : public std::basic_string<char, std::char_traits<char>, Allocator<char>>
{
    using Base = std::basic_string<char, std::char_traits<char>, Allocator<char>>;
public:
    String(IAllocator* allocator)
        : Base(Allocator<char>(allocator))
    { }

    String(const char* s, IAllocator* allocator)
        : Base(s, Allocator<char>(allocator))
    { }
};

template<typename T>
class Vector : public std::vector<T, Allocator<T>>
{
    using Base = std::vector<T, Allocator<T>>;
public:
    Vector(IAllocator* allocator)
        : Base(Allocator<T>(allocator))
    { }

    Vector(size_t size, IAllocator* allocator)
        : Base(size, Allocator<T>(allocator))
    { }
};

template<typename T>
class Queue : public std::queue<T, std::deque<T, Allocator<T>>>
{
    using Base = std::queue<T, std::deque<T, Allocator<T>>>;
public:
    Queue(IAllocator* allocator)
        : Base(Allocator<T>(allocator))
    { }
};

template<typename T>
class Deleter
{
public:
    Deleter(IAllocator* allocator) : m_allocator(allocator) { }
    void operator()(T* ptr)
    { 
        ptr->~T();
        m_allocator->Deallocate(ptr, sizeof(T));
    }

private:
    IAllocator* m_allocator;
};

template<typename T>
class UniquePtr : public std::unique_ptr<T, Deleter<T>>
{
public:
    UniquePtr(T* val, IAllocator* allocator)
        : std::unique_ptr<T, Deleter<T>>(val, Deleter<T>(allocator))
    { }
};

// Constructs a new object of type T using the allocator.
// Returns a UniquePtr that will delete the object using the same allocator.
template<typename T, class... TArgs>
UniquePtr<T> MakeUnique(IAllocator* allocator, TArgs&&... args)
{
    T* object = new(allocator->Allocate(sizeof(T))) T(args...);
    return UniquePtr<T>(object, allocator);
}

// Constructs a new object of type T using the allocator, passes the allocator as the first argument to the object ctor.
// Returns a UniquePtr that will delete the object using the same allocator.
template<typename T, class... TArgs>
UniquePtr<T> MakeUniqueWithAllocator(IAllocator* allocator, TArgs&&... args)
{
    T* object = new(allocator->Allocate(sizeof(T))) T(allocator, args...);
    return UniquePtr<T>(object, allocator);
}

template<typename T, class... TArgs>
std::shared_ptr<T> MakeShared(IAllocator* allocator, TArgs&&... args)
{
    T* object = new(allocator->Allocate(sizeof(T))) T(args...);
    return std::shared_ptr<T>(object, Deleter<T>(allocator));
}

}