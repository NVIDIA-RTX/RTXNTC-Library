/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "ThreadPool.h"
#include <cassert>

namespace ntc
{

ThreadPool::ThreadPool(IAllocator* allocator, uint32_t numThreads)
    : m_allocator(allocator)
    , m_threads(allocator)
    , m_tasks(allocator)
{
    assert(numThreads > 0);
    m_threads.resize(numThreads);
    for (uint32_t i = 0; i < numThreads; ++i)
    {
        m_threads[i] = std::thread(StaticThreadProc, this);
    }
    m_ownerThread = std::this_thread::get_id();
}

ThreadPool::~ThreadPool()
{
    WaitForTasks();

    m_terminate.store(true);

    m_forward.notify_all();

    for (std::thread& thread : m_threads)
        thread.join();
}

void ThreadPool::AddTask(std::shared_ptr<ThreadPoolTask> const& task)
{
    assert(std::this_thread::get_id() == m_ownerThread);
    
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_tasks.push(task);
        ++m_pendingTasks;
    }
    m_forward.notify_one();
}

bool ThreadPool::WaitForTasks()
{
    assert(std::this_thread::get_id() == m_ownerThread);
    
    while(m_pendingTasks.load() != 0)
        std::this_thread::yield();

    bool success = m_failedTasks.load() == 0;
    m_failedTasks.store(0);
    return success;
}

void ThreadPool::StaticThreadProc(ThreadPool* self)
{
    self->ThreadProc();
}

void ThreadPool::ThreadProc()
{
    while(!m_terminate.load())
    {
        std::shared_ptr<ThreadPoolTask> task;
        
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_forward.wait(lock, [this] { return !m_tasks.empty() || m_terminate.load(); });

            if (!m_tasks.empty())
            {
                task = std::move(m_tasks.front());
                m_tasks.pop();
            }
        }

        if (task)
        {
            try
            {
                if (!task->Run())
                    ++m_failedTasks;
            }
            catch(...)
            {
                ++m_failedTasks;
            }
            --m_pendingTasks;
        }
        else
            std::this_thread::yield();
    }
}

}