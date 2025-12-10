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

#include "StdTypes.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace std::chrono;

namespace ntc
{

class ThreadPoolTask
{
public:
    // Execute the task.
    // Returns true if successful, false on error.
    virtual bool Run() = 0;
};

class ThreadPool
{
public:
    ThreadPool(IAllocator* allocator, uint32_t numThreads);
    ~ThreadPool();

    // Enqueues a task for execution in the thread pool.
    // If any thread is available, the task immediately starts executing.
    // Note: AddTask must be called from the same thread that created the ThreadPool.
    void AddTask(std::shared_ptr<ThreadPoolTask> const& task);

    // Waits for all previously added tasks to complete or fail.
    // If any task has returned 'false' or threw an exception, WaitForTasks returns false.
    // Note: WaitForTasks must be called from the same thread that created the ThreadPool.
    bool WaitForTasks();

private:
    static void StaticThreadProc(ThreadPool* self);
    void ThreadProc();

    IAllocator* m_allocator;
    Vector<std::thread> m_threads;
    Queue<std::shared_ptr<ThreadPoolTask>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_forward;
    std::atomic<bool> m_terminate = false;
    std::atomic<int> m_pendingTasks = 0;
    std::atomic<int> m_failedTasks = 0;
    std::thread::id m_ownerThread;
};

}