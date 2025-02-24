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

namespace ntc
{

class Context;

class CudaDeviceGuard
{
public:
    CudaDeviceGuard(Context const* context);
    ~CudaDeviceGuard();
    bool Success() const { return m_success; }
private:
    bool m_success = false;
    int m_originalDevice = -1;
};

}