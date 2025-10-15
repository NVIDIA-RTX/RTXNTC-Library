# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

function(LibNTC_ConfigureShaderCompilerVersions)
    option(SHADERMAKE_FIND_COMPILERS "" ON)
    option(SHADERMAKE_FIND_SLANG "" ON)
    option(SHADERMAKE_FIND_DXC "" ON)
    option(SHADERMAKE_FIND_DXC_VK "" OFF)
    set(SHADERMAKE_DXC_VERSION "v1.8.2505.1" CACHE STRING "")
    set(SHADERMAKE_DXC_DATE "2025_07_14" CACHE STRING "")
    set(SHADERMAKE_SLANG_VERSION "2025.18.2" CACHE STRING "")
endfunction()
