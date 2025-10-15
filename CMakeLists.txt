# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.18)
include(FetchContent)
include(tools/ConfigureCompilers.cmake)

set(LANGUAGES CXX)

option(NTC_BUILD_SHARED "Build LibNTC as a shared library (DLL or .so)" ON)
option(NTC_WITH_CUDA "Compile NTC with CUDA support" ON)
option(NTC_WITH_DX12 "Compile NTC with DX12 support" "${WIN32}")
option(NTC_WITH_VULKAN "Compile NTC with Vulkan support" ON)
option(NTC_WITH_PREBUILT_SHADERS "Compile NTC with pre-built shaders for decompression etc." ON)

if (NTC_WITH_CUDA)
    list(APPEND LANGUAGES CUDA)
endif()

project(
    libntc
    DESCRIPTION "Neural Texture Compression Library"
    LANGUAGES ${LANGUAGES}
)

# Version info

set(NTC_VERSION_MAJOR 0)
set(NTC_VERSION_MINOR 8)
set(NTC_VERSION_POINT 0)

# Read Git branch name into NTC_VERSION_BRANCH
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NTC_VERSION_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Read Git short hash into NTC_VERSION_HASH
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NTC_VERSION_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "LibNTC version: ${NTC_VERSION_MAJOR}.${NTC_VERSION_MINOR}.${NTC_VERSION_POINT} "
                "${NTC_VERSION_BRANCH}-${NTC_VERSION_HASH}")


if ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    set (LIBNTC_IS_SUBMODULE OFF)
    set (LIBNTC_BIN_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
    if (MSVC)
        set (CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif ()
else()
    set (LIBNTC_IS_SUBMODULE ON)
endif()

# Include Vulkan-Headers if it hasn't been included before
if (NOT TARGET Vulkan-Headers AND NOT TARGET Vulkan::Headers AND NTC_WITH_VULKAN)
    FetchContent_Declare(
        Vulkan-Headers
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG v1.4.307
    )
    FetchContent_MakeAvailable(Vulkan-Headers)
endif()

# Include DirectX-Headers if it hasn't been included before and if it's needed
if (NOT TARGET DirectX-Headers AND NTC_WITH_DX12)
    FetchContent_Declare(
        DirectX-Headers
        GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers.git
        GIT_TAG v1.717.0-preview
    )
    FetchContent_MakeAvailable(DirectX-Headers)
endif()

# Include ShaderMake if it hasn't been included before and if it's needed
if (NOT TARGET ShaderMake AND NTC_WITH_PREBUILT_SHADERS AND (NTC_WITH_DX12 OR NTC_WITH_VULKAN))
    LibNTC_ConfigureShaderCompilerVersions()

    set(SHADERMAKE_BIN_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/ShaderMake" CACHE STRING "")
    FetchContent_Declare(
        ShaderMake
        GIT_REPOSITORY https://github.com/NVIDIAGameWorks/ShaderMake.git
        GIT_TAG de47d8677843ee2fd1a231b94970e71818252a62
    )
    FetchContent_MakeAvailable(ShaderMake)
endif()

if (MSVC AND NOT LIBNTC_IS_SUBMODULE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()

add_subdirectory(external/rapidjson)
add_subdirectory(src)
