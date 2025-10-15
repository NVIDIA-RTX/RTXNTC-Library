# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

if (NTC_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(NTC_CUDA_ARCHITECTURES "75-virtual;89-virtual;86;89"
        CACHE STRING "List of CUDA device architectures to generate code for")
    set(NTC_CUDA_MIN_COMPUTE_MAJOR 7)
    set(NTC_CUDA_MIN_COMPUTE_MINOR 5)

    set(CMAKE_CUDA_STANDARD 17)
    set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

    if (MSVC)
    else()
        list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wall")
        list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
    endif()
    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
    list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
    list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

    option(NTC_CUDA_DEBUG "Compile CUDA kernels for GPU debugging" OFF)
    if (NTC_CUDA_DEBUG)
        list(APPEND CUDA_NVCC_FLAGS "--debug")
        list(APPEND CUDA_NVCC_FLAGS "--device-debug")
    endif()
    
    option(NTC_CUDA_LINEINFO "Embed source line information into CUDA kernels for profiling" OFF)
    if (NTC_CUDA_LINEINFO)
        list(APPEND CUDA_NVCC_FLAGS "--generate-line-info")
    endif()
endif()

if (NTC_BUILD_SHARED)
    add_library(libntc SHARED)

    target_compile_definitions(libntc PRIVATE NTC_BUILD_SHARED)
else()
    add_library(libntc STATIC)

    set_target_properties (libntc PROPERTIES 
        POSITION_INDEPENDENT_CODE ON)

    target_compile_definitions(libntc PUBLIC NTC_STATIC)
endif()

# Use configure_file and not macros to avoid full libntc rebuilds when git hash changes.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/VersionInfoTemplate.h
    ${CMAKE_CURRENT_BINARY_DIR}/VersionInfo.h    
)
target_include_directories(libntc PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

set(include_directory "${CMAKE_CURRENT_SOURCE_DIR}/../include")
set(bin_directory "${CMAKE_CURRENT_SOURCE_DIR}/../bin")

target_include_directories(libntc PUBLIC "${include_directory}")

if (LIBNTC_BIN_DIRECTORY)
    set_target_properties(libntc PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY                "${LIBNTC_BIN_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${LIBNTC_BIN_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${LIBNTC_BIN_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${LIBNTC_BIN_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${LIBNTC_BIN_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY                "${LIBNTC_BIN_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG          "${LIBNTC_BIN_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL     "${LIBNTC_BIN_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE        "${LIBNTC_BIN_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${LIBNTC_BIN_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY                "${LIBNTC_BIN_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG          "${LIBNTC_BIN_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     "${LIBNTC_BIN_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE        "${LIBNTC_BIN_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${LIBNTC_BIN_DIRECTORY}"
    )
endif()

if (NOT WIN32)
    set_target_properties(libntc PROPERTIES
        PREFIX "" # Avoid "liblibntc.so" on Linux
    )
endif()

if (MSVC)
    set_property(TARGET libntc PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_link_libraries(libntc PRIVATE RapidJSON)

if (MSVC)
    target_compile_definitions(libntc PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(libntc PRIVATE _USE_MATH_DEFINES)
    target_compile_definitions(libntc PRIVATE NOMINMAX)
endif()

target_compile_definitions(libntc PRIVATE
    NTC_WITH_DX12=$<BOOL:${NTC_WITH_DX12}>
    NTC_WITH_VULKAN=$<BOOL:${NTC_WITH_VULKAN}>
    NTC_WITH_CUDA=$<BOOL:${NTC_WITH_CUDA}>
    NTC_WITH_PREBUILT_SHADERS=$<BOOL:${NTC_WITH_PREBUILT_SHADERS}>)

if (WIN32)
    target_sources(libntc PRIVATE Resources.rc)
endif()

if (NTC_WITH_CUDA)
    target_link_libraries(libntc PRIVATE ${CUDA_LIBRARIES} CUDA::cudart_static)
    target_compile_options(libntc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_target_properties(libntc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${NTC_CUDA_ARCHITECTURES}")

    target_compile_definitions(libntc PRIVATE
        NTC_CUDA_MIN_COMPUTE_MAJOR=${NTC_CUDA_MIN_COMPUTE_MAJOR}
        NTC_CUDA_MIN_COMPUTE_MINOR=${NTC_CUDA_MIN_COMPUTE_MINOR})

    file(GLOB tin_sources "${CMAKE_CURRENT_SOURCE_DIR}/tin/*.h")

    target_sources(libntc PRIVATE 
        ${tin_sources}
        CudaArray.h
        CudaDeviceGuard.cpp
        CudaDeviceGuard.h
        CudaRandomGen.cu
        CudaRandomGen.h
        CudaUtils.h
        FeatureGridDevice.h
        FeatureGridHost.cpp
        FeatureGridHost.h
        ImageProcessing.cu
        ImageProcessing.h
        Inference.cu
        Optimizer.cu
        Optimizer.h
        Quantizer.cu
        Quantizer.h
        Regression.cu
        Regression.h
        RegressionFast.cu
        RegressionStable.cu
        RegressionKernels.h
        SharedTexture.h
        SharedTexture.cpp
        TextureSet.cpp
        TextureSet.h
    )

    source_group("TIN Library" FILES ${tin_sources})
    source_group("Generated CUDA Kernels" FILES ${kernels})
endif()

set(public_headers
    ${include_directory}/libntc/ntc.h
    ${include_directory}/libntc/wrappers.h
    ${include_directory}/libntc/shaders/BlockCompressConstants.h
    ${include_directory}/libntc/shaders/DecompressConstants.h
    ${include_directory}/libntc/shaders/ImageDifferenceConstants.h
    ${include_directory}/libntc/shaders/InferenceConstants.h
    ${include_directory}/libntc/shaders/Bindings.h
)

target_sources(libntc PRIVATE 
    ${public_headers}
    AdaptiveCompressionSession.h
    AdaptiveCompressionSession.cpp
    ChannelInfo.h
    Context.h
    Context.cpp
    CoopVecWeightConverter.h
    CoopVecWeightConverter.cpp
    Errors.cpp
    Errors.h
    FeatureGridMath.cpp
    FeatureGridMath.h
    GraphicsResources.h
    GraphicsResources.cpp
    JsonFileFormat.cpp
    JsonFileFormat.h
    JsonSerialization.h
    JsonSerialization.cpp
    KnownLatentShapes.h
    MathUtils.h
    MlpDesc.cpp
    MlpDesc.h
    PublicFunctions.cpp
    RegressionCommon.h
    Shaders.h
    StdTypes.h
    Stream.h
    Stream.cpp
    TextureMetadata.h
    TextureMetadata.cpp
    TextureSetMetadata.h
    TextureSetMetadata.cpp
)

source_group("Public Headers" FILES ${public_headers})

if (NTC_WITH_VULKAN)
    if (TARGET Vulkan-Headers)
        target_link_libraries(libntc PRIVATE Vulkan-Headers)
    elseif (TARGET Vulkan::Headers)
        target_link_libraries(libntc PRIVATE Vulkan::Headers)
    else()
        message(FATAL_ERROR "Neither Vulkan-Headers nor Vulkan::Headers targets are defined.")
    endif()
endif()

if (NTC_WITH_DX12)
    target_link_libraries(libntc PRIVATE DirectX-Headers dxgi)
endif()

if (NTC_WITH_PREBUILT_SHADERS AND (NTC_WITH_DX12 OR NTC_WITH_VULKAN))

    target_link_libraries(libntc PRIVATE ShaderMakeBlob RapidJSON)

    target_sources(libntc PRIVATE 
        Shaders.cpp
    )

    set(public_shader_headers
        ${include_directory}/libntc/shaders/BlockCompressConstants.h
        ${include_directory}/libntc/shaders/ColorSpaces.hlsli
        ${include_directory}/libntc/shaders/DecompressConstants.h
        ${include_directory}/libntc/shaders/ImageDifferenceConstants.h
        ${include_directory}/libntc/shaders/Inference.hlsli
        ${include_directory}/libntc/shaders/InferenceConstants.h
        ${include_directory}/libntc/shaders/InferenceCoopVec.hlsli
    )

    set(shader_sources
        shaders/BlockCompressCommon.hlsli
        shaders/CompressBC1.hlsl
        shaders/CompressBC2.hlsl
        shaders/CompressBC3.hlsl
        shaders/CompressBC4.hlsl
        shaders/CompressBC5.hlsl
        shaders/CompressBC6.hlsl
        shaders/CompressBC7.hlsl
        shaders/DecompressCommon.hlsli
        shaders/DecompressINT8.hlsl
        shaders/DecompressCoopVecFP8.slang
        shaders/HashBasedRNG.hlsli
        shaders/ImageDifference.hlsl
        shaders/BindingHelpers.hlsli
    )

    source_group("Public Headers" FILES ${public_shader_headers})
    source_group("Shaders" FILES ${shader_sources})

    set(shader_config_file "${CMAKE_CURRENT_SOURCE_DIR}/shaders/Shaders.cfg")
    set(shader_config_file_coopvec "${CMAKE_CURRENT_SOURCE_DIR}/shaders/CoopVecShaders.cfg")
    set(shader_output_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")

    add_custom_target(libntc-shaders
        DEPENDS ShaderMake
        SOURCES ${shader_sources} ${public_shader_headers})

    set_source_files_properties(${shader_sources} PROPERTIES VS_TOOL_OVERRIDE "None") 

    # Specify the shader build products so that Make/Ninja knows to build libntc when the generated .h files change.

    set(shader_byproducts_dxil
        ${shader_output_dir}/CompressBC1.dxil.h
        ${shader_output_dir}/CompressBC2.dxil.h
        ${shader_output_dir}/CompressBC3.dxil.h
        ${shader_output_dir}/CompressBC4.dxil.h
        ${shader_output_dir}/CompressBC5.dxil.h
        ${shader_output_dir}/CompressBC6.dxil.h
        ${shader_output_dir}/CompressBC7.dxil.h
        ${shader_output_dir}/DecompressINT8.dxil.h
        ${shader_output_dir}/ImageDifference.dxil.h
    )

    set(shader_byproducts_spirv
        ${shader_output_dir}/CompressBC1.spirv.h
        ${shader_output_dir}/CompressBC2.spirv.h
        ${shader_output_dir}/CompressBC3.spirv.h
        ${shader_output_dir}/CompressBC4.spirv.h
        ${shader_output_dir}/CompressBC5.spirv.h
        ${shader_output_dir}/CompressBC6.spirv.h
        ${shader_output_dir}/CompressBC7.spirv.h
        ${shader_output_dir}/DecompressINT8.spirv.h
        ${shader_output_dir}/ImageDifference.spirv.h
    )

    set(shader_byproducts_coopvec_spirv
        ${shader_output_dir}/DecompressCoopVecFP8.spirv.h
    )

    set(shader_byproducts_coopvec_dxil
        ${shader_output_dir}/DecompressCoopVecFP8.dxil.h
    )

    if (NTC_WITH_DX12)
        add_custom_command(TARGET libntc-shaders PRE_BUILD BYPRODUCTS ${shader_byproducts_dxil} COMMAND ShaderMake
            --config ${shader_config_file}
            --out ${shader_output_dir}
            --platform DXIL
            --project LibNTC
            --headerBlob
            --compiler ${SHADERMAKE_DXC_PATH}
            --shaderModel 6_5
            --hlsl2021
            -I ${include_directory})

        add_custom_command(TARGET libntc-shaders PRE_BUILD BYPRODUCTS ${shader_byproducts_coopvec_dxil} COMMAND ShaderMake
            --config ${shader_config_file_coopvec}
            --out ${shader_output_dir}
            --platform DXIL
            --project LibNTC
            --headerBlob
            --compiler ${SHADERMAKE_DXC_PATH}
            --shaderModel 6_9
            -I ${include_directory})
    endif()

    if (NTC_WITH_VULKAN)
        add_custom_command(TARGET libntc-shaders PRE_BUILD BYPRODUCTS ${shader_byproducts_spirv} COMMAND ShaderMake
            --config ${shader_config_file}
            --out ${shader_output_dir}
            --project LibNTC
            --platform SPIRV
            --headerBlob
            --compiler ${SHADERMAKE_DXC_PATH}
            --noRegShifts
            --vulkanVersion 1.2
            --hlsl2021
            -I ${include_directory}
            -D SPIRV)

        add_custom_command(TARGET libntc-shaders PRE_BUILD BYPRODUCTS ${shader_byproducts_coopvec_spirv} COMMAND ShaderMake
            --config ${shader_config_file_coopvec}
            --out ${shader_output_dir}
            --project LibNTC
            --platform SPIRV
            --headerBlob
            --compiler ${SHADERMAKE_SLANG_PATH}
            --slang
            --noRegShifts
            -X -Wno-41017
            -X "-capability spvCooperativeVectorNV"
            -I ${include_directory}
            -D SPIRV)
    endif()

    target_include_directories(libntc PRIVATE ${shader_output_dir})
    add_dependencies(libntc libntc-shaders)

endif()
