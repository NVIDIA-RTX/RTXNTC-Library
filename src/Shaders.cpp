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

#include "Shaders.h"
#include "MlpDesc.h"
#include <ShaderMake/ShaderBlob.h>

#if NTC_WITH_DX12
#include <DecompressINT8.dxil.h>
#include <DecompressCoopVecFP8.dxil.h>
#include <CompressBC1.dxil.h>
#include <CompressBC2.dxil.h>
#include <CompressBC3.dxil.h>
#include <CompressBC4.dxil.h>
#include <CompressBC5.dxil.h>
#include <CompressBC6.dxil.h>
#include <CompressBC7.dxil.h>
#include <ImageDifference.dxil.h>
#endif
#if NTC_WITH_VULKAN
#include <DecompressINT8.spirv.h>
#include <DecompressCoopVecFP8.spirv.h>
#include <CompressBC1.spirv.h>
#include <CompressBC2.spirv.h>
#include <CompressBC3.spirv.h>
#include <CompressBC4.spirv.h>
#include <CompressBC5.spirv.h>
#include <CompressBC6.spirv.h>
#include <CompressBC7.spirv.h>
#include <ImageDifference.spirv.h>
#endif

namespace ntc
{
void GetBC7ShaderBytecode(const uint8_t* blobData, size_t blobSize, bool useModeBuffer,
    const void** pOutData, size_t* pOutSize)
{
    ShaderMake::ShaderConstant constants[] = {
        { "USE_MODE_BUFFER", useModeBuffer ? "1" : "0" }
    };

    ShaderMake::FindPermutationInBlob(blobData, blobSize,
        constants, sizeof(constants) / sizeof(constants[0]), pOutData, pOutSize);
}

#define SET_SHADER_BYTECODE(symbol) \
    *pOutData = symbol; \
    *pOutSize = sizeof(symbol);

#if NTC_WITH_DX12
void GetDecompressDxilShaderBytecode(InferenceMath mathVersion, const void** pOutData, size_t* pOutSize)
{
    if (mathVersion == InferenceMath::CoopVecFP8)
    {
        SET_SHADER_BYTECODE(g_DecompressCoopVecFP8_dxil);
    }
    else
    {
        SET_SHADER_BYTECODE(g_DecompressINT8_dxil);
    }
}

void GetBlockCompressDxilShaderBytecode(BlockCompressedFormat format, bool useModeBuffer, const void** pOutData, size_t* pOutSize)
{
    switch(format)
    {
        case BlockCompressedFormat::BC1:
            SET_SHADER_BYTECODE(g_CompressBC1_dxil);
            break;
        case BlockCompressedFormat::BC2:
            SET_SHADER_BYTECODE(g_CompressBC2_dxil);
            break;
        case BlockCompressedFormat::BC3:
            SET_SHADER_BYTECODE(g_CompressBC3_dxil);
            break;
        case BlockCompressedFormat::BC4:
            SET_SHADER_BYTECODE(g_CompressBC4_dxil);
            break;
        case BlockCompressedFormat::BC5:
            SET_SHADER_BYTECODE(g_CompressBC5_dxil);
            break;
        case BlockCompressedFormat::BC6:
            SET_SHADER_BYTECODE(g_CompressBC6_dxil);
            break;
        case BlockCompressedFormat::BC7:
            GetBC7ShaderBytecode(g_CompressBC7_dxil, sizeof(g_CompressBC7_dxil), useModeBuffer, pOutData, pOutSize);
            break;
    }
}
void GetImageDifferenceDxilShaderBytecode(const void** pOutData, size_t* pOutSize)
{
    SET_SHADER_BYTECODE(g_ImageDifference_dxil);
}
#endif

#if NTC_WITH_VULKAN
void GetDecompressSpirvShaderBytecode(InferenceMath mathVersion, const void** pOutData, size_t* pOutSize)
{
    if (mathVersion == InferenceMath::CoopVecFP8)
    {
        SET_SHADER_BYTECODE(g_DecompressCoopVecFP8_spirv);
    }
    else
    {
        SET_SHADER_BYTECODE(g_DecompressINT8_spirv);
    }
}

void GetBlockCompressSpirvShaderBytecode(BlockCompressedFormat format, bool useModeBuffer, const void** pOutData, size_t* pOutSize)
{
    switch (format)
    {
        case BlockCompressedFormat::BC1:
            SET_SHADER_BYTECODE(g_CompressBC1_spirv);
            break;
        case BlockCompressedFormat::BC2:
            SET_SHADER_BYTECODE(g_CompressBC2_spirv);
            break;
        case BlockCompressedFormat::BC3:
            SET_SHADER_BYTECODE(g_CompressBC3_spirv);
            break;
        case BlockCompressedFormat::BC4:
            SET_SHADER_BYTECODE(g_CompressBC4_spirv);
            break;
        case BlockCompressedFormat::BC5:
            SET_SHADER_BYTECODE(g_CompressBC5_spirv);
            break;
        case BlockCompressedFormat::BC6:
            SET_SHADER_BYTECODE(g_CompressBC6_spirv);
            break;
        case BlockCompressedFormat::BC7:
            GetBC7ShaderBytecode(g_CompressBC7_spirv, sizeof(g_CompressBC7_spirv), useModeBuffer, pOutData, pOutSize);
            break;
    }
}
void GetImageDifferenceSpirvShaderBytecode(const void** pOutData, size_t* pOutSize)
{
    SET_SHADER_BYTECODE(g_ImageDifference_spirv);
}
#endif
}