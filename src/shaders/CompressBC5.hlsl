#define OUTPUT_FORMAT uint4
#include "BlockCompressCommon.hlsli"

[numthreads(BLOCK_COMPRESS_CS_GROUP_WIDTH, BLOCK_COMPRESS_CS_GROUP_HEIGHT, 1)]
void main(uint2 globalIdx : SV_DispatchThreadID)
{
    if (globalIdx.x >= g_Const.widthInBlocks || globalIdx.y >= g_Const.heightInBlocks)
        return;

    // Load the input pixels.
    
    float reds[PIXELS_PER_BLOCK];
    float greens[PIXELS_PER_BLOCK];
    FOREACH_PIXEL(idx)
    {
        const float4 color = t_Input[BCn_GetPixelPos(globalIdx, idx) + uint2(g_Const.srcLeft, g_Const.srcTop)];
        reds[idx] = color.r;
        greens[idx] = color.g;
    }

    // Compress red and green components as BC4.

    uint2 compressedRed = BC4_CompressFast(reds);
    uint2 compressedGreen = BC4_CompressFast(greens);
    
    // Write the output.

    u_Output[globalIdx + uint2(g_Const.dstOffsetX, g_Const.dstOffsetY)] = uint4(compressedRed, compressedGreen);
}