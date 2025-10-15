/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <libntc/ntc.h>

namespace ntc
{

class AdaptiveCompressionSession : public IAdaptiveCompressionSession
{
public:
    AdaptiveCompressionSession() = default;

    Status Reset(float targetPsnr, float maxBitsPerPixel = 0.f) override;
    bool Finished() override;
    void GetCurrentPreset(float*  pOutBitsPerPixel, LatentShape* pOutLatentShape) override;
    void Next(float currentPsnr) override;
    int GetIndexOfFinalRun() override;

    static bool Test();

private:
    float m_targetPsnr = 0;
    int m_currentPreset = -1;
    int m_leftPreset = -1;
    int m_rightPreset = -1;
    float m_leftPsnr = 0.f;
    float m_rightPsnr = 0.f;
    int m_currentRunIndex = 0;
    int m_maxPreset = 0;
    int m_presetCount = 0;
    
    static constexpr int MaxRuns = 16; // Should be never more than 5 but let's be extra sure
    int m_presetHistory[MaxRuns];
};

}