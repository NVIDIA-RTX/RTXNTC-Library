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

#include "AdaptiveCompressionSession.h"
#include "Errors.h"
#include "KnownLatentShapes.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <algorithm>

#define RUN_TEST 0

namespace ntc
{

Status AdaptiveCompressionSession::Reset(float targetPsnr, float maxBitsPerPixel)
{
    int const presetCount = GetKnownLatentShapeCount();

    m_targetPsnr = targetPsnr;
    m_currentRunIndex = 0;
    m_presetCount = presetCount;

    if (maxBitsPerPixel > 0.f)
    {
        // If a maximum bitrate is provided, convert that to a preset index.
        m_maxPreset = -1;
        for (int index = presetCount - 1; index >= 0; --index)
        {
            if (g_KnownLatentShapes[index].bitsPerPixel <= maxBitsPerPixel)
            {
                m_maxPreset = index;
                break;
            }
        }

        // Handle out-of-range bpp values.
        if (m_maxPreset < 0)
        {
            SetErrorMessage("Provided maxBitsPerPixel value (%.2f) is out of the supported range "
                "(%.1f-%.0f).",
                maxBitsPerPixel,
                g_KnownLatentShapes[0].bitsPerPixel,
                g_KnownLatentShapes[presetCount - 1].bitsPerPixel);
            return Status::InvalidArgument;
        }
    }
    else
    {
        // No maximum bitrate is provided.
        m_maxPreset = presetCount - 1;
    }

    // Start the full-range search at some midpoint.
    m_currentPreset = m_maxPreset / 2;
    assert(g_KnownLatentShapes[m_currentPreset].bitsPerPixel <= 3.5f);

    // Reset the boundaries.
    m_leftPreset = -1;
    m_rightPreset = -1;
    m_leftPsnr = -1;
    m_rightPsnr = -1;

    ClearErrorMessage();
    return Status::Ok;
}

bool AdaptiveCompressionSession::Finished()
{
    return m_leftPreset == m_rightPreset && m_leftPreset >= 0;
}

void AdaptiveCompressionSession::GetCurrentPreset(float *pOutBitsPerPixel, LatentShape *pOutLatentShape)
{
    KnownLatentShape const& preset = g_KnownLatentShapes[m_currentPreset];
    if (pOutBitsPerPixel)
        *pOutBitsPerPixel = preset.bitsPerPixel;
    if (pOutLatentShape)
        *pOutLatentShape = preset.shape;
}

/* This is the function used for interpolation search - a linear dependency of PSNR vs. BPP
   The forward version is not used here, just for reference.
static float Model(float x, float a, float b)
{
    return a * x + b;
}*/

// The inverse of Model(...)
static float InverseModel(float y, float a, float b)
{
    return (y - b) / a;
}

// Calculates the (a,b) parameters of the Model function using two points (x1,y1) and (x2,y2)
static void GetModelParams(float x1, float x2, float y1, float y2, float& a, float& b)
{
    a = (y2 - y1) / (x2 - x1);
    b = y1 - a * x1;
}

// Finds the index of the preset with BPP most closely matching the given one.
// Only searches within the (excludeLeft, excludeRight) index range, non-inclusively.
// When no matching preset found, returns -1.
static int FindClosestPreset(float targetBpp, int excludeLeft, int excludeRight, int presetCount)
{
    // If the requested BPP is clearly out of range, early out.
    if (excludeLeft >= 0 && targetBpp <= g_KnownLatentShapes[excludeLeft].bitsPerPixel ||
        excludeRight < presetCount && targetBpp >= g_KnownLatentShapes[excludeRight].bitsPerPixel)
        return -1;

    int bestIndex = -1;
    float bestBpp = -1;
    for (int index = excludeLeft + 1; index < excludeRight; ++index)
    {
        float currentBpp = g_KnownLatentShapes[index].bitsPerPixel;
        if ((bestIndex < 0) || (fabsf(currentBpp - targetBpp) < fabsf(bestBpp - targetBpp)))
        {
            bestIndex = index;
            bestBpp = currentBpp;
        }
    }
    return bestIndex;
}

void AdaptiveCompressionSession::Next(float currentPsnr)
{
    // Store the current preset in the history array.
    assert(m_currentRunIndex < MaxRuns);
    m_presetHistory[m_currentRunIndex] = m_currentPreset;
    ++m_currentRunIndex;

    // This is the first experiment - we got the midpoint result.
    // Choose if we test the lowest or highest preset next.
    if (m_leftPreset < 0 && m_rightPreset < 0)
    {
        if (currentPsnr >= m_targetPsnr)
        {
            m_rightPreset = m_currentPreset;
            m_rightPsnr = currentPsnr;
            m_leftPreset = 0;
            m_leftPsnr = -1;
            m_currentPreset = m_leftPreset;
        }
        else
        {
            m_leftPreset = m_currentPreset;
            m_leftPsnr = currentPsnr;
            m_rightPreset = m_maxPreset;
            m_rightPsnr = -1;
            m_currentPreset = m_rightPreset;
        }
        return;
    }

    // Maybe this is the second experiment - we got either the left or right result.
    if (m_currentPreset == m_leftPreset)
        m_leftPsnr = currentPsnr;
    else if (m_currentPreset == m_rightPreset)
        m_rightPsnr = currentPsnr;
    else
    {
        // No, it's some midpoint.
        // Update the boundaries according to the experiment result.
        if (currentPsnr >= m_targetPsnr)
        {
            m_rightPreset = m_currentPreset;
            m_rightPsnr = currentPsnr;
        }
        else
        {
            m_leftPreset = m_currentPreset;
            m_leftPsnr = currentPsnr;
        }
    }

    // Early out if the search range is empty after updating the boundaries.
    if (m_leftPreset == m_rightPreset)
        return;

    // Fit a model curve to the current boundaries
    float a, b;
    float const leftBpp = g_KnownLatentShapes[m_leftPreset].bitsPerPixel;
    float const rightBpp = g_KnownLatentShapes[m_rightPreset].bitsPerPixel;
    GetModelParams(leftBpp, rightBpp, m_leftPsnr, m_rightPsnr, a, b);

    // Predict the optimal BPP using the fitted model
    float const expectedBpp = InverseModel(m_targetPsnr, a, b);

    // Find a real BPP value most closely matching the predicted BPP, but excluding the left and right points.
    int const expectedPreset = FindClosestPreset(expectedBpp, m_leftPreset, m_rightPreset, m_presetCount);

    // If the prediction is not matching any real point between left and right, stop the search.
    if (expectedPreset < 0)
    {
        if (m_leftPsnr >= m_targetPsnr)
        {
            m_rightPreset = m_leftPreset;
            m_rightPsnr = m_leftPsnr;
        }
        else
        {
            m_leftPreset = m_rightPreset;
            m_leftPsnr = m_rightPsnr;
        }
        m_currentPreset = m_leftPreset;
    }
    else
    {
        m_currentPreset = expectedPreset;
    }
}

int AdaptiveCompressionSession::GetIndexOfFinalRun()
{
    if (!Finished())
        return -1;

    for (int index = 0; index < m_currentRunIndex; ++index)
    {
        if (m_presetHistory[index] == m_currentPreset)
            return index;
    }

    assert(false);
    return -1;
}

#if RUN_TEST

// Experimental PSNR vs. BPP curves from some materials.
constexpr int TestMaterialCount = 32;
static const float TestData[TestMaterialCount][KnownLatentShapeCount] = {
  { 29.81f, 31.13f, 32.88f, 34.73f, 36.29f, 37.57f, 41.34f, 45.31f, 47.11f },
  { 25.53f, 26.70f, 28.22f, 29.99f, 31.67f, 33.07f, 36.39f, 40.15f, 42.44f },
  { 25.50f, 26.77f, 28.79f, 30.91f, 32.88f, 33.58f, 38.15f, 41.92f, 44.10f },
  { 44.59f, 45.91f, 47.49f, 49.32f, 50.80f, 51.92f, 53.11f, 55.01f, 56.84f },
  { 45.15f, 46.34f, 47.79f, 49.40f, 50.64f, 52.04f, 53.96f, 56.01f, 56.48f },
  { 35.58f, 36.89f, 37.85f, 39.59f, 40.77f, 42.12f, 44.22f, 46.90f, 48.40f },
  { 36.18f, 37.32f, 38.39f, 39.93f, 40.99f, 42.13f, 44.14f, 46.64f, 48.07f },
  { 28.67f, 30.32f, 31.96f, 33.66f, 34.85f, 35.95f, 39.26f, 42.87f, 45.17f },
  { 41.15f, 42.01f, 43.00f, 44.29f, 45.22f, 46.71f, 49.07f, 51.73f, 53.24f },
  { 28.75f, 29.74f, 30.74f, 32.35f, 33.67f, 35.28f, 38.74f, 43.01f, 45.14f },
  { 33.42f, 35.03f, 37.02f, 39.18f, 40.84f, 42.73f, 45.65f, 48.39f, 50.07f },
  { 34.70f, 36.05f, 37.29f, 39.24f, 40.62f, 42.02f, 44.52f, 47.76f, 49.55f },
  { 38.94f, 40.34f, 41.65f, 43.58f, 44.99f, 46.25f, 48.49f, 50.71f, 52.26f },
  { 37.07f, 38.69f, 41.16f, 43.96f, 46.02f, 47.59f, 50.86f, 52.92f, 54.34f },
  { 41.39f, 42.51f, 43.64f, 45.17f, 46.33f, 47.38f, 49.10f, 51.75f, 52.79f },
  { 26.39f, 27.37f, 28.48f, 30.46f, 31.99f, 33.70f, 37.04f, 41.24f, 43.67f },
  { 50.29f, 51.96f, 53.69f, 56.22f, 56.69f, 59.05f, 61.13f, 61.04f, 63.11f },
  { 32.12f, 33.79f, 35.49f, 37.55f, 39.14f, 40.83f, 43.89f, 47.53f, 49.62f },
  { 27.10f, 29.05f, 31.56f, 34.40f, 36.12f, 36.85f, 41.58f, 45.47f, 47.84f },
  { 30.44f, 31.57f, 32.86f, 34.98f, 36.64f, 38.14f, 41.87f, 45.38f, 47.33f },
  { 27.01f, 27.92f, 28.99f, 30.39f, 31.85f, 34.05f, 38.94f, 44.69f, 47.08f },
  { 35.51f, 36.73f, 38.04f, 39.89f, 41.21f, 42.35f, 44.45f, 47.58f, 49.10f },
  { 35.59f, 36.65f, 37.79f, 39.45f, 40.59f, 41.74f, 43.94f, 47.13f, 49.09f },
  { 32.85f, 33.68f, 34.67f, 36.31f, 37.54f, 39.03f, 41.98f, 46.32f, 48.16f },
  { 31.64f, 32.59f, 33.79f, 35.47f, 36.87f, 38.50f, 41.72f, 45.33f, 47.15f },
  { 32.61f, 34.06f, 35.69f, 37.79f, 39.27f, 41.06f, 44.45f, 47.80f, 48.80f },
  { 33.03f, 34.21f, 35.40f, 37.23f, 38.58f, 40.12f, 42.90f, 46.46f, 48.40f },
  { 39.06f, 40.53f, 41.60f, 43.48f, 44.75f, 46.27f, 48.93f, 51.82f, 53.25f },
  { 23.06f, 23.86f, 25.06f, 26.96f, 28.52f, 29.99f, 33.00f, 37.28f, 40.08f },
  { 43.67f, 45.46f, 48.14f, 51.10f, 53.16f, 54.79f, 57.36f, 58.80f, 59.38f },
  { 27.34f, 28.82f, 30.45f, 32.57f, 34.26f, 35.64f, 39.28f, 42.43f, 44.25f },
  { 38.75f, 40.42f, 42.43f, 44.25f, 45.40f, 46.15f, 47.89f, 49.97f, 50.71f },
};

bool AdaptiveCompressionSession::Test()
{
    bool testPassed = true;
    float const maxBitsPerPixel = 20.f;
    int const presetCount = GetKnownLatentShapeCount();
    AdaptiveCompressionSession session;
    for (int materialIndex = 0; materialIndex < TestMaterialCount; ++materialIndex)
    {
        for (float targetPsnr = 30.f; targetPsnr <= 50.f; targetPsnr += 5.f)
        {
            float bppHistory[MaxRuns];

            // Find the optimal BPP using the adaptive compression session
            int experimentCount = 0;
            session.Reset(targetPsnr, maxBitsPerPixel);
            while (!session.Finished())
            {
                float bpp;
                session.GetCurrentPreset(&bpp, nullptr);
                bppHistory[experimentCount] = bpp;
                
                int preset = FindClosestPreset(bpp, -1, presetCount + 1, presetCount);
                assert(preset >= 0);

                float psnr = TestData[materialIndex][preset];
                ++experimentCount;
                session.Next(psnr);
            }

            float finalBpp;
            session.GetCurrentPreset(&finalBpp, nullptr);

            // Verify that GetIndexOfFinalRun() returns the correct index
            int finalIndex = session.GetIndexOfFinalRun();
            assert(bppHistory[finalIndex] == finalBpp);

            // Find the optimal BPP using linear search.
            // This loop will produce the lowest BPP that results in the target PSNR or more,
            // unless the target PSNR cannot be reached within 'maxBitsPerPixel', in which case the highest
            // supported BPP will be returned.
            float idealBpp = -1;
            for (int presetIndex = 0;
                presetIndex < presetCount &&
                    (maxBitsPerPixel <= 0.f || g_KnownLatentShapes[presetIndex].bitsPerPixel <= maxBitsPerPixel);
                ++presetIndex)
            {
                idealBpp = g_KnownLatentShapes[presetIndex].bitsPerPixel;

                if (TestData[materialIndex][presetIndex] >= targetPsnr)
                    break;
            }

            // Compare the results
            char const* testResult;
            if (idealBpp == finalBpp)
                testResult = "OK";
            else if (idealBpp < finalBpp)
                testResult = "SUBOPT";
            else
            {
                testResult = "FAIL";
                testPassed = false;
            }

            printf("Material %d: target = %.2f dB, found %5.2f bpp, ideal %5.2f bpp, %d experiments, final #%d - %s\n",
                materialIndex, targetPsnr, finalBpp, idealBpp, experimentCount, finalIndex, testResult);
        }
    }

    return testPassed;
}

// Poor man's test framework: call a function at module init.
static bool g_TestPassed = AdaptiveCompressionSession::Test();

#endif

}