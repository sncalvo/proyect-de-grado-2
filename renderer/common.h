// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef COMMON_NANOVDB_H

#define COMMON_NANOVDB_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <fstream>
#include <nanovdb/NanoVDB.h>

#include "ComputePrimitives.h"

inline __hostdev__ uint32_t CompactBy1(uint32_t x)
{
    x &= 0x55555555;
    x = (x ^ (x >> 1)) & 0x33333333;
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    x = (x ^ (x >> 8)) & 0x0000ffff;
    return x;
}

inline __hostdev__ uint32_t SeparateBy1(uint32_t x)
{
    x &= 0x0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f;
    x = (x ^ (x << 2)) & 0x33333333;
    x = (x ^ (x << 1)) & 0x55555555;
    return x;
}

inline __hostdev__ void mortonDecode(uint32_t code, uint32_t& x, uint32_t& y)
{
    x = CompactBy1(code);
    y = CompactBy1(code >> 1);
}

inline __hostdev__ void mortonEncode(uint32_t& code, uint32_t x, uint32_t y)
{
    code = SeparateBy1(x) | (SeparateBy1(y) << 1);
}

/*
template<typename RenderFn, typename GridT>
inline float renderImage(bool useCuda, const RenderFn renderOp, int width, int height, float* image, const GridT* grid)
{
    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    computeForEach(
        useCuda, width * height, 128, __FILE__, __LINE__, [renderOp, image, grid] __hostdev__(int start, int end) {
            renderOp(start, end, image, grid);
        });
    computeSync(useCuda, __FILE__, __LINE__);

    auto t1 = ClockT::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    return duration;
}*/

using Vec3T = nanovdb::Vec3<float>;

inline __hostdev__ void writeBuffer(float* outImage, int i, int w, int h, float value, float alpha, Vec3T& color)
{
    int offset;
    offset = i * 3;

    const float bgr = 0.f;
    const float bgg = 0.f;
    const float bgb = 0.f;
    outImage[offset] = color[0] * (1.f - alpha) + alpha * value + (1.0f - alpha) * bgr;
    outImage[offset + 1] = color[1] * (1.f - alpha) + alpha * value + (1.0f - alpha) * bgg;
    outImage[offset + 2] = color[2] * (1.f - alpha) + alpha * value + (1.0f - alpha) * bgb;
}

#endif // COMMON_NANOVDB_H