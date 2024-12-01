#ifndef H_RAY

#define H_RAY

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>

using RealT = float;
using Vec3T = nanovdb::Vec3<RealT>;

constexpr float FOV = 45.f;

struct Camera
{
    float mWBBoxDimZ;
    Vec3T mWBBoxCenter;
    int width;
    int height;
    float aspect;
    float tanFOV;

    inline Camera(float wBBoxDimZ, Vec3T wBBoxCenter, int width, int height)
        : mWBBoxDimZ(wBBoxDimZ), mWBBoxCenter(wBBoxCenter),
        width(width),
        height(height),
        aspect(width / float(height)),
        tanFOV(tanf(FOV / 2 * 3.14159265358979323846f / 180.f))
    {
    }

    inline __hostdev__ Vec3T direction(int i) const
    {
        // perspective camera along Z-axis...
        uint32_t x, y;
        x = i % width;
        y = i / width;

        const float u = (float(x) + 0.5f) / width;
        const float v = (float(y) + 0.5f) / height;
        const float Px = (2.f * u - 1.f) * tanFOV * aspect;
        const float Py = (2.f * v - 1.f) * tanFOV;
        Vec3T dir(Px, Py, -1.f);
        dir.normalize();

        return dir;
    }

    inline __hostdev__ Vec3T origin() const
    {
        return mWBBoxCenter + Vec3T(0, 0, mWBBoxDimZ);
    }
};

#endif