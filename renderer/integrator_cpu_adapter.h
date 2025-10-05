// CPU Adapter for OpenVDB - wraps OpenVDB types to work with common integrator
#ifndef H_INTEGRATOR_CPU_ADAPTER
#define H_INTEGRATOR_CPU_ADAPTER

#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>

using RealT = float;
using Vec3T = openvdb::math::Vec3<RealT>;
using Vec4T = openvdb::math::Vec4<RealT>;
using RayT = openvdb::math::Ray<RealT>;
using CoordT = openvdb::Coord;
using GridT = openvdb::FloatGrid;

constexpr float FOV = 45.f;

// Camera structure for CPU
struct CPUCamera
{
    float mWBBoxDimZ;
    Vec3T mWBBoxCenter;
    int width;
    int height;
    float aspect;
    float tanFOV;

    CPUCamera(float wBBoxDimZ, Vec3T wBBoxCenter, int width, int height)
        : mWBBoxDimZ(wBBoxDimZ), mWBBoxCenter(wBBoxCenter),
        width(width),
        height(height),
        aspect(width / float(height)),
        tanFOV(tanf(FOV / 2 * 3.14159265358979323846f / 180.f))
    {
    }

    Vec3T direction(int i) const
    {
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

    Vec3T origin() const
    {
        return mWBBoxCenter + Vec3T(0, 0, mWBBoxDimZ);
    }
};

// CPU World wrapper for OpenVDB
struct CPUWorld {
    CPUCamera camera;
    GridT::Ptr grid;
    openvdb::CoordBBox box;
    float maxSigma;
    openvdb::math::Transform::Ptr transform;
    openvdb::math::BBox<Vec3T> bboxFloat;

    CPUWorld(const CPUCamera& cam, GridT::Ptr g, const openvdb::CoordBBox& b, float sigma, openvdb::math::Transform::Ptr t)
        : camera(cam), grid(g), box(b), maxSigma(sigma), transform(t),
          bboxFloat(Vec3T(box.min().asVec3d()), Vec3T(box.max().asVec3d()))
    {}

    inline Vec3T getRayDirection(int i) const {
        return camera.direction(i);
    }

    inline Vec3T worldToIndex(const Vec3T& worldPos) const {
        return Vec3T(transform->worldToIndex(openvdb::math::Vec3d(worldPos)));
    }

    inline RayT worldToIndexRay(const RayT& wRay) const {
        Vec3T iRayOrigin = Vec3T(transform->worldToIndex(openvdb::math::Vec3d(wRay.eye())));
        Vec3T iRayDir = Vec3T(transform->worldToIndex(openvdb::math::Vec3d(wRay.eye() + wRay.dir()))) - iRayOrigin;
        iRayDir.normalize();
        return RayT(iRayOrigin, iRayDir);
    }

    inline bool clipRay(RayT& ray) const {
        return ray.clip(bboxFloat);
    }

    inline float getRayT0(const RayT& ray) const {
        return std::max(ray.t0(), 0.0f);
    }

    inline float getRayT1(const RayT& ray) const {
        return ray.t1();
    }

    inline CoordT floorCoord(const Vec3T& pos) const {
        return CoordT(int(std::floor(pos.x())), 
                     int(std::floor(pos.y())), 
                     int(std::floor(pos.z())));
    }

    inline bool isActive(const CoordT& coord) const {
        return grid->tree().isValueOn(coord);
    }

    inline RayT createIndexRay(const Vec3T& newOrigin, const Vec3T& newDir, const Vec3T& wRayEye) const {
        // Convert the new direction from world to index space
        Vec3T newIRayDir = Vec3T(transform->worldToIndex(openvdb::math::Vec3d(wRayEye + newDir))) 
                         - Vec3T(transform->worldToIndex(openvdb::math::Vec3d(wRayEye)));
        newIRayDir.normalize();
        return RayT(newOrigin, newIRayDir);
    }
};

#endif // H_INTEGRATOR_CPU_ADAPTER
