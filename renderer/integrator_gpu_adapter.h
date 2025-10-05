// GPU Adapter for NanoVDB - wraps NanoVDB types to work with common integrator
#ifndef H_INTEGRATOR_GPU_ADAPTER
#define H_INTEGRATOR_GPU_ADAPTER

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/Ray.h>
#include "ray.cuh"

using GridT = nanovdb::FloatGrid;
using CoordT = nanovdb::Coord;
using RealT = float;
using Vec3T = nanovdb::Vec3<RealT>;
using Vec4T = nanovdb::Vec4<RealT>;
using RayT = nanovdb::Ray<RealT>;

// GPU World wrapper for NanoVDB
struct GPUWorld {
    Camera camera;
    GridT* grid;
    nanovdb::CoordBBox box;
    float maxSigma;

    inline __device__ Vec3T getRayDirection(int i) const {
        return camera.direction(i);
    }

    inline __device__ Vec3T worldToIndex(const Vec3T& worldPos) const {
        return grid->worldToIndex(worldPos);
    }

    inline __device__ RayT worldToIndexRay(const RayT& wRay) const {
        return wRay.worldToIndexF(*grid);
    }

    inline __device__ bool clipRay(RayT& ray) const {
        return ray.clip(box);
    }

    inline __device__ float getRayT0(const RayT& ray) const {
        return ray.t0();
    }

    inline __device__ float getRayT1(const RayT& ray) const {
        return ray.t1();
    }

    inline __device__ CoordT floorCoord(const Vec3T& pos) const {
        return CoordT::Floor(pos);
    }

    inline __device__ bool isActive(const CoordT& coord) const {
        return grid->tree().isActive(coord);
    }

    inline __device__ RayT createIndexRay(const Vec3T& newOrigin, const Vec3T& newDir, const Vec3T& wRayEye) const {
        // For GPU, we can directly create a ray in index space
        return RayT(newOrigin, newDir);
    }
};

#endif // H_INTEGRATOR_GPU_ADAPTER
