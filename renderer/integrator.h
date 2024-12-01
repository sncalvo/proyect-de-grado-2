#ifndef H_INTEGRATOR
#define H_INTEGRATOR

#include <chrono>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>

#include "ray.h"
#include "common.h"

#include <curand.h>
#include <curand_kernel.h>

using BufferT = nanovdb::CudaDeviceBuffer;
using GridT = nanovdb::FloatGrid;
using CoordT = nanovdb::Coord;
using RealT = float;
using Vec3T = nanovdb::Vec3<RealT>;
using RayT = nanovdb::Ray<RealT>;
using ClockT = std::chrono::high_resolution_clock;

constexpr unsigned int DEPTH_LIMIT = 10;
constexpr float SIGMA_A = 0.02f;
constexpr float SIGMA_S = 0.2f;
constexpr float MIN_STEP = 0.01f;
constexpr float MAX_STEP = 0.1f;
constexpr float PI = 3.14159265358979323846f;

Vec3T UP = Vec3T {0.0f, 1.0f, 0.0f};

Vec3T __hostdev__ cross(const Vec3T& a, const Vec3T& b) {
    return Vec3T(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

std::pair<Vec3T, Vec3T> __hostdev__ coordinateSystem(const Vec3T& a) {
    Vec3T b;

    if (fabs(a[0]) > fabs(a[1])) {
        b = Vec3T(-a[2], 0.0f, a[0]) / sqrtf(a[0] * a[0] + a[2] * a[2]);
    } else {
        b = Vec3T(0.0f, a[2], -a[1]) / sqrtf(a[1] * a[1] + a[2] * a[2]);
    }

    Vec3T c = cross(a, b);
    return std::make_pair(b, c);
}

Vec3T __hostdev__ sampleHenyeyGreenstein(float g, Vec3T w) {
    float invHG = (1.f - g * g) / (1.f - g + 2.f * g * rand());
    float cosTheta = (1.f + g * g - invHG * invHG) / (2.f * g);

    float sinTheta = sqrtf(1.f - cosTheta * cosTheta);

    float phi = 2.f * PI * rand();

    Vec3T u, v;

    std::tie(u, v) = coordinateSystem(w);

    return cosTheta * w + sinTheta * cosf(phi) * u + sinTheta * sinf(phi) * v;
}

struct World {
  Camera camera;
  GridT* grid;
  nanovdb::CoordBBox box;
};

void __device__ runRender(curandState* state, int width, int height, int start, int end, float* image, const World world);
float getDuration(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
}

class Integrator {
private:
    bool m_useCuda;
    nanovdb::GridHandle<BufferT>* m_handle;

public:
    Integrator(bool useCuda, nanovdb::GridHandle<BufferT>* handle)
        : m_useCuda(useCuda), m_handle(handle)
    {
        if (!this->hostGrid())
            throw std::runtime_error("GridHandle does not contain a valid host grid");
        
        if (!deviceGrid())
            throw std::runtime_error("GridHandle does not contain a valid device grid");
    }

    float start(int width, int height, float* image)
    {
        Camera camera(wBBoxDimZ(), wBBoxCenter(), width, height);
        World world = { camera, deviceGrid(), hostGrid()->tree().bbox() };

        auto t0 = ClockT::now();

        computeForEach(
            m_useCuda, width * height, 128, __FILE__, __LINE__, [width, height, image, world] __device__ (int start, int end, curandState * dStates) {
              runRender(dStates, width, height, start, end, image, world);
            });
        computeSync(m_useCuda, __FILE__, __LINE__);

        auto t1 = ClockT::now();
        return getDuration(t0, t1);
    }

private:
    GridT* hostGrid() { return m_handle->grid<float>(); }
    GridT* deviceGrid() { return m_handle->deviceGrid<float>(); }
    float wBBoxDimZ() { return (float)hostGrid()->worldBBox().dim()[2] * 2; }
    Vec3T wBBoxCenter() { return Vec3T(hostGrid()->worldBBox().min() + hostGrid()->worldBBox().dim() * 0.5f); }
};

float __device__ clamp(float min, float max, float value) {
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

void __device__ runRender(curandState* dStates, int width, int height, int start, int end, float* image, const World world) {
    // get an accessor.
    auto acc = world.grid->tree().getAccessor();

    for (int i = start; i < end; ++i) {
        Vec3T rayEye = world.camera.origin();
        Vec3T rayDir = world.camera.direction(i);

        // generate ray.
        RayT wRay(rayEye, rayDir);

        // transform the ray to the grid's index-space.
        RayT iRay = wRay.worldToIndexF(*world.grid);

        // clip to bounds.
        //if (iRay.clip(world.box) == false) {
        //    writeBuffer(image, i, width, height, 0.0f, 0.0f, Vec3T(0.0f, 0.0f, 0.0f));
        //    return;
        //}
        // integrate...
        const float dt = 0.5f;
        float       transmittance = 1.0f;
        bool unsetFirstHit = true;
        Vec3T firstHit;

        bool absorbed = false;
        Vec3T color{ 0.f,0.f,0.f };

        auto far = iRay.t0();

       	float maxSigma = nanovdb::getExtrema(*world.grid, world.grid->indexBBox());
        // float maxSigma = 1.f;
        float sigmaMAJ = maxSigma * (SIGMA_A + SIGMA_S);
        auto id = threadIdx.x + blockIdx.x * blockDim.x;

        curandState localState = dStates[i];

        for (unsigned int depth = 0; !absorbed && depth < DEPTH_LIMIT; ++depth) {
            float sigma = acc.getValue(CoordT::Floor(iRay(far))) * 0.1f;
            auto muA = sigma * SIGMA_A;
            auto muS = sigma * SIGMA_S;
            auto muT = muA + muS;
            
            float pathLength;
            if (sigma > 0.f) {
                pathLength = -logf(1.0f - curand_uniform(&localState)) / sigmaMAJ;
                pathLength *= 0.1f;
                pathLength = clamp(MIN_STEP, MAX_STEP, pathLength);
            } else {
                pathLength = MIN_STEP * 10.f;
            }
            far += pathLength;
            if (far > iRay.t1()) {
                break;
            }		

            if (sigma < 0.0f) {
                sigma = 0.001;
                // continue;
            }
            

            auto absorbtion = muA / sigmaMAJ;
            auto scattering = muS / sigmaMAJ;
            // printf("CALC ABS AND SCATTER\n");

            // auto nullPoint = std::max(0.0f, 1.f - sigmaMAJ - pathLength);
            auto nullPoint = 1.f - absorbtion - scattering;
            // printf("NULL POINT %.2f\n", nullPoint);

            float sampleAttenuation = exp(-(pathLength) * muT);
            transmittance *= sampleAttenuation;
            // TODO: Calculate PDF for Ray
            // based on muT

            float sample = curand_uniform(&localState);

            // null-scattering
            if (sample < nullPoint) 
            {
                // We use same vector and continue in same direction
                continue;
            }
            //absorption
            else if (sample < nullPoint + absorbtion) {
                color += Vec3T(0.5f, 0.0f, 0.0f);
                absorbed = true;
            }
            //scattering
            else
            {			
                constexpr float g = 0.5f; // TODO: Get g from grid
                rayDir = sampleHenyeyGreenstein(g, rayDir);

                Vec3T iRayOrigin = { iRay(far) }; 
                iRay = nanovdb::Ray<float>(iRayOrigin, rayDir);

                // clip to bounds.
                if (!iRay.clip(world.box)) {
                    // std::cout << "scattering failed";

                    break;
                }

                far = iRay.t0();
            }
        }
        writeBuffer(image, i, width, height, 0.0f, 1.0f - transmittance, color);
    }
}

#endif