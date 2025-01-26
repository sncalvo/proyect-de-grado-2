#ifndef H_INTEGRATOR
#define H_INTEGRATOR

#include <chrono>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>

#include "ray.cuh"
#include "common.cuh"

#include <curand.h>
#include <curand_kernel.h>

using BufferT = nanovdb::CudaDeviceBuffer;
using GridT = nanovdb::FloatGrid;
using CoordT = nanovdb::Coord;
using RealT = float;
using Vec3T = nanovdb::Vec3<RealT>;
using Vec4T = nanovdb::Vec4<RealT>;
using RayT = nanovdb::Ray<RealT>;
using ClockT = std::chrono::high_resolution_clock;

constexpr unsigned int DEPTH_LIMIT = 100;
constexpr float SIGMA_A = 0.05f;
constexpr float SIGMA_S = 0.95f;
constexpr float MIN_STEP = 0.5f;
constexpr float MAX_STEP = 5.f;
constexpr float PI = 3.14159265358979323846f;
constexpr float DENSITY = 1.f;
constexpr unsigned int PIXEL_SAMPLES = 16;

Vec3T UP = Vec3T {0.0f, 1.0f, 0.0f};

Vec3T __hostdev__ cross(const Vec3T& a, const Vec3T& b) {
    return Vec3T(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

void __hostdev__ coordinateSystem(const Vec3T& a, Vec3T& b, Vec3T& c) {
    if (fabs(a[0]) > fabs(a[1])) {
        b = Vec3T(-a[2], 0.0f, a[0]) / sqrtf(a[0] * a[0] + a[2] * a[2]);
    } else {
        b = Vec3T(0.0f, a[2], -a[1]) / sqrtf(a[1] * a[1] + a[2] * a[2]);
    }

    c = cross(a, b);
}

Vec3T __hostdev__ sampleHenyeyGreenstein(float g, Vec3T w, float random1, float random2) {
    float invHG = (1.f - g * g) / (1.f - g + 2.f * g * random1);
    float cosTheta = (1.f + g * g - invHG * invHG) / (2.f * g);

    float sinTheta = sqrtf(1.f - cosTheta * cosTheta);

    float phi = 2.f * PI * random2;

    Vec3T u, v;

    coordinateSystem(w, u, v);

    return cosTheta * w + sinTheta * cosf(phi) * u + sinTheta * sinf(phi) * v;
}

struct World {
  Camera camera;
  GridT* grid;
  nanovdb::CoordBBox box;
  float maxSigma;
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
        float maxSigma = nanovdb::getExtrema(*hostGrid(), hostGrid()->indexBBox());
        World world = { camera, deviceGrid(), hostGrid()->tree().bbox(), maxSigma };

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

Vec4T __device__ DeltaTrackingIntegration(int i, const World& world, Vec3T& rayEye, float* image, int width, int height, curandState* localState, float sigmaMAJ, nanovdb::DefaultReadAccessor<float>& acc);

void __device__ runRender(curandState* dStates, int width, int height, int start, int end, float* image, const World world) {
    auto acc = world.grid->tree().getAccessor();
    auto id = threadIdx.x + blockIdx.x * blockDim.x;

    curandState* localState = &dStates[id];
    Vec3T rayEye = world.camera.origin();

    float sigmaMAJ = world.maxSigma * (SIGMA_A + SIGMA_S);


    for (int i = start; i < end; ++i) {
        Vec4T totalColor;
        for (int j = 0; j < PIXEL_SAMPLES; ++j) {
            Vec4T color = DeltaTrackingIntegration(i, world, rayEye, image, width, height, localState, sigmaMAJ, acc);
            totalColor += color;
        }
        totalColor /= PIXEL_SAMPLES;
        writeBuffer(image, i, width, height, 0.0f, 0.0f, Vec3T(totalColor[0], totalColor[1], totalColor[2]));
    }
}

Vec4T __device__ DeltaTrackingIntegration(int i, const World& world, Vec3T& rayEye, float* image, int width, int height, curandState* localState, float sigmaMAJ, nanovdb::DefaultReadAccessor<float>& acc) {
    Vec3T rayDir = world.camera.direction(i);

    RayT wRay(rayEye, rayDir);
    RayT iRay = wRay.worldToIndexF(*world.grid);

    auto backgroundColor = Vec3T(0.36f, 0.702f, 0.98f);

    if (!iRay.clip(world.box)) {
        auto color = Vec4T(backgroundColor[0], backgroundColor[1], backgroundColor[2], 0.0f);
        return color;
    }

    float transmittance = 1.0f;
    bool absorbed = false;
    Vec3T color{ 0.f,0.f,0.f };
    auto far = iRay.t0();

    for (unsigned int depth = 0; !absorbed && depth < DEPTH_LIMIT; ++depth) {
        auto coord = CoordT::Floor(iRay(far));

        if (false && !world.box.isInside(coord)) {
            break;
        }

        float sigma = 0.0f;
        if (world.grid->tree().isActive(coord)) {
            sigma = acc.getValue(coord) * DENSITY;
        }

        auto muA = sigma * SIGMA_A;
        auto muS = sigma * SIGMA_S;
        auto muT = muA + muS;

        float pathLength;
        if (sigma > 0.f) {
            pathLength = -logf(1.0f - curand_uniform(localState)) / sigmaMAJ;

            pathLength *= 0.1f;
            pathLength = clamp(MIN_STEP, MAX_STEP, pathLength);
        }
        else {
            pathLength = MIN_STEP * 10.f;
        }

        far += pathLength;
        auto t1 = iRay.t1();
        if (far > t1)
            break;

        if (sigma <= 0.f) {
            continue;
        }

        auto absorbtion = muA / sigmaMAJ;
        auto scattering = muS / sigmaMAJ;
        auto nullPoint = 1.f - absorbtion - scattering;

        float sampleAttenuation = exp(-(pathLength)*muT);
        transmittance *= sampleAttenuation;

        float sample = curand_uniform(localState);

        if (sample < nullPoint) {
            continue;
        }
        else if (sample < nullPoint + absorbtion) {
            color += Vec3T(1.f, 1.f, 1.f);
            absorbed = true;
        }
        else {
            float g = 0.2f;
            if (world.grid->tree().isActive(coord)) {
                g = acc.getValue(coord);
            }
            float sample1 = curand_uniform(localState);
            float sample2 = curand_uniform(localState);
            rayDir = sampleHenyeyGreenstein(g, rayDir, sample1, sample2);

            Vec3T iRayOrigin = iRay(far);

            iRay = nanovdb::Ray<float>(iRayOrigin, rayDir);

            if (!iRay.clip(world.box)) {
                absorbed = true;
                break;
            }
            far = iRay.t0();
        }
    }

    if (!absorbed) {
        color += backgroundColor;
    }

    return Vec4T(color[0], color[1], color[2], 1.f - transmittance);
}

#endif