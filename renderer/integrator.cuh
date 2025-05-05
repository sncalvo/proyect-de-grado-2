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
#include "settings.h"

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
constexpr float DENSITY = 0.1f;
constexpr unsigned int PIXEL_SAMPLES = 8;

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

void __device__ runRender(curandState* state, int width, int height, int start, int end, float* image, const World world, Vec3T lightPosition);
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
        auto cameraPositionOffset = Settings::getInstance().cameraLocation;
        Camera camera(wBBoxDimZ(), wBBoxCenter(), width, height);
        float maxSigma = nanovdb::getExtrema(*hostGrid(), hostGrid()->indexBBox());
        World world = { camera, deviceGrid(), hostGrid()->tree().bbox(), maxSigma };
        auto lightPosition = Settings::getInstance().lightLocation;

        auto t0 = ClockT::now();

        auto vectorLightPosition = Vec3T(lightPosition[0], lightPosition[1], lightPosition[2]);

        printf("Light position: (%.4f, %.4f, %.4f)\n", vectorLightPosition[0], vectorLightPosition[1], vectorLightPosition[2]);

        computeForEach(
            m_useCuda, width * height, 128, __FILE__, __LINE__, [width, height, image, world, vectorLightPosition] __device__ (int start, int end, curandState * dStates) {
              runRender(dStates, width, height, start, end, image, world, vectorLightPosition);
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

Vec4T __device__ DeltaTrackingIntegration(int i, const World& world, Vec3T& rayEye, float* image, int width, int height, curandState* localState, float sigmaMAJ, nanovdb::DefaultReadAccessor<float>& acc, Vec3T lightPosition);

void __device__ runRender(curandState* dStates, int width, int height, int start, int end, float* image, const World world, Vec3T lightPosition) {
    auto acc = world.grid->tree().getAccessor();
    auto id = threadIdx.x + blockIdx.x * blockDim.x;

    curandState* localState = &dStates[id];
    Vec3T rayEye = world.camera.origin();

    float sigmaMAJ = world.maxSigma * (SIGMA_A + SIGMA_S);


    for (int i = start; i < end; ++i) {
        Vec4T totalColor;
        for (int j = 0; j < PIXEL_SAMPLES; ++j) {
            Vec4T color = DeltaTrackingIntegration(i, world, rayEye, image, width, height, localState, sigmaMAJ, acc, lightPosition);
            totalColor += color;
        }
        totalColor /= PIXEL_SAMPLES;
        writeBuffer(image, i, width, height, 0.0f, 0.0f, Vec3T(totalColor[0], totalColor[1], totalColor[2]));
    }
}

float __device__ PointToPointMarching(
    const World& world, 
    Vec3T& startPos, // point in cloud
    Vec3T& endPos, // light position
    curandState* localState, 
    float sigmaMAJ, 
    nanovdb::DefaultReadAccessor<float>& acc,
    float pfar,
    RayT& piRay
) {
    // Calculate direction vector from start to end
    auto start = piRay(pfar);
    auto iEnd = world.grid->worldToIndex(endPos);
    auto wStart = piRay.indexToWorldF(*world.grid)(pfar);
    Vec3T rayDir = iEnd - start;
    float totalDistance = rayDir.length();
    auto transmittance = 1.0f;
    
    // Normalize the direction
    rayDir.normalize();

    // Set up rays in world and index space
    RayT iRay(start, rayDir);
    // RayT iRay = wRay.worldToIndexF(*world.grid);

    // If ray doesn't intersect with the grid, return background color. Might be unnecessary condition
    if (!iRay.clip(world.box)) {
        printf("RIP\n");
        return transmittance;
    }

    bool absorbed = false;
    // auto start = iRay.start();
    // printf("iRayStart: (%.4f, %.4f, %.4f), start: (%.4f, %.4f, %.4f)\n", iRay(pfar)[0], iRay(pfar)[1], iRay(pfar)[2], start[0], start[1], start[2]);
    auto far = iRay.t0();
    float distanceTraveled = 0.0f;

    while (true) {
        auto coord = CoordT::Floor(iRay(far));

        // Get current 3D position in world space
        Vec3T currentPos = iRay(far);
        
        // Calculate remaining distance
        Vec3T remainingVec = iEnd - currentPos;
        distanceTraveled = (start - currentPos).length();
        
        // If we've reached (or passed) the target point, break
        if (distanceTraveled >= totalDistance) {
            break;
        }

        // Sample the grid at current position
        float sigma = 0.0f;
        if (world.grid->tree().isActive(coord)) {
            sigma = acc.getValue(coord) * DENSITY;
        }

        auto muA = sigma * SIGMA_A;
        auto muS = sigma * SIGMA_S;
        auto muT = muA + muS;

        // Sample path length
        float pathLength;
        if (sigma > 0.f) {
            pathLength = -logf(1.0f - curand_uniform(localState)) / sigmaMAJ;
            pathLength *= 1.0f; // step size multiplier
            pathLength = clamp(MIN_STEP, MAX_STEP, pathLength);
        } else {
            pathLength = MIN_STEP * 10.f;
        }

        far += pathLength;
        auto t1 = iRay.t1();
        if (far > t1)
            return transmittance;

        if (sigma <= 0.f) {
            continue;
        }

        float sampleAttenuation = exp(-(pathLength)*muT);
        transmittance *= sampleAttenuation;

        if (transmittance < 0.05f) {
            float q = 0.75f; // check what q should be like

            float sample = curand_uniform(localState);
            if (sample < q) {
                transmittance = 0.f;
            } else {
                transmittance /= 1.0f - q;
            }
        }

        if (transmittance <= 0.0f) {
            return 0.f;
        }
    }

    return transmittance;
}

float __device__ dot(const Vec3T& a, const Vec3T& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

float __device__ henyey_greenstein(const float& g, const float& cos_theta)
{
    float denom = 1 + g * g - 2 * g * cos_theta;
    return 1 / (4 * PI) * (1 - g * g) / (denom * sqrtf(denom));
}

Vec3T __device__ vecFromRay(Vec3T vec)
{
    return Vec3T(vec[0], vec[1], vec[2]);
}

Vec4T __device__ DeltaTrackingIntegration(int i, const World& world, Vec3T& rayEye, float* image, int width, int height, curandState* localState, float sigmaMAJ, nanovdb::DefaultReadAccessor<float>& acc, Vec3T lightPosition) {
    Vec3T rayDir = world.camera.direction(i);

    RayT wRay(rayEye, rayDir);
    RayT iRay = wRay.worldToIndexF(*world.grid);

    auto backgroundColor = Vec3T(0.36f, 0.702f, 0.98f);

    if (!iRay.clip(world.box)) {
        auto color = Vec4T(backgroundColor[0], backgroundColor[1], backgroundColor[2], 1.0f);
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

            // pathLength *= 0.1f;
            pathLength = clamp(MIN_STEP, MAX_STEP, pathLength);
        }
        else {
            pathLength = MIN_STEP * 10.f;
        }

        auto destFar = far;
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
            color += Vec3T(0.0f, 0.0f, 0.0f);
            absorbed = true;
        }
        else {
            // sample direct light
            float g = 0.2f;
            if (world.grid->tree().isActive(coord)) {
                g = acc.getValue(coord);
            }

            // Vec3T lightPos(-0.92f, 100.f, 70.96f);
            auto positionInCloud = wRay(destFar);  // Convert from index to world coordinates
            auto iEye = CoordT::Floor(iRay(destFar)).asVec3d();
            auto start = iRay.start();

            // printf("first: (%.4f, %.4f, %.4f), second: (%.4f, %.4f, %.4f)\n", start[0], start[1], start[2], iEye[0], iEye[1], iEye[2]);

            float lightTransmittance = PointToPointMarching(world, positionInCloud, lightPosition, localState, sigmaMAJ, acc, destFar, iRay);
            float cosTheta = dot(rayDir, lightPosition - positionInCloud);
            // TODO: Missing GreensteinPDF
            float greensteinPDF = henyey_greenstein(g, cosTheta);

            // Check if greensteinPDF is NaN and replace with 0 if it is
            if (isnan(greensteinPDF)) {
                greensteinPDF = 0.0f;
            }
            // color += lightTransmittance * transmittance * Vec3T(1.f, 1.f, 1.f) * pathLength * greensteinPDF * 0.0001f;
            // printf(
            //     "rayEye: (%.4f, %.4f, %.4f), rayDir: (%.4f, %.4f, %.4f), iEye: (%.4f, %.4f, %.4f), eye: (%.4f, %4.f, %4.f), lightTransmittance: %.4f, transmittance: %.4f, greensteinPDF: %.4f\n",
            //     rayEye[0], rayEye[1], rayEye[2],
            //     rayDir[0], rayDir[1], rayDir[2],
            //     eye[0], eye[1], eye[2],
            //     iEye[0], iEye[1], iEye[2],
            //     lightTransmittance, transmittance, greensteinPDF
            // );

            // color += lightTransmittance * transmittance * Vec3T(1.f, 1.f, 1.f) * greensteinPDF * 100.f;
            color += lightTransmittance * transmittance * Vec3T(1.f, 1.f, 1.f);

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
        color += backgroundColor * transmittance;

        // // sample direct light
        // Vec3T lightPos(1.f, 1.f, 1.f);
        // float directLight = PointToPointMarching(world, rayEye, lightPos, image, width, height, localState, sigmaMAJ, acc);
        // float cosTheta = dot(rayDir, lightPos - rayEye);
        // // TODO: Missing GreensteinPDF
        // color += directLight * cosTheta * transmittance * backgroundColor;
    }

    return Vec4T(color[0], color[1], color[2], 1.f - transmittance);
}

#endif