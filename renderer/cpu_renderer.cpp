// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <iostream>

#include "image.h"
#include "settings.h"

// Include OpenVDB headers for CPU rendering
#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb/tools/RayIntersector.h>

// Include the rendering algorithm headers
#include "platform_macros.h"
#include "random_interface.h"

using RealT = float;
using Vec3T = openvdb::math::Vec3<RealT>;
using Vec4T = openvdb::math::Vec4<RealT>;
using RayT = openvdb::math::Ray<RealT>;
using CoordT = openvdb::Coord;
using GridT = openvdb::FloatGrid;

// Constants from integrator.cuh
constexpr unsigned int DEPTH_LIMIT = 100;
constexpr float SIGMA_A = 0.05f;
constexpr float SIGMA_S = 0.95f;
constexpr float MIN_STEP = 0.5f;
constexpr float MAX_STEP = 5.f;
constexpr float PI = 3.14159265358979323846f;
constexpr float DENSITY = 0.1f;
constexpr float FOV = 45.f;

// Camera structure for CPU
struct Camera
{
    float mWBBoxDimZ;
    Vec3T mWBBoxCenter;
    int width;
    int height;
    float aspect;
    float tanFOV;

    Camera(float wBBoxDimZ, Vec3T wBBoxCenter, int width, int height)
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

// World structure
struct World {
    Camera camera;
    GridT::Ptr grid;
    openvdb::CoordBBox box;
    float maxSigma;
    openvdb::math::Transform::Ptr transform;
};

// Utility functions
inline float clamp(float min, float max, float value) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

inline Vec3T cross(const Vec3T& a, const Vec3T& b) {
    return Vec3T(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

inline float dot(const Vec3T& a, const Vec3T& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline void coordinateSystem(const Vec3T& a, Vec3T& b, Vec3T& c) {
    if (fabs(a[0]) > fabs(a[1])) {
        b = Vec3T(-a[2], 0.0f, a[0]) / sqrtf(a[0] * a[0] + a[2] * a[2]);
    } else {
        b = Vec3T(0.0f, a[2], -a[1]) / sqrtf(a[1] * a[1] + a[2] * a[2]);
    }
    c = cross(a, b);
}

inline Vec3T sampleHenyeyGreenstein(float g, Vec3T w, float random1, float random2) {
    float invHG = (1.f - g * g) / (1.f - g + 2.f * g * random1);
    float cosTheta = (1.f + g * g - invHG * invHG) / (2.f * g);
    float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
    float phi = 2.f * PI * random2;

    Vec3T u, v;
    coordinateSystem(w, u, v);

    return cosTheta * w + sinTheta * cosf(phi) * u + sinTheta * sinf(phi) * v;
}

inline float henyey_greenstein(const float& g, const float& cos_theta)
{
    float denom = 1 + g * g - 2 * g * cos_theta;
    return 1 / (4 * PI) * (1 - g * g) / (denom * sqrtf(denom));
}

inline float balanceHeuristic(float pdf1, int n1, float pdf2, int n2) {
    return (n1 * pdf1) / (n1 * pdf1 + n2 * pdf2);
}

inline void writeBuffer(float* outImage, int i, int w, int h, float value, float alpha, const Vec3T& color)
{
    int offset = i * 3;
    const float bgr = 0.f;
    const float bgg = 0.f;
    const float bgb = 0.f;
    outImage[offset] = color[0] * (1.f - alpha) + alpha * value + (1.0f - alpha) * bgr;
    outImage[offset + 1] = color[1] * (1.f - alpha) + alpha * value + (1.0f - alpha) * bgg;
    outImage[offset + 2] = color[2] * (1.f - alpha) + alpha * value + (1.0f - alpha) * bgb;
}

// Point-to-point marching for shadow rays (light transmission)
float PointToPointMarching(
    const World& world, 
    const Vec3T& startPos,
    const Vec3T& endPos,
    RandomState* localState, 
    float sigmaMAJ, 
    GridT::Accessor& acc,
    float pfar,
    RayT& piRay
) {
    Vec3T start = Vec3T(piRay(pfar));
    Vec3T iEnd = Vec3T(world.transform->worldToIndex(openvdb::math::Vec3d(endPos)));
    Vec3T rayDir = iEnd - start;
    float totalDistance = rayDir.length();
    auto transmittance = 1.0f;
    
    rayDir.normalize();

    RayT iRay(start, rayDir);

    // Clip ray to grid bounding box (in index space)
    openvdb::math::BBox<Vec3T> bbox(Vec3T(world.box.min().asVec3d()), Vec3T(world.box.max().asVec3d()));
    if (!iRay.clip(bbox)) {
        return transmittance;
    }

    bool absorbed = false;
    auto far = std::max(iRay.t0(), 0.0f);
    float distanceTraveled = 0.0f;

    while (true) {
        Vec3T currentPos = iRay(far);
        auto coord = CoordT(int(std::floor(currentPos.x())), 
                           int(std::floor(currentPos.y())), 
                           int(std::floor(currentPos.z())));

        Vec3T remainingVec = iEnd - currentPos;
        distanceTraveled = (start - currentPos).length();
        
        if (distanceTraveled >= totalDistance) {
            break;
        }

        float sigma = 0.0f;
        if (world.grid->tree().isValueOn(coord)) {
            sigma = acc.getValue(coord) * DENSITY;
        }

        auto muA = sigma * SIGMA_A;
        auto muS = sigma * SIGMA_S;
        auto muT = muA + muS;

        float pathLength;
        if (sigma > 0.f) {
            pathLength = -logf(1.0f - curand_uniform(localState)) / sigmaMAJ;
            pathLength = clamp(MIN_STEP, MAX_STEP, pathLength);
        } else {
            pathLength = MIN_STEP * 10.f;
        }

        far += pathLength;
        if (far > iRay.t1())
            return transmittance;

        if (sigma <= 0.f) {
            continue;
        }

        float sampleAttenuation = exp(-(pathLength)*muT);
        transmittance *= sampleAttenuation;

        if (transmittance < 0.05f) {
            float q = 0.75f;
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

// Delta tracking integration - main rendering algorithm
Vec4T DeltaTrackingIntegration(
    int i, 
    const World& world, 
    const Vec3T& rayEye, 
    float* image, 
    int width, 
    int height, 
    RandomState* localState, 
    float sigmaMAJ, 
    GridT::Accessor& acc, 
    const Vec3T& lightPosition
) {
    Vec3T rayDir = world.camera.direction(i);

    RayT wRay(rayEye, rayDir);
    // Convert world ray to index space
    Vec3T iRayOrigin = Vec3T(world.transform->worldToIndex(openvdb::math::Vec3d(wRay.eye())));
    Vec3T iRayDir = Vec3T(world.transform->worldToIndex(openvdb::math::Vec3d(wRay.eye() + wRay.dir()))) - iRayOrigin;
    iRayDir.normalize();
    RayT iRay(iRayOrigin, iRayDir);

    auto backgroundColor = Vec3T(0.36f, 0.702f, 0.98f);

    // Clip ray to grid bounding box
    openvdb::math::BBox<Vec3T> bbox(Vec3T(world.box.min().asVec3d()), Vec3T(world.box.max().asVec3d()));
    if (!iRay.clip(bbox)) {
        auto color = Vec4T(backgroundColor[0], backgroundColor[1], backgroundColor[2], 1.0f);
        return color;
    }

    float transmittance = 1.0f;
    bool absorbed = false;
    Vec3T color{ 0.f, 0.f, 0.f };
    auto far = std::max(iRay.t0(), 0.0f);

    auto totalRayPDF = 1.f;

    for (unsigned int depth = 0; !absorbed && depth < DEPTH_LIMIT; ++depth) {
        Vec3T currentPos = iRay(far);
        auto coord = CoordT(int(std::floor(currentPos.x())), 
                           int(std::floor(currentPos.y())), 
                           int(std::floor(currentPos.z())));

        float sigma = 0.0f;
        if (world.grid->tree().isValueOn(coord)) {
            sigma = acc.getValue(coord) * DENSITY;
        }

        auto muA = sigma * SIGMA_A;
        auto muS = sigma * SIGMA_S;
        auto muT = muA + muS;

        float pathLength;
        if (sigma > 0.f) {
            pathLength = -logf(1.0f - curand_uniform(localState)) / sigmaMAJ;
            pathLength = clamp(MIN_STEP, MAX_STEP, pathLength);
        }
        else {
            pathLength = MIN_STEP * 10.f;
        }

        auto destFar = far;
        far += pathLength;
        if (far > iRay.t1())
            break;

        if (sigma <= 0.f) {
            continue;
        }

        auto absorbtion = muA / sigmaMAJ;
        auto scattering = muS / sigmaMAJ;
        auto nullPoint = 1.f - absorbtion - scattering;

        float sampleAttenuation = exp(-(pathLength)*muT);
        transmittance *= sampleAttenuation;

        auto pdfModifier = muT * sampleAttenuation;
        totalRayPDF *= pdfModifier;

        float sample = curand_uniform(localState);

        if (sample < nullPoint) {
            continue;
        }
        else if (sample < nullPoint + absorbtion) {
            color += Vec3T(0.0f, 0.0f, 0.0f);
            absorbed = true;
        }
        else {
            // Sample direct light
            float g = 0.2f;
            if (world.grid->tree().isValueOn(coord)) {
                g = acc.getValue(coord);
            }

            auto positionInCloud = wRay(destFar);
            
            float lightTransmittance = PointToPointMarching(world, positionInCloud, lightPosition, localState, sigmaMAJ, acc, destFar, iRay);
            float cosTheta = dot(rayDir, lightPosition - positionInCloud);
            float greensteinPDF = henyey_greenstein(g, cosTheta);

            // Check if greensteinPDF is NaN and replace with 0 if it is
            if (std::isnan(greensteinPDF)) {
                greensteinPDF = 0.0f;
            }

            float sample1 = curand_uniform(localState);
            float sample2 = curand_uniform(localState);
            auto previousDirection = rayDir;
            rayDir = sampleHenyeyGreenstein(g, rayDir, sample1, sample2);
            cosTheta = dot(rayDir, previousDirection);

            float newPDF = henyey_greenstein(g, cosTheta);
            auto sectionPDF = balanceHeuristic(newPDF, 1, greensteinPDF, 1);
            
            color += lightTransmittance * transmittance * Vec3T(1.f, 1.f, 1.f);

            Vec3T newIRayOrigin = iRay(far);
            Vec3T newIRayDir = Vec3T(world.transform->worldToIndex(openvdb::math::Vec3d(wRay.eye() + rayDir))) - Vec3T(world.transform->worldToIndex(openvdb::math::Vec3d(wRay.eye())));
            newIRayDir.normalize();
            iRay = RayT(newIRayOrigin, newIRayDir);

            // Clip new ray to bounding box
            if (!iRay.clip(bbox)) {
                absorbed = true;
                break;
            }
            far = std::max(iRay.t0(), 0.0f);
        }
    }

    if (!absorbed) {
        color += backgroundColor * transmittance;
    }

    return Vec4T(color[0], color[1], color[2], 1.f - transmittance);
}

// Render function for each pixel
void runRender(
    RandomState* localState,
    int width, 
    int height, 
    int start, 
    int end, 
    float* image, 
    const World& world, 
    Vec3T lightPosition, 
    unsigned int pixelSamples
) {
    auto acc = world.grid->getAccessor();
    Vec3T rayEye = world.camera.origin();
    float sigmaMAJ = world.maxSigma * (SIGMA_A + SIGMA_S);

    for (int i = start; i < end; ++i) {
        Vec4T totalColor;
        for (unsigned int j = 0; j < pixelSamples; ++j) {
            Vec4T color = DeltaTrackingIntegration(i, world, rayEye, image, width, height, localState, sigmaMAJ, acc, lightPosition);
            totalColor += color;
        }
        totalColor /= static_cast<float>(pixelSamples);
        writeBuffer(image, i, width, height, 0.0f, 0.0f, Vec3T(totalColor[0], totalColor[1], totalColor[2]));
    }
}

// Main CPU rendering entry point
void runCPU(openvdb::FloatGrid::Ptr grid, Image& image)
{
    std::cout << "Running CPU renderer with Delta Tracking integration (OpenVDB)..." << std::endl;
    
    if (!grid) {
        throw std::runtime_error("Invalid OpenVDB grid");
    }
    
    // Get grid bounding box in world space
    auto wBBox = grid->evalActiveVoxelBoundingBox();
    auto wBBoxMin = grid->indexToWorld(wBBox.min().asVec3d());
    auto wBBoxMax = grid->indexToWorld(wBBox.max().asVec3d());
    auto wBBoxDim = wBBoxMax - wBBoxMin;
    
    std::cout << "Grid bounds (world): min=" << wBBoxMin << ", max=" << wBBoxMax << std::endl;
    std::cout << "Image size: " << image.width() << "x" << image.height() << std::endl;
    
    // Get image dimensions
    const int width = image.width();
    const int height = image.height();
    const int numPixels = width * height;
    
    // Setup camera and world
    float wBBoxDimZ = wBBoxDim.z() * 2;
    Vec3T wBBoxCenter = (wBBoxMin + wBBoxMax) * 0.5f;
    Camera camera(wBBoxDimZ, wBBoxCenter, width, height);
    
    // Get max sigma from grid - iterate through active voxels
    float maxSigma = 0.0f;
    for (auto iter = grid->cbeginValueOn(); iter; ++iter) {
        maxSigma = std::max(maxSigma, iter.getValue());
    }
    
    std::cout << "Max sigma (raw): " << maxSigma << std::endl;
    
    // Setup world
    World world = { 
        camera, 
        grid, 
        grid->evalActiveVoxelBoundingBox(), 
        maxSigma,
        grid->transformPtr()
    };
    
    // Get settings
    auto lightLocation = Settings::getInstance().lightLocation;
    Vec3T lightPosition(lightLocation[0], lightLocation[1], lightLocation[2]);
    unsigned int pixelSamples = Settings::getInstance().pixelSamples;
    
    std::cout << "Light position: (" << lightPosition[0] << ", " << lightPosition[1] << ", " << lightPosition[2] << ")" << std::endl;
    std::cout << "Pixel samples: " << pixelSamples << std::endl;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Get image buffer
    float* imageData = reinterpret_cast<float*>(image.deviceUpload());
    
    std::cout << "Rendering " << numPixels << " pixels..." << std::endl;
    
    // Initialize random state for CPU
    RandomState randomState;
    random_init(&randomState, 42, 0, 0);
    
    // Render all pixels
    runRender(&randomState, width, height, 0, numPixels, imageData, world, lightPosition, pixelSamples);
    
    // Note: For CPU builds, deviceDownload is a no-op since data is already in RAM
    image.deviceDownload();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    
    std::cout << "Duration(CPU-OpenVDB) = " << duration << " ms" << std::endl;
    std::cout << "CPU rendering complete!" << std::endl;
}
