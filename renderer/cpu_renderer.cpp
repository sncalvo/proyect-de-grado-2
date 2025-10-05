// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <iostream>

#include "image.h"
#include "settings.h"

// Include shared algorithm and CPU adapter
#include "integrator_common.h"
#include "integrator_cpu_adapter.h"

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

// Render function for each pixel
void runRender(
    RandomState* localState,
    int width, 
    int height, 
    int start, 
    int end, 
    float* image, 
    const CPUWorld& world, 
    Vec3T lightPosition, 
    unsigned int pixelSamples
) {
    auto acc = world.grid->getAccessor();
    Vec3T rayEye = world.camera.origin();
    float sigmaMAJ = world.maxSigma * (SIGMA_A + SIGMA_S);

    for (int i = start; i < end; ++i) {
        Vec4T totalColor;
        for (unsigned int j = 0; j < pixelSamples; ++j) {
            // Use shared algorithm
            Vec4T color = DeltaTrackingIntegration<CPUWorld, Vec3T, Vec4T, RayT, GridT::Accessor, CoordT>(
                i, world, rayEye, image, width, height, localState, sigmaMAJ, acc, lightPosition);
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
    CPUCamera camera(wBBoxDimZ, wBBoxCenter, width, height);
    
    // Get max sigma from grid - iterate through active voxels
    float maxSigma = 0.0f;
    for (auto iter = grid->cbeginValueOn(); iter; ++iter) {
        maxSigma = std::max(maxSigma, iter.getValue());
    }
    
    std::cout << "Max sigma (raw): " << maxSigma << std::endl;
    
    // Setup world
    CPUWorld world(
        camera, 
        grid, 
        grid->evalActiveVoxelBoundingBox(), 
        maxSigma,
        grid->transformPtr()
    );
    
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
