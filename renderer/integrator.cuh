#ifndef H_INTEGRATOR
#define H_INTEGRATOR

#include <chrono>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/GridStats.h>

#include "common.cuh"
#include "settings.h"

// Include shared algorithm and GPU adapter
#include "integrator_common.h"
#include "integrator_gpu_adapter.h"

using BufferT = nanovdb::CudaDeviceBuffer;
using ClockT = std::chrono::high_resolution_clock;

void __device__ runRender(curandState* state, int width, int height, int start, int end, float* image, const GPUWorld world, Vec3T lightPosition, unsigned int pixelSamples);
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
        GPUWorld world = { camera, deviceGrid(), hostGrid()->tree().bbox(), maxSigma };
        auto lightPosition = Settings::getInstance().lightLocation;

        auto t0 = ClockT::now();

        auto vectorLightPosition = Vec3T(lightPosition[0], lightPosition[1], lightPosition[2]);
        auto pixelSamples = Settings::getInstance().pixelSamples;
        printf("Light position: (%.4f, %.4f, %.4f)\npixel samples: %d\n", vectorLightPosition[0], vectorLightPosition[1], vectorLightPosition[2], pixelSamples);

        computeForEach(
            m_useCuda, width * height, 128, __FILE__, __LINE__, [width, height, image, world, vectorLightPosition, pixelSamples] __device__ (int start, int end, curandState * dStates) {
              runRender(dStates, width, height, start, end, image, world, vectorLightPosition, pixelSamples);
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

void __device__ runRender(curandState* dStates, int width, int height, int start, int end, float* image, const GPUWorld world, Vec3T lightPosition, unsigned int pixelSamples) {
    auto acc = world.grid->tree().getAccessor();
    auto id = threadIdx.x + blockIdx.x * blockDim.x;

    curandState* localState = &dStates[id];
    Vec3T rayEye = world.camera.origin();

    float sigmaMAJ = world.maxSigma * (SIGMA_A + SIGMA_S);

    for (int i = start; i < end; ++i) {
        Vec4T totalColor;
        for (unsigned int j = 0; j < pixelSamples; ++j) {
            // Use shared algorithm
            Vec4T color = DeltaTrackingIntegration<GPUWorld, Vec3T, Vec4T, RayT, nanovdb::DefaultReadAccessor<float>, CoordT>(
                i, world, rayEye, image, width, height, localState, sigmaMAJ, acc, lightPosition);
            totalColor += color;
        }
        totalColor /= static_cast<float>(pixelSamples);
        writeBuffer(image, i, width, height, 0.0f, 0.0f, Vec3T(totalColor[0], totalColor[1], totalColor[2]));
    }
}

#endif