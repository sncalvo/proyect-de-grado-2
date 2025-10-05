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

void __device__ runRenderOneSample(curandState* state, int width, int height, int start, int end, float* image, const GPUWorld world, Vec3T lightPosition);
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

    float start(int width, int height, float* image, int currentSample)
    {
        auto cameraPositionOffset = Settings::getInstance().cameraLocation;
        Camera camera(wBBoxDimZ(), wBBoxCenter(), width, height);
        float maxSigma = nanovdb::getExtrema(*hostGrid(), hostGrid()->indexBBox());
        GPUWorld world = { camera, deviceGrid(), hostGrid()->tree().bbox(), maxSigma };
        auto lightPosition = Settings::getInstance().lightLocation;

        auto t0 = ClockT::now();

        auto vectorLightPosition = Vec3T(lightPosition[0], lightPosition[1], lightPosition[2]);
        printf("Light position: (%.4f, %.4f, %.4f)\nRendering sample %d\n", vectorLightPosition[0], vectorLightPosition[1], vectorLightPosition[2], currentSample + 1);

        computeForEach(
            m_useCuda, width * height, 128, __FILE__, __LINE__, [width, height, image, world, vectorLightPosition] __device__ (int start, int end, curandState * dStates) {
              runRenderOneSample(dStates, width, height, start, end, image, world, vectorLightPosition);
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

void __device__ runRenderOneSample(curandState* dStates, int width, int height, int start, int end, float* image, const GPUWorld world, Vec3T lightPosition) {
    auto acc = world.grid->tree().getAccessor();
    auto id = threadIdx.x + blockIdx.x * blockDim.x;

    curandState* localState = &dStates[id];
    Vec3T rayEye = world.camera.origin();

    float sigmaMAJ = world.maxSigma * (SIGMA_A + SIGMA_S);

    for (int i = start; i < end; ++i) {
        // Render ONE sample for this pixel
        Vec4T color = DeltaTrackingIntegration<GPUWorld, Vec3T, Vec4T, RayT, nanovdb::DefaultReadAccessor<float>, CoordT>(
            i, world, rayEye, image, width, height, localState, sigmaMAJ, acc, lightPosition);
        
        // Write the sample result directly (no averaging here - handled by Image class)
        writeBuffer(image, i, width, height, 0.0f, 0.0f, Vec3T(color[0], color[1], color[2]));
    }
}

#endif