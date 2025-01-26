// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>

#include "common.cuh"
#include "image.h"
#include "integrator.cuh"
#include "ray.cuh"

using BufferT = nanovdb::CudaDeviceBuffer;

void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, Image& image)
{
    using GridT = nanovdb::FloatGrid;
    using CoordT = nanovdb::Coord;
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using RayT = nanovdb::Ray<RealT>;

    handle.deviceUpload();

    Integrator integrator(true, &handle);
    auto duration = integrator.start(image.width(), image.height(), image.deviceUpload());
    std::cout << "Duration(NanoVDB-Cuda) = " << duration << " ms" << std::endl;
}
