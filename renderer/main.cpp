// #include "Camera.h"
// #include "Window.h"

#include <openvdb/tools/LevelSetSphere.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/util/CudaDeviceBuffer.h> // required for CUDA memory management
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/GridBuilder.h>

#include <openvdb/openvdb.h>

#include "Window.h"
#include "Camera.h"

void version(const char* progName, int exitStatus = EXIT_SUCCESS)
{
    printf("\n%s was build against NanoVDB version %s\n", progName, nanovdb::Version().c_str());
    exit(exitStatus);
}

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::CudaDeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

extern void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int width, int height, BufferT& imageBuffer);

int main(void)
{
    try {
        auto handle = nanovdb::createLevelSetSphere<float, BufferT>(100);

        auto* dstGrid = handle.grid<float>(); // Get a (raw) pointer to the NanoVDB grid form the GridManager.
        if (!dstGrid)
            throw std::runtime_error("GridHandle does not contain a grid with value type float");

        // Access and print out a single value (inside the level set) from both grids
        printf("NanoVDB cpu: %4.2f\n", dstGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));

        const int numIterations = 50;

        const int width = 1024;
        const int height = 1024;
        BufferT   imageBuffer;
        imageBuffer.init(width * height * sizeof(float));

        runNanoVDB(handle, numIterations, width, height, imageBuffer);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}