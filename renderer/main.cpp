#include <openvdb/tools/LevelSetSphere.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/OpenToNanoVDB.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/util/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/util/CudaDeviceBuffer.h> // required for CUDA memory management
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/GridBuilder.h>

#include <openvdb/openvdb.h>

#include "Window.h"
#include "Camera.h"

#include "image.h"

void version(const char* progName, int exitStatus = EXIT_SUCCESS)
{
    printf("\n%s was build against NanoVDB version %s\n", progName, nanovdb::Version().c_str());
    exit(exitStatus);
}

using BufferT = nanovdb::CudaDeviceBuffer;

extern void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, Image& image);

int main(int ac, char** av)
{
    try {
        nanovdb::GridHandle<BufferT> handle;
        if (true || ac > 1) {
            {
                //const auto image = av[1];
                const auto image = "bunny_cloud.vdb";
                openvdb::initialize();
                // Create a VDB file object.
                openvdb::io::File file(image);
                file.open();
                openvdb::GridBase::Ptr baseGrid;
                for (openvdb::io::File::NameIterator nameIter = file.beginName();
                    nameIter != file.endName(); ++nameIter)
                {
                    // Read in only the grid we are interested in.
                    if (nameIter.gridName() == "density")
                    {
                        baseGrid = file.readGrid(nameIter.gridName());
                    }
                    else
                    {
                        std::cout << "skipping grid " << nameIter.gridName() << std::endl;
                    }
                }
                file.close();

                openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

                // Convert the OpenVDB grid into a NanoVDB grid handle.
                auto handle = nanovdb::openToNanoVDB(grid);
                // Define a (raw) pointer to the NanoVDB grid on the host. Note we match the value type of the srcGrid!
                auto* dstGrid = handle.grid<float>();
                if (!dstGrid)
                    throw std::runtime_error("GridHandle does not contain a grid with value type float");

                nanovdb::io::writeGrid("bunny.nvdb", handle); // Write the NanoVDB grid to file and throw if writing fails
            }
            handle = nanovdb::io::readGrid<BufferT>("bunny.nvdb");
            std::cout << "Loaded NanoVDB grid[" << handle.gridMetaData()->shortGridName() << "]...\n";
        } else {
            handle = nanovdb::createFogVolumeSphere<float, BufferT>(100.0f, nanovdb::Vec3d(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
        }

        if (handle.gridMetaData()->isFogVolume() == false) {
            throw std::runtime_error("Grid must be a fog volume");
        }

        const int width = 1080;
        const int height = 1080;
        // BufferT   imageBuffer;
        // imageBuffer.init(width * height * 3 * sizeof(float));

        Image image(width, height);

        runNanoVDB(handle, image);

        image.save("raytrace_level_set-nanovdb-cuda-1.pfm");
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}