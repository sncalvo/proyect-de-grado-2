#include <openvdb/tools/LevelSetSphere.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/util/OpenToNanoVDB.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/util/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/GridBuilder.h>

// Platform-specific buffer interface
#include "buffer_interface.h"

#include <openvdb/openvdb.h>

#include "Window.h"
#include "Camera.h"

#include "image.h"
#include "settings.h"

#include "GLRender.h"

void version(const char* progName, int exitStatus = EXIT_SUCCESS)
{
    printf("\n%s was build against NanoVDB\n", progName);
    exit(exitStatus);
}

// BufferT is now defined in buffer_interface.h

#ifdef USE_CUDA
    extern void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, Image& image);
#else
    extern void runCPU(openvdb::FloatGrid::Ptr grid, Image& image);
#endif

#ifdef USE_CUDA
// Render ONE sample - called from window loop
void runNanoOneSample(nanovdb::GridHandle<BufferT>* handle, Image* image, MCRenderer::SampleWindow* window) {
    // Render one sample for all pixels
    runNanoVDB(*handle, *image);
    
    // Accumulate this sample
    image->accumulateSample();
    
    // Update display progressively
    if (window) {
        window->loadImageToPixels(image);
    }
}
#else
// Render ONE sample - called from window loop
void runNanoOneSample(openvdb::FloatGrid::Ptr* grid, Image* image, MCRenderer::SampleWindow* window) {
    // Render one sample for all pixels
    runCPU(*grid, *image);
    
    // Accumulate this sample
    image->accumulateSample();
    
    // Update display progressively
    if (window) {
        window->loadImageToPixels(image);
    }
}
#endif

// Start rendering - called when user clicks "Render" button
void startRender(Image* image) {
    auto lightPos = Settings::getInstance().lightLocation;
    unsigned int pixelSamples = Settings::getInstance().pixelSamples;
    
    std::cout << "Begin progressive render with light (" << lightPos[0] << "," << lightPos[1] << "," << lightPos[2] << ")" << std::endl;
    std::cout << "Total samples to render: " << pixelSamples << std::endl;
    
    // Clear and reset accumulation
    image->clear();
}

int main(int ac, char** av)
{
    auto render = MCRenderer::GLRender();
    try {
        openvdb::initialize();
        
#ifdef USE_CUDA
        nanovdb::GridHandle<BufferT> handle;
#else
        openvdb::FloatGrid::Ptr grid;
#endif
        
        if (true || ac > 1) {
            //const auto image = av[1];
            const auto image = "bunny_cloud.vdb";
            
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

            openvdb::FloatGrid::Ptr vdbGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
            render.setGrid(vdbGrid);

#ifdef USE_CUDA
            // Convert the OpenVDB grid into a NanoVDB grid handle.
            auto handle = nanovdb::tools::openToNanoVDB(vdbGrid);
            // Define a (raw) pointer to the NanoVDB grid on the host. Note we match the value type of the srcGrid!
            auto* dstGrid = handle.grid<float>();
            if (!dstGrid)
                throw std::runtime_error("GridHandle does not contain a grid with value type float");

            nanovdb::io::writeGrid("bunny.nvdb", handle); // Write the NanoVDB grid to file and throw if writing fails
            handle = nanovdb::io::readGrid<BufferT>("bunny.nvdb");
            std::cout << "Loaded NanoVDB grid[" << handle.gridMetaData()->shortGridName() << "]...\n";
            
            if (handle.gridMetaData()->isFogVolume() == false) {
                throw std::runtime_error("Grid must be a fog volume");
            }
#else
            // For CPU, use OpenVDB directly
            grid = vdbGrid;
            std::cout << "Loaded OpenVDB grid for CPU rendering...\n";
#endif
        } else {
#ifdef USE_CUDA
            handle = nanovdb::tools::createFogVolumeSphere<float, BufferT>(100.0f, nanovdb::Vec3d(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
            if (handle.gridMetaData()->isFogVolume() == false) {
                throw std::runtime_error("Grid must be a fog volume");
            }
#else
            // For CPU, create sphere using OpenVDB
            throw std::runtime_error("Procedural generation not yet implemented for CPU mode");
#endif
        }

        const int width = 1080;
        const int height = 1080;

        Image image(width, height);

        MCRenderer::SampleWindow window("Raytracing", MCRenderer::Camera(), 1.0f);
        render.init();
        window.setRenderer(&render);

        // Lambda called when user clicks "Render" button
        std::function<void()> startRenderLambda = [&]() {
            startRender(&image);
            
            // Capture the target sample count at start - don't read it dynamically
            unsigned int targetSamples = Settings::getInstance().pixelSamples;
            
            // Set up progressive rendering callback
#ifdef USE_CUDA
            window.startProgressiveRender([&, targetSamples]() {
                unsigned int currentSample = image.getCurrentSample();
                if (currentSample < targetSamples) {
                    runNanoOneSample(&handle, &image, &window);
                    std::cout << "Progress: " << image.getCurrentSample() << "/" << targetSamples << " samples completed" << std::endl;
                } else {
                    window.stopProgressiveRender();
                    image.save("raytrace_level_set-nanovdb-cuda.pfm");
                    std::cout << "Rendering complete: " << targetSamples << " samples accumulated and saved" << std::endl;
                }
            }, &image);
#else
            window.startProgressiveRender([&, targetSamples]() {
                unsigned int currentSample = image.getCurrentSample();
                if (currentSample < targetSamples) {
                    runNanoOneSample(&grid, &image, &window);
                    std::cout << "Progress: " << image.getCurrentSample() << "/" << targetSamples << " samples completed" << std::endl;
                } else {
                    window.stopProgressiveRender();
                    image.save("raytrace_level_set-cpu.pfm");
                    std::cout << "Rendering complete: " << targetSamples << " samples accumulated and saved" << std::endl;
                }
            }, &image);
#endif
        };

        window.run(startRenderLambda);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}