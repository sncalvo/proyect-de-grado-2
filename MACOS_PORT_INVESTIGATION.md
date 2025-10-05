# macOS/XCode Port Investigation Report

## Executive Summary

This document outlines the necessary changes to enable this Monte Carlo Renderer project to build and run on macOS using XCode, replacing CUDA/NanoVDB with OpenVDB for CPU-based rendering.

---

## Current State Analysis

### 1. CMake Configuration

#### Root CMakeLists.txt Issues:
- **Line 11**: Declares `CUDA` as a required language - needs to be conditional
- **Line 14**: Sets `CMAKE_CUDA_STANDARD 17` - should only be set when CUDA is available
- **Line 16**: Sets `CMAKE_CUDA_ARCHITECTURES 75` - CUDA-specific, needs to be conditional
- **Line 21**: Hardcoded Windows vcpkg path - needs platform detection or environment variable

#### renderer/CMakeLists.txt Issues:
- **Lines 3-11**: Sets CUDA-specific properties without conditions
- **Line 14**: `nanovdb.cu` file included unconditionally - this is CUDA code
- **Lines 47-48**: Compile definitions for both OpenVDB and CUDA are set unconditionally
  - `NANOVDB_USE_OPENVDB` 
  - `NANOVDB_USE_CUDA`
- **Lines 50-55**: CUDA compilation flags set unconditionally

#### CMakePresets.json Issues:
- **Line 7**: Generator hardcoded to "Visual Studio 17 2022"
- Needs an additional preset for XCode on macOS

---

### 2. Source Code Dependencies on CUDA/NanoVDB

#### Files with Direct CUDA Dependencies:

1. **nanovdb.cu** (CUDA kernel file)
   - Uses `nanovdb::CudaDeviceBuffer`
   - Calls CUDA-specific device upload/download methods
   - This entire file should be conditionally compiled

2. **integrator.cuh**
   - Includes `<curand.h>` and `<curand_kernel.h>` (lines 16-17)
   - Uses `curandState` throughout
   - Uses `__device__` and `__hostdev__` CUDA keywords
   - Relies on `nanovdb::CudaDeviceBuffer` (line 19)
   - Calls `computeForEach` with CUDA-specific lambda

3. **ComputePrimitives.cuh**
   - Includes `<cuda_runtime_api.h>` (line 9)
   - Includes `<curand_kernel.h>` (line 14)
   - Contains CUDA-specific functions:
     - `checkCUDA`, `checkErrorCUDA`
     - `__global__` kernel definitions
     - `cudaMemset`, `cudaMemcpy`, `cudaDeviceSynchronize`, etc.

4. **image.h**
   - Uses `nanovdb::CudaDeviceBuffer` for buffer management (line 5, 9, 14)
   - Calls `deviceUpload()` and `deviceDownload()` methods

5. **ray.cuh**
   - Includes `nanovdb/util/CudaDeviceBuffer.h` (line 5)
   - Uses CUDA-specific buffer type

6. **common.cuh**
   - Uses `__hostdev__` decorator (CUDA-specific)
   - Contains `ComputePrimitives.cuh` include

7. **main.cpp**
   - Lines 3-5: Includes NanoVDB CUDA utilities
   - Line 26: Typedef `BufferT = nanovdb::CudaDeviceBuffer`
   - Line 28: External function declaration expecting CUDA buffer
   - Lines 72-80: Uses `nanovdb::openToNanoVDB` converter

---

### 3. OpenGL Dependencies

The project currently uses:
- **GLFW3**: For window management (cross-platform, already included)
- **GLAD**: For OpenGL function loading (cross-platform, already in project)
- **OpenGL**: Using modern OpenGL 3.3 core profile

**Good News**: These are all cross-platform and should work on macOS without changes. The shaders use `#version 330 core` which is supported on macOS.

---

## Required Changes

### Phase 1: CMake Configuration Updates

#### 1.1 Root CMakeLists.txt

```cmake
# Add platform detection and optional CUDA support
option(USE_CUDA "Build with CUDA support" ON)

# Detect platform
if(APPLE)
    set(USE_CUDA OFF CACHE BOOL "CUDA not supported on macOS" FORCE)
endif()

# Configure languages based on CUDA availability
if(USE_CUDA)
    project(MonteCarloRenderer LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75)
else()
    project(MonteCarloRenderer LANGUAGES CXX)
endif()

# Make vcpkg toolchain path configurable
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
elseif(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # Fallback to hardcoded path for backward compatibility
    if(EXISTS "C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake")
    endif()
endif()
```

#### 1.2 renderer/CMakeLists.txt

```cmake
# Conditional source files
if(USE_CUDA)
    set(RENDERER_SOURCES
        nanovdb.cu
        # ... other sources
    )
else()
    set(RENDERER_SOURCES
        # CPU implementation (to be created)
        cpu_renderer.cpp
        # ... other sources (excluding .cu files)
    )
endif()

add_executable(mcrenderer ${RENDERER_SOURCES})

# Conditional compile definitions
if(USE_CUDA)
    target_compile_definitions(mcrenderer PRIVATE 
        "NANOVDB_USE_CUDA"
        "USE_CUDA"
    )
    
    set_target_properties(mcrenderer PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
    
    set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
    if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
        set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
    endif()
    
    set(CMAKE_CUDA_FLAGS "${NANOVDB_CUDA_EXTENDED_LAMBDA} -G -use_fast_math ${CMAKE_CUDA_FLAGS}")
else()
    target_compile_definitions(mcrenderer PRIVATE "USE_CPU")
endif()

target_compile_definitions(mcrenderer PRIVATE "NANOVDB_USE_OPENVDB")
```

#### 1.3 CMakePresets.json

Add macOS preset:

```json
{
    "version": 2,
    "configurePresets": [
        {
            "name": "default",
            "binaryDir": "${sourceDir}/build",
            "generator": "Visual Studio 17 2022",
            "architecture": "x64",
            "cacheVariables": {
                "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
                "USE_CUDA": "ON"
            }
        },
        {
            "name": "macos-xcode",
            "binaryDir": "${sourceDir}/build",
            "generator": "Xcode",
            "cacheVariables": {
                "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
                "USE_CUDA": "OFF",
                "CMAKE_OSX_DEPLOYMENT_TARGET": "10.15"
            }
        },
        {
            "name": "macos-ninja",
            "binaryDir": "${sourceDir}/build",
            "generator": "Ninja",
            "cacheVariables": {
                "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
                "USE_CUDA": "OFF",
                "CMAKE_BUILD_TYPE": "Debug",
                "CMAKE_OSX_DEPLOYMENT_TARGET": "10.15"
            }
        }
    ]
}
```

---

### Phase 2: Code Refactoring

#### 2.1 Create Abstraction Layer

Create new header files to abstract CUDA vs CPU implementations:

**renderer/buffer_interface.h** (NEW FILE):
```cpp
#pragma once

#include <vector>
#include <cstring>

#ifdef USE_CUDA
    #include <nanovdb/util/CudaDeviceBuffer.h>
    using BufferT = nanovdb::CudaDeviceBuffer;
#else
    // CPU buffer implementation
    class HostBuffer {
    public:
        HostBuffer() = default;
        
        void init(size_t size, bool zero = false);
        void clear();
        void* data();
        const void* data() const;
        void* deviceData(); // For CPU, same as data()
        void deviceUpload();
        void deviceDownload();
        size_t size() const;
        
    private:
        std::vector<uint8_t> m_data;
        size_t m_size = 0;
    };
    
    using BufferT = HostBuffer;
#endif
```

**renderer/random_interface.h** (NEW FILE):
```cpp
#pragma once

#ifdef USE_CUDA
    #include <curand.h>
    #include <curand_kernel.h>
    using RandomState = curandState;
#else
    #include <random>
    struct RandomState {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist{0.0f, 1.0f};
    };
    
    inline void random_init(RandomState* state, unsigned long long seed, unsigned long long id) {
        state->generator.seed(seed + id);
    }
    
    inline float random_uniform(RandomState* state) {
        return state->dist(state->generator);
    }
#endif
```

**renderer/compute_interface.h** (NEW FILE):
```cpp
#pragma once

#include <functional>

#ifdef USE_CUDA
    #include "ComputePrimitives.cuh"
#else
    // CPU implementations
    inline void computeSync(bool useCuda, const char* file, int line) {
        // No-op for CPU
    }
    
    inline void computeFill(bool useCuda, void* data, uint8_t value, size_t size) {
        std::memset(data, value, size);
    }
    
    template<typename FunctorT, typename... Args>
    inline void computeForEach(bool useCuda, int numItems, int blockSize, 
                               const char* file, int line, 
                               const FunctorT& op, Args... args) {
        // Simple sequential CPU implementation
        for (int i = 0; i < numItems; ++i) {
            op(i, i + 1, nullptr, args...);
        }
    }
#endif
```

#### 2.2 Refactor Existing Files

**main.cpp** changes:
```cpp
#include "buffer_interface.h"

// Remove direct CUDA includes, use abstractions
// Keep OpenVDB includes (both versions use it)

// BufferT is now defined in buffer_interface.h

#ifdef USE_CUDA
    extern void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, Image& image);
#else
    extern void runCPU(nanovdb::GridHandle<BufferT>& handle, Image& image);
#endif

void runNano(nanovdb::GridHandle<BufferT>* handle, Image* image) {
    auto lightPos = Settings::getInstance().lightLocation;
    std::cout << "Begin render with light" << lightPos[0] << "," 
              << lightPos[1] << "," << lightPos[2] << std::endl;
    image->clear();
    
#ifdef USE_CUDA
    runNanoVDB(*handle, *image);
#else
    runCPU(*handle, *image);
#endif
    
    image->save("raytrace_level_set-cpu.pfm");
    std::cout << "End render" << std::endl;
}
```

**image.h** changes:
```cpp
#include "buffer_interface.h"

// Rest of the class remains mostly the same
// BufferT is now platform-agnostic
```

**integrator.cuh → integrator.h** (rename and refactor):
- Replace `__device__` with `inline` for CPU builds
- Replace `__hostdev__` with `inline`
- Replace `curandState*` with `RandomState*`
- Replace `curand_uniform` with `random_uniform`
- Keep the algorithms the same (delta tracking, Henyey-Greenstein, etc.)

#### 2.3 Create CPU Renderer Implementation

**renderer/cpu_renderer.cpp** (NEW FILE):
```cpp
#include <chrono>
#include <iostream>

#include "buffer_interface.h"
#include "compute_interface.h"
#include "integrator.h"
#include "image.h"

using BufferT = HostBuffer;

void runCPU(nanovdb::GridHandle<BufferT>& handle, Image& image) {
    using GridT = nanovdb::FloatGrid;
    using RealT = float;
    
    // CPU version doesn't need device upload
    // Grid is already in host memory
    
    Integrator integrator(false, &handle);
    auto duration = integrator.start(image.width(), image.height(), image.deviceUpload());
    std::cout << "Duration(CPU-OpenVDB) = " << duration << " ms" << std::endl;
}
```

---

### Phase 3: Additional Considerations

#### 3.1 Performance Expectations

- **CUDA version**: GPU-accelerated, typically 10-100x faster for ray tracing
- **CPU version**: Will be significantly slower but functional
- Consider adding multi-threading (OpenMP or std::thread) to improve CPU performance

#### 3.2 NanoVDB vs OpenVDB

- **NanoVDB**: GPU-optimized data structure, very compact
- **OpenVDB**: Full-featured, CPU-oriented
- On macOS, we'll use OpenVDB's host-side grid directly
- The vcpkg.json already includes OpenVDB with nanovdb feature

#### 3.3 Build Instructions for macOS

```bash
# Set up vcpkg (if not already done)
export VCPKG_ROOT=/path/to/vcpkg

# Install dependencies
vcpkg install openvdb[nanovdb]:arm64-osx  # For Apple Silicon
vcpkg install openvdb[nanovdb]:x64-osx     # For Intel Macs

# Configure with CMake
cmake --preset=macos-xcode

# Build
cmake --build build --config Debug

# Or use Xcode directly
open build/MonteCarloRenderer.xcodeproj
```

#### 3.4 Testing Strategy

1. First, ensure the project compiles on macOS
2. Test with simple scenes (sphere primitives)
3. Verify OpenGL rendering works (should be straightforward)
4. Test with actual VDB files
5. Performance profiling and optimization

---

## Implementation Priority

### High Priority (Required for Basic Functionality)
1. ✅ CMake configuration changes (Platform detection, conditional CUDA)
2. ✅ Create buffer abstraction layer
3. ✅ Create compute abstraction layer  
4. ✅ Refactor main.cpp to use abstractions
5. ✅ Create CPU renderer implementation

### Medium Priority (Required for Full Functionality)
6. Refactor integrator to work with CPU
7. Test and debug rendering pipeline
8. Verify OpenGL compatibility on macOS

### Low Priority (Performance & Polish)
9. Add multi-threading to CPU renderer
10. Optimize data structures for CPU cache
11. Add macOS-specific optimizations
12. Create comprehensive build documentation

---

## Potential Issues & Solutions

### Issue 1: Random Number Generation
- **Problem**: CUDA uses curand, which is GPU-specific
- **Solution**: Use C++11 `<random>` with Mersenne Twister for CPU

### Issue 2: Device Memory Management
- **Problem**: CUDA uses explicit device/host transfers
- **Solution**: CPU buffer abstraction makes device/host operations no-ops

### Issue 3: Parallel Execution Model
- **Problem**: CUDA uses blocks and threads
- **Solution**: CPU version can use sequential loop or OpenMP

### Issue 4: __device__ and __host__ Decorators
- **Problem**: These are CUDA-specific
- **Solution**: Use preprocessor to make them `inline` or empty on CPU

### Issue 5: OpenGL Context on macOS
- **Problem**: macOS deprecated OpenGL
- **Solution**: Should still work, but may show warnings. OpenGL 3.3 core is still supported.

---

## Conclusion

The port to macOS is feasible and requires:
1. **~15-20% code changes** (mostly adding abstractions and CPU implementations)
2. **Moderate CMake refactoring**
3. **No changes to OpenGL rendering code** (already cross-platform)

The project is well-structured with clear separation between rendering logic and CUDA-specific code, which makes this port relatively straightforward.

**Estimated effort**: 2-3 days for experienced developer
- Day 1: CMake + abstractions
- Day 2: CPU implementation + refactoring
- Day 3: Testing and debugging
