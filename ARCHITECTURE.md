# Architecture Transformation: Windows/CUDA → Cross-Platform

## Current Architecture (Windows + CUDA Only)

```
┌─────────────────────────────────────────────────────────────┐
│                         main.cpp                             │
│  - Entry point                                               │
│  - Loads VDB file with OpenVDB                              │
│  - Converts OpenVDB → NanoVDB                               │
│  - Creates window and UI                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──────────────────────────────────┐
                     ↓                                  ↓
        ┌────────────────────────┐      ┌─────────────────────────┐
        │   nanovdb.cu           │      │   GLRender.cpp          │
        │   (CUDA Kernel)        │      │   (OpenGL Rendering)    │
        ├────────────────────────┤      ├─────────────────────────┤
        │ - GPU ray tracing      │      │ - Visualize grid bounds │
        │ - Calls integrator     │      │ - OpenGL 3.3 Core       │
        │ - Device memory mgmt   │      │ - GLFW + GLAD + ImGui   │
        └──────────┬─────────────┘      └─────────────────────────┘
                   │                                    ↑
                   ↓                                    │
        ┌────────────────────────┐                     │
        │   integrator.cuh       │                     │
        │   (CUDA Device Code)   │                     │
        ├────────────────────────┤                     │
        │ - Delta tracking       │                     │
        │ - Path tracing         │                     │
        │ - Henyey-Greenstein    │                     │
        │ - Uses curand          │                     │
        └──────────┬─────────────┘                     │
                   │                                    │
    ┌──────────────┴──────────────┐                   │
    ↓                              ↓                   │
┌───────────────┐      ┌──────────────────────┐       │
│ image.h       │      │ ComputePrimitives    │       │
│ - CudaDevice  │      │ - Parallel kernels   │       │
│   Buffer      │      │ - CUDA wrappers      │       │
└───────────────┘      └──────────────────────┘       │
                                                       │
                        Dependencies                   │
                    ┌─────────────────┐               │
                    │ CUDA Runtime    │               │
                    │ curand          │               │
                    │ NanoVDB (CUDA)  │               │
                    │ OpenVDB         ├───────────────┤
                    │ OpenGL          │               │
                    │ GLFW            │               │
                    │ ImGui           │               │
                    └─────────────────┘               │
                              ↓                        │
                    ┌─────────────────┐               │
                    │ Windows Only    │───────────────┘
                    │ Visual Studio   │
                    │ vcpkg (Windows) │
                    └─────────────────┘
```

**Problems:**
- ❌ Hardcoded CUDA dependency
- ❌ No abstraction between GPU/CPU code
- ❌ Platform-specific CMake configuration
- ❌ Cannot build on macOS
- ❌ Tight coupling to CUDA APIs

---

## Target Architecture (Cross-Platform: Windows/CUDA + macOS/CPU)

```
┌─────────────────────────────────────────────────────────────┐
│                         main.cpp                             │
│  - Entry point (PLATFORM AGNOSTIC)                          │
│  - Loads VDB file with OpenVDB                              │
│  - Uses BufferT (abstracted)                                │
│  - Creates window and UI                                     │
└────────────────────┬───────────────────────────────────────┘
                     │
                     │  #ifdef USE_CUDA              #else
                     ├─────────────────────────────────────────┐
                     ↓                                         ↓
        ┌────────────────────────┐          ┌─────────────────────────┐
        │   nanovdb.cu           │          │   cpu_renderer.cpp      │
        │   (CUDA Path)          │          │   (CPU Path)            │
        ├────────────────────────┤          ├─────────────────────────┤
        │ - GPU ray tracing      │          │ - CPU ray tracing       │
        │ - Calls integrator     │          │ - Calls integrator      │
        │ - CudaDeviceBuffer     │          │ - HostBuffer            │
        └──────────┬─────────────┘          └───────────┬─────────────┘
                   │                                    │
                   └──────────────┬─────────────────────┘
                                  ↓
                   ┌──────────────────────────────────┐
                   │     Abstraction Layer            │
                   │  ┌────────────────────────────┐  │
                   │  │ buffer_interface.h         │  │
                   │  │ - BufferT typedef          │  │
                   │  │ - CudaDeviceBuffer OR      │  │
                   │  │   HostBuffer               │  │
                   │  └────────────────────────────┘  │
                   │  ┌────────────────────────────┐  │
                   │  │ random_interface.h         │  │
                   │  │ - RandomState typedef      │  │
                   │  │ - curand OR std::mt19937   │  │
                   │  └────────────────────────────┘  │
                   │  ┌────────────────────────────┐  │
                   │  │ compute_interface.h        │  │
                   │  │ - CUDA kernels OR          │  │
                   │  │   CPU loops                │  │
                   │  └────────────────────────────┘  │
                   │  ┌────────────────────────────┐  │
                   │  │ platform_macros.h          │  │
                   │  │ - __device__ OR inline     │  │
                   │  │ - __hostdev__ OR inline    │  │
                   │  └────────────────────────────┘  │
                   └──────────────┬───────────────────┘
                                  ↓
                   ┌──────────────────────────────────┐
                   │   integrator.h (Refactored)      │
                   │   (PLATFORM AGNOSTIC)            │
                   ├──────────────────────────────────┤
                   │ - Delta tracking algorithm       │
                   │ - Path tracing algorithm         │
                   │ - Henyey-Greenstein phase func   │
                   │ - Uses RandomState (abstracted)  │
                   │ - Uses DEVICE_FUNC macros        │
                   └──────────────┬───────────────────┘
                                  │
                   ┌──────────────┴───────────────┐
                   ↓                              ↓
        ┌──────────────────┐          ┌──────────────────┐
        │ image.h          │          │ ray.h            │
        │ - Uses BufferT   │          │ - Uses BufferT   │
        │ - Abstracted     │          │ - Abstracted     │
        └──────────────────┘          └──────────────────┘

                   ┌──────────────────────────────────┐
                   │      GLRender.cpp                │
                   │      (ALREADY CROSS-PLATFORM)    │
                   ├──────────────────────────────────┤
                   │ - OpenGL 3.3 Core                │
                   │ - Works on Windows & macOS       │
                   │ - GLFW + GLAD + ImGui            │
                   └──────────────────────────────────┘

                        Dependencies
         ┌─────────────────────────────────────────┐
         │                                         │
    ┌────┴─────────────┐              ┌───────────┴──────┐
    │  Windows Path    │              │   macOS Path     │
    ├──────────────────┤              ├──────────────────┤
    │ CUDA Runtime     │              │ (No CUDA)        │
    │ curand           │              │ <random>         │
    │ NanoVDB (CUDA)   │              │ OpenVDB only     │
    │ OpenVDB          │              │ OpenVDB          │
    │ OpenGL           │              │ OpenGL           │
    │ GLFW             │              │ GLFW             │
    │ ImGui            │              │ ImGui            │
    │ vcpkg (Windows)  │              │ vcpkg (macOS)    │
    │ Visual Studio    │              │ XCode or Ninja   │
    └──────────────────┘              └──────────────────┘
```

**Benefits:**
- ✅ Platform abstraction through interface headers
- ✅ Conditional compilation (#ifdef USE_CUDA)
- ✅ Same algorithms work on GPU and CPU
- ✅ OpenGL rendering unchanged
- ✅ Can build on Windows and macOS
- ✅ Easy to extend to Linux in future

---

## Key Abstraction Layers

### 1. Buffer Abstraction

| Aspect          | Windows/CUDA                | macOS/CPU              |
|-----------------|-----------------------------|-----------------------|
| Type            | `nanovdb::CudaDeviceBuffer` | `HostBuffer` (custom) |
| Memory Location | GPU device memory           | System RAM            |
| Upload          | cudaMemcpy(H→D)             | No-op                 |
| Download        | cudaMemcpy(D→H)             | No-op                 |
| Access          | Through device pointer      | Direct pointer        |

### 2. Random Number Abstraction

| Aspect          | Windows/CUDA      | macOS/CPU                      |
|-----------------|-------------------|--------------------------------|
| Type            | `curandState`     | `std::mt19937`                 |
| Initialization  | `curand_init()`   | `generator.seed()`             |
| Generation      | `curand_uniform()`| `dist(generator)`              |
| Quality         | XORWOW algorithm  | Mersenne Twister               |
| Performance     | Very fast (GPU)   | Fast (CPU)                     |

### 3. Compute Abstraction

| Aspect          | Windows/CUDA                | macOS/CPU                    |
|-----------------|-----------------------------|-----------------------------|
| Execution       | CUDA kernel <<<blocks,threads>>>| for loop                |
| Parallelism     | Thousands of threads        | Sequential (or OpenMP)      |
| Sync            | `cudaDeviceSynchronize()`   | No-op                       |
| Memory Ops      | `cudaMemset`, `cudaMemcpy`  | `memset`, `memcpy`          |

### 4. Function Decorators

| Decorator    | Windows/CUDA | macOS/CPU |
|--------------|--------------|-----------|
| `__device__` | GPU only     | `inline`  |
| `__host__`   | CPU only     | `inline`  |
| `__hostdev__`| Both         | `inline`  |
| `__global__` | Kernel entry | Error     |

---

## CMake Configuration Flow

### Before (Windows Only)
```cmake
project(MonteCarloRenderer LANGUAGES CXX CUDA)  # Always CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_TOOLCHAIN_FILE "C:/tools/vcpkg/...")  # Hardcoded
add_executable(mcrenderer nanovdb.cu ...)       # Always CUDA
target_compile_definitions(mcrenderer PRIVATE "NANOVDB_USE_CUDA")
```

### After (Cross-Platform)
```cmake
option(USE_CUDA "Build with CUDA support" ON)

if(APPLE)
    set(USE_CUDA OFF CACHE BOOL "" FORCE)  # No CUDA on macOS
endif()

if(USE_CUDA)
    project(MonteCarloRenderer LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75)
else()
    project(MonteCarloRenderer LANGUAGES CXX)
endif()

# Auto-detect vcpkg or use environment variable
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()

# Conditional sources
if(USE_CUDA)
    set(SOURCES nanovdb.cu main.cpp ...)
    target_compile_definitions(mcrenderer PRIVATE "USE_CUDA")
else()
    set(SOURCES cpu_renderer.cpp main.cpp ...)
    target_compile_definitions(mcrenderer PRIVATE "USE_CPU")
endif()

add_executable(mcrenderer ${SOURCES})
```

---

## Build Process Comparison

### Windows (CUDA)
```bash
# Setup
set VCPKG_ROOT=C:\tools\vcpkg
vcpkg install openvdb[nanovdb]:x64-windows

# Configure
cmake --preset=default

# Build
cmake --build build --config Debug

# Run
.\build\Debug\mcrenderer.exe
```

### macOS (CPU)
```bash
# Setup
export VCPKG_ROOT=/usr/local/vcpkg
vcpkg install openvdb[nanovdb]:arm64-osx  # or x64-osx

# Configure
cmake --preset=macos-xcode

# Build
cmake --build build --config Debug
# OR
open build/MonteCarloRenderer.xcodeproj

# Run
./build/Debug/mcrenderer
```

---

## Performance Characteristics

### Rendering Pipeline

| Stage                    | Windows/CUDA           | macOS/CPU              |
|--------------------------|------------------------|------------------------|
| VDB Loading              | ~100ms (same)          | ~100ms (same)          |
| Grid Upload              | 10-50ms (H→D transfer) | 0ms (already in RAM)   |
| Ray Tracing (1080x1080)  | 50-200ms (GPU)         | 5,000-20,000ms (CPU)   |
| Image Download           | 10-20ms (D→H transfer) | 0ms (already in RAM)   |
| OpenGL Rendering         | 1-2ms (same)           | 1-2ms (same)           |
| **Total Frame Time**     | ~100-300ms             | ~5-20 seconds          |

**CPU Performance Notes:**
- CPU version will be ~10-100x slower
- Can improve with:
  - Lower resolution (e.g., 512x512)
  - Fewer pixel samples
  - OpenMP parallelization
  - Compiler optimizations (-O3, -march=native)

---

## File Structure Before/After

### Before
```
renderer/
├── main.cpp              (CUDA-dependent)
├── nanovdb.cu            (CUDA kernel)
├── integrator.cuh        (CUDA device code)
├── image.h               (CudaDeviceBuffer)
├── ray.cuh               (CudaDeviceBuffer)
├── common.cuh            (CUDA macros)
├── ComputePrimitives.cuh (CUDA primitives)
├── GLRender.cpp          (OpenGL, cross-platform ✓)
├── Camera.h              (Pure C++, cross-platform ✓)
├── Window.h              (GLFW, cross-platform ✓)
└── settings.h            (Pure C++, cross-platform ✓)
```

### After
```
renderer/
├── main.cpp              (Abstracted, cross-platform ✓)
├── nanovdb.cu            (CUDA kernel, conditional)
├── cpu_renderer.cpp      (NEW - CPU implementation)
├── integrator.h          (Refactored, cross-platform ✓)
├── image.h               (Abstracted, cross-platform ✓)
├── ray.h                 (Abstracted, cross-platform ✓)
├── common.h              (Abstracted, cross-platform ✓)
├── buffer_interface.h    (NEW - Buffer abstraction)
├── random_interface.h    (NEW - RNG abstraction)
├── compute_interface.h   (NEW - Compute abstraction)
├── platform_macros.h     (NEW - Macro abstraction)
├── ComputePrimitives.cuh (CUDA primitives, conditional)
├── GLRender.cpp          (OpenGL, cross-platform ✓)
├── Camera.h              (Pure C++, cross-platform ✓)
├── Window.h              (GLFW, cross-platform ✓)
└── settings.h            (Pure C++, cross-platform ✓)
```

---

## Testing Strategy

### Phase 1: Compilation
1. ✓ Windows build compiles with CUDA
2. ✓ macOS build compiles without CUDA
3. ✓ All warnings resolved

### Phase 2: Basic Functionality
1. ✓ Window opens on both platforms
2. ✓ OpenGL context initializes
3. ✓ ImGui renders
4. ✓ VDB file loads

### Phase 3: Rendering
1. ✓ Simple sphere renders correctly (CPU vs GPU)
2. ✓ Complex VDB renders correctly
3. ✓ Output images match (within tolerance)

### Phase 4: Performance
1. ✓ Profile CPU bottlenecks
2. ✓ Optimize critical paths
3. ✓ Add parallel execution (optional)

---

## Future Enhancements

### Short Term
- Add OpenMP for CPU parallelization
- Implement progressive rendering (show partial results)
- Add Metal compute shader path for macOS (GPU acceleration)

### Medium Term
- Linux support (should be trivial after macOS)
- Vulkan compute shader path (cross-platform GPU)
- WebGPU support for browser

### Long Term
- Hybrid CPU+GPU rendering
- Distributed rendering across multiple machines
- Real-time preview mode

---

## Conclusion

The architecture transformation maintains the core rendering algorithms while abstracting platform-specific details. The OpenGL rendering path remains unchanged, and the same Monte Carlo path tracing logic works on both GPU (CUDA) and CPU (native C++).

**Key Insight**: By separating "what to compute" (algorithms) from "how to compute" (CUDA vs CPU), we achieve true cross-platform compatibility without duplicating business logic.
