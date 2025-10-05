# macOS Port Implementation Checklist

## Phase 1: CMake Configuration ✓ Ready to Implement

### Root CMakeLists.txt
- [ ] Add `option(USE_CUDA "Build with CUDA support" ON)` at the top
- [ ] Add platform detection for macOS (force USE_CUDA=OFF)
- [ ] Make `project()` language conditional on USE_CUDA
- [ ] Make `CMAKE_CUDA_STANDARD` conditional
- [ ] Make `CMAKE_CUDA_ARCHITECTURES` conditional
- [ ] Replace hardcoded vcpkg path with environment variable or conditional
- [ ] Pass USE_CUDA to subdirectories

### renderer/CMakeLists.txt
- [ ] Make source file list conditional (separate CUDA and CPU sources)
- [ ] Conditionally include `nanovdb.cu` only when USE_CUDA=ON
- [ ] Make CUDA properties conditional
- [ ] Replace `NANOVDB_USE_CUDA` definition to be conditional
- [ ] Add `USE_CUDA` or `USE_CPU` compile definition based on platform
- [ ] Make CUDA flags conditional
- [ ] Add `cpu_renderer.cpp` to sources when USE_CUDA=OFF

### CMakePresets.json
- [ ] Add `macos-xcode` preset with Xcode generator
- [ ] Add `macos-ninja` preset for command-line builds
- [ ] Set USE_CUDA=OFF for macOS presets
- [ ] Add CMAKE_OSX_DEPLOYMENT_TARGET

---

## Phase 2: Create Abstraction Headers ✓ Ready to Implement

### New File: renderer/buffer_interface.h
- [ ] Create header with include guards
- [ ] Add conditional includes (CudaDeviceBuffer vs HostBuffer)
- [ ] Implement HostBuffer class with same interface as CudaDeviceBuffer
  - [ ] `init(size_t, bool)` method
  - [ ] `clear()` method
  - [ ] `data()` method
  - [ ] `deviceData()` method (returns same as data() for CPU)
  - [ ] `deviceUpload()` method (no-op for CPU)
  - [ ] `deviceDownload()` method (no-op for CPU)
  - [ ] `size()` method
- [ ] Define `BufferT` typedef conditionally

### New File: renderer/random_interface.h
- [ ] Create header with include guards
- [ ] Add conditional includes (curand vs <random>)
- [ ] Define RandomState struct for CPU (wraps std::mt19937)
- [ ] Implement `random_init()` function
- [ ] Implement `random_uniform()` function
- [ ] Typedef or alias for CUDA version

### New File: renderer/compute_interface.h
- [ ] Create header with include guards
- [ ] Add conditional include of ComputePrimitives.cuh for CUDA
- [ ] Implement CPU versions:
  - [ ] `computeSync()`
  - [ ] `computeFill()`
  - [ ] `computeForEach()` with simple loop
- [ ] Consider adding OpenMP pragmas for parallel CPU execution

### New File: renderer/platform_macros.h
- [ ] Create header with include guards
- [ ] Define `DEVICE_FUNC` macro
  - `#ifdef USE_CUDA` → `__device__`
  - `#else` → `inline`
- [ ] Define `HOST_DEVICE_FUNC` macro
  - `#ifdef USE_CUDA` → `__hostdev__`
  - `#else` → `inline`
- [ ] Define `GLOBAL_FUNC` macro
  - `#ifdef USE_CUDA` → `__global__`
  - `#else` → (empty or error)

---

## Phase 3: Refactor Existing Files

### main.cpp
- [ ] Replace `#include <nanovdb/util/CudaDeviceBuffer.h>` with `#include "buffer_interface.h"`
- [ ] Remove `using BufferT = nanovdb::CudaDeviceBuffer;` (now in buffer_interface.h)
- [ ] Add conditional compilation for `runNanoVDB` vs `runCPU` function call
- [ ] Update function declarations with `#ifdef USE_CUDA`
- [ ] Update `runNano()` to call appropriate renderer

### image.h
- [ ] Replace `#include <nanovdb/util/CudaDeviceBuffer.h>` with `#include "buffer_interface.h"`
- [ ] Remove `using BufferT = nanovdb::CudaDeviceBuffer;`
- [ ] Test that existing methods still work with HostBuffer

### integrator.cuh → Split into Two Files

#### Option A: Keep as .cuh with heavy preprocessing
- [ ] Add `#include "buffer_interface.h"`
- [ ] Add `#include "random_interface.h"`  
- [ ] Add `#include "compute_interface.h"`
- [ ] Add `#include "platform_macros.h"`
- [ ] Replace all `__device__` with `DEVICE_FUNC`
- [ ] Replace all `__hostdev__` with `HOST_DEVICE_FUNC`
- [ ] Replace `curandState*` with `RandomState*`
- [ ] Replace `curand_uniform()` with `random_uniform()`
- [ ] Make `Integrator` class work with both BufferT types

#### Option B: Separate into integrator.h and integrator_cuda.cuh
- [ ] Create shared integrator.h with algorithm logic
- [ ] Keep CUDA-specific kernel in integrator_cuda.cuh
- [ ] Create integrator_cpu.cpp with CPU implementation

**Recommendation**: Start with Option A (simpler), move to Option B if needed

### ray.cuh
- [ ] Replace `#include <nanovdb/util/CudaDeviceBuffer.h>` with `#include "buffer_interface.h"`
- [ ] Verify no other CUDA-specific code exists
- [ ] Consider renaming to ray.h if no CUDA-specific code remains

### common.cuh
- [ ] Add `#include "platform_macros.h"`
- [ ] Replace `__hostdev__` with `HOST_DEVICE_FUNC`
- [ ] Make `#include "ComputePrimitives.cuh"` conditional
- [ ] Consider renaming to common.h

### ComputePrimitives.cuh
- [ ] This file stays CUDA-only
- [ ] Ensure it's only included when USE_CUDA is defined
- [ ] No changes needed to the file itself

---

## Phase 4: Create CPU Implementation

### New File: renderer/cpu_renderer.cpp
- [ ] Include necessary headers (buffer_interface, integrator, etc.)
- [ ] Implement `runCPU()` function
  - [ ] Accept `nanovdb::GridHandle<BufferT>&` and `Image&`
  - [ ] Create Integrator with useCuda=false
  - [ ] Call integrator.start()
  - [ ] Print timing information
- [ ] Ensure no CUDA-specific calls

### Optional: New File: renderer/parallel_cpu.h
- [ ] Add OpenMP support for parallel CPU execution
- [ ] Implement thread pool for better performance
- [ ] Add CPU-specific optimizations

---

## Phase 5: Testing & Validation

### Build Testing
- [ ] Test Windows build still works (with CUDA)
- [ ] Test macOS build compiles (without CUDA)
- [ ] Verify CMake configuration works for both platforms
- [ ] Check that all dependencies resolve correctly

### Runtime Testing
- [ ] Test sphere primitive rendering on CPU
- [ ] Test VDB file loading on macOS
- [ ] Verify OpenGL rendering works on macOS
- [ ] Compare output images (CPU vs CUDA)
- [ ] Performance profiling

### Integration Testing
- [ ] Test camera controls
- [ ] Test settings configuration
- [ ] Test ImGui interface on macOS
- [ ] Test window resizing and input

---

## Phase 6: Documentation & Cleanup

### Documentation
- [ ] Update README.md with macOS build instructions
- [ ] Add vcpkg setup instructions for macOS
- [ ] Document performance expectations (CPU vs GPU)
- [ ] Add troubleshooting section
- [ ] Create separate MACOS_BUILD.md if needed

### Code Cleanup
- [ ] Remove any debug print statements
- [ ] Add comments explaining platform-specific sections
- [ ] Ensure consistent code style
- [ ] Run linter/formatter

### CI/CD (Optional)
- [ ] Add GitHub Actions for macOS builds
- [ ] Add automated testing
- [ ] Setup dependency caching

---

## Common Issues & Solutions

### ❌ Issue: "CUDA not found" on macOS
**Solution**: Set `USE_CUDA=OFF` in CMake configuration

### ❌ Issue: vcpkg not found
**Solution**: Set `VCPKG_ROOT` environment variable

### ❌ Issue: OpenVDB linking errors
**Solution**: Install with `vcpkg install openvdb[nanovdb]:arm64-osx`

### ❌ Issue: OpenGL warnings on macOS
**Expected**: macOS shows deprecation warnings, ignore them (OpenGL 3.3 still works)

### ❌ Issue: Performance too slow on CPU
**Solution**: 
1. Reduce image resolution
2. Reduce pixel samples
3. Add OpenMP parallelization
4. Consider Metal compute shaders (advanced)

---

## Quick Start Commands

### Windows (CUDA):
```bash
cmake --preset=default
cmake --build build --config Debug
```

### macOS (CPU):
```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset=macos-xcode
cmake --build build --config Debug
# Or open build/MonteCarloRenderer.xcodeproj
```

---

## Estimated Time

- Phase 1 (CMake): **2-3 hours**
- Phase 2 (Abstractions): **3-4 hours**
- Phase 3 (Refactoring): **4-6 hours**
- Phase 4 (CPU impl): **2-3 hours**
- Phase 5 (Testing): **3-4 hours**
- Phase 6 (Docs): **1-2 hours**

**Total: 15-22 hours** (2-3 work days)

---

## Files to Create

1. ✅ `MACOS_PORT_INVESTIGATION.md` (this report)
2. ✅ `IMPLEMENTATION_CHECKLIST.md` (this checklist)
3. ⏳ `renderer/buffer_interface.h`
4. ⏳ `renderer/random_interface.h`
5. ⏳ `renderer/compute_interface.h`
6. ⏳ `renderer/platform_macros.h`
7. ⏳ `renderer/cpu_renderer.cpp`
8. ⏳ `MACOS_BUILD.md` (optional)

## Files to Modify

1. ⏳ `CMakeLists.txt` (root)
2. ⏳ `renderer/CMakeLists.txt`
3. ⏳ `CMakePresets.json`
4. ⏳ `main.cpp`
5. ⏳ `image.h`
6. ⏳ `integrator.cuh`
7. ⏳ `ray.cuh`
8. ⏳ `common.cuh`
9. ⏳ `README.md`

---

## Notes

- Keep the Windows/CUDA path working - don't break existing functionality
- Test frequently on both platforms if possible
- Commit after each phase for easy rollback
- Consider feature branches for major changes

**Legend**: ✅ Done | ⏳ To Do | ❌ Issue
