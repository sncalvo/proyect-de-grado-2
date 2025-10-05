# Building on macOS

This guide covers how to build the Monte Carlo Renderer on macOS. The macOS version runs in **CPU mode** (no CUDA) using OpenVDB for volumetric data structures.

## Prerequisites

### 1. Install Xcode Command Line Tools

```bash
xcode-select --install
```

### 2. Install vcpkg (Package Manager)

```bash
# Clone vcpkg
cd ~
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
./bootstrap-vcpkg.sh

# Add to your shell profile (~/.zshrc or ~/.bash_profile)
export VCPKG_ROOT="$HOME/vcpkg"
export PATH="$VCPKG_ROOT:$PATH"

# Reload your shell
source ~/.zshrc  # or source ~/.bash_profile
```

### 3. Install Dependencies

For **Apple Silicon (M1/M2/M3)**:
```bash
vcpkg install openvdb[nanovdb]:arm64-osx
vcpkg install yaml-cpp:arm64-osx
vcpkg install tbb:arm64-osx
vcpkg install boost:arm64-osx
vcpkg install blosc:arm64-osx
```

For **Intel Macs**:
```bash
vcpkg install openvdb[nanovdb]:x64-osx
vcpkg install yaml-cpp:x64-osx
vcpkg install tbb:x64-osx
vcpkg install boost:x64-osx
vcpkg install blosc:x64-osx
```

**Note**: This will take 30-60 minutes as it builds everything from source.

## Build Methods

### Option 1: Using Xcode (Recommended for Development) ✅ WORKING

```bash
# Configure
cmake --preset=macos-xcode

# Open in Xcode
open build-xcode/MonteCarloRenderer.xcodeproj

# Build and run from Xcode
# Select "mcrenderer" scheme and press Cmd+R
```

**See [XCODE_SETUP_GUIDE.md](XCODE_SETUP_GUIDE.md) for detailed Xcode instructions.**

### Option 2: Using Ninja (Faster Command-Line Builds) ✅ WORKING

```bash
# Install Ninja if not already installed
brew install ninja

# Configure
cmake --preset=macos-ninja

# Build
cmake --build build --config Debug

# Run
./build/mcrenderer
```

### Option 3: Manual CMake Configuration

```bash
# Create build directory
mkdir -p build
cd build

# Configure
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DUSE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build . --config Debug

# Run
./mcrenderer
```

## Build Configuration

### Debug vs Release

**Debug** (default):
```bash
cmake --preset=macos-xcode  # or macos-ninja
cmake --build build --config Debug
```

**Release** (optimized, faster):
```bash
cmake --preset=macos-ninja
cmake --build build --config Release
```

### Compiler Flags

The macOS build automatically applies these optimizations for CPU rendering:
- `-O3`: Maximum optimization
- `-march=native`: Use all available CPU instructions

## Performance Expectations

Since the macOS build runs on CPU (no GPU acceleration), rendering will be significantly slower than the Windows CUDA version:

| Resolution | Pixel Samples | Approximate Time |
|------------|---------------|------------------|
| 512x512    | 1             | ~2-5 seconds     |
| 1080x1080  | 1             | ~10-20 seconds   |
| 1080x1080  | 10            | ~2-3 minutes     |

**Tips for better performance:**
1. Use Release build (`-DCMAKE_BUILD_TYPE=Release`)
2. Reduce image resolution in code
3. Lower pixel samples in settings
4. Close other applications to free up CPU

## Troubleshooting

### Issue: `VCPKG_ROOT not found`

```bash
export VCPKG_ROOT="$HOME/vcpkg"
```

Add this to your `~/.zshrc` to make it permanent.

### Issue: `OpenVDB not found`

Re-install OpenVDB:
```bash
vcpkg remove openvdb
vcpkg install openvdb[nanovdb]:arm64-osx  # or x64-osx
```

### Issue: `Cannot open file: bunny_cloud.vdb`

The renderer expects a VDB file. Either:
1. Provide your own VDB file and update the path in `main.cpp`
2. Download sample VDB files from [OpenVDB website](https://www.openvdb.org/download/)

### Issue: OpenGL deprecation warnings

macOS shows warnings like "OpenGL is deprecated". This is expected. OpenGL 3.3 still works on macOS, these are just warnings about future removal.

### Issue: Build takes too long

vcpkg builds all dependencies from source. First build takes 30-60 minutes. Subsequent builds are much faster.

### Issue: Linking errors with TBB or Boost

Make sure you installed the correct architecture:
```bash
# Check your architecture
uname -m
# arm64 = Apple Silicon
# x86_64 = Intel Mac

# Install matching packages
vcpkg install openvdb[nanovdb]:arm64-osx    # For Apple Silicon
vcpkg install openvdb[nanovdb]:x64-osx      # For Intel
```

## Project Structure (macOS-Specific)

```
renderer/
├── cpu_renderer.cpp         ← CPU-only renderer (macOS)
├── buffer_interface.h       ← CPU/CUDA buffer abstraction
├── random_interface.h       ← std::mt19937 (CPU) vs curand (CUDA)
├── compute_interface.h      ← Serial loops (CPU) vs kernels (CUDA)
├── platform_macros.h        ← Macro definitions for CPU/CUDA
└── (other files shared between platforms)
```

## Running the Renderer

```bash
# From build directory
./mcrenderer

# The renderer will:
# 1. Load bunny_cloud.vdb
# 2. Create a window with OpenGL visualization
# 3. Render on CPU when you click "Render" button
# 4. Save output to raytrace_level_set-cpu.pfm
```

## Known Limitations

1. **No GPU acceleration** - CPU rendering only
2. **Slower than CUDA** - Expected 10-100x slower than GPU
3. **No real-time preview** - Rendering takes seconds to minutes
4. **OpenGL deprecated** - Still works but may be removed in future macOS

## Future Improvements

Planned enhancements for macOS:
- [ ] Metal compute shaders for GPU acceleration
- [ ] OpenMP parallelization for multi-core CPU
- [ ] Progressive rendering (show partial results)
- [ ] WebGPU support for cross-platform GPU

## Comparison: macOS vs Windows

| Feature           | macOS (CPU)     | Windows (CUDA)  |
|-------------------|-----------------|-----------------|
| GPU Acceleration  | ❌ No           | ✅ Yes          |
| Render Speed      | Slow (CPU)      | Fast (GPU)      |
| Dependencies      | OpenVDB         | OpenVDB + CUDA  |
| Multi-platform    | ✅ Yes          | ❌ Windows only |
| Development       | Xcode/Ninja     | Visual Studio   |

## Getting Help

If you encounter issues:
1. Check this troubleshooting section
2. Review `MACOS_PORT_INVESTIGATION.md` for technical details
3. Check `IMPLEMENTATION_CHECKLIST.md` for known issues
4. Verify all dependencies are installed correctly

## Contributing

To improve macOS support:
- Add OpenMP parallelization for CPU renderer
- Implement Metal compute shader path
- Optimize data structures for CPU cache
- Add progressive rendering

See `ARCHITECTURE.md` for system design details.
