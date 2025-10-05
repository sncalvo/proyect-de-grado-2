# Monte Carlo Volumetric Renderer

A cross-platform volumetric path tracer using OpenVDB/NanoVDB for rendering participating media with delta tracking and Henyey-Greenstein phase functions.

## Features

- 🎨 **Volumetric Rendering**: Path tracing through participating media
- 📊 **Delta Tracking**: Efficient null-collision algorithm
- 🌊 **Phase Functions**: Henyey-Greenstein for realistic scattering
- 📦 **VDB Support**: Load and render OpenVDB volumetric datasets
- 🖼️ **OpenGL Visualization**: Real-time bounding box preview
- 🎛️ **ImGui Interface**: Interactive controls for rendering parameters

## Platform Support

| Platform | Acceleration | Status |
|----------|-------------|--------|
| **Windows** | NVIDIA CUDA | ✅ Fully Supported |
| **macOS** | CPU | ✅ Fully Supported |
| **Linux** | NVIDIA CUDA | 🔄 Coming Soon |

## Requirements

### Windows (CUDA Build)
- Windows 10/11
- Visual Studio 2019 or newer
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0 or newer
- vcpkg package manager

### macOS (CPU Build)
- macOS 10.15 (Catalina) or newer
- Xcode Command Line Tools
- vcpkg package manager
- **Note**: Runs on CPU (no GPU acceleration)

## Quick Start

### Windows

```bash
# Install dependencies
vcpkg install openvdb[nanovdb]:x64-windows yaml-cpp:x64-windows

# Configure
cmake --preset=default

# Build
cmake --build build --config Debug

# Run
.\build\Debug\mcrenderer.exe
```

### macOS

```bash
# Install dependencies (via Homebrew - easier than vcpkg)
brew install openvdb yaml-cpp

# Option 1: Xcode (for IDE development)
cmake --preset=macos-xcode
open build-xcode/MonteCarloRenderer.xcodeproj

# Option 2: Ninja (for command-line builds)
cmake --preset=macos-ninja
cmake --build build
./build/renderer/mcrenderer
```

**For detailed macOS instructions:**
- **[XCODE_SETUP_GUIDE.md](XCODE_SETUP_GUIDE.md)** - Xcode development guide
- **[MACOS_BUILD.md](MACOS_BUILD.md)** - General macOS build guide

## Documentation

- 📘 **[MACOS_BUILD.md](MACOS_BUILD.md)** - Complete macOS build guide
- 🔬 **[MACOS_PORT_INVESTIGATION.md](MACOS_PORT_INVESTIGATION.md)** - Technical details of the port
- ✅ **[IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)** - Development checklist
- 🏗️ **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design

## Building from Source

### Dependencies

All platforms require:
- CMake 3.5 or newer
- OpenVDB with NanoVDB support
- yaml-cpp
- GLFW3 (included as submodule)
- OpenGL

Windows additionally requires:
- CUDA Toolkit 11.0+

### Configuration Options

```bash
# Enable/disable CUDA (auto-detected on macOS)
cmake -DUSE_CUDA=ON ..   # Windows default
cmake -DUSE_CUDA=OFF ..  # macOS default

# Set vcpkg toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake ..
```

## Usage

1. **Load VDB File**: The renderer expects `bunny_cloud.vdb` by default
2. **Adjust Settings**: Modify `config.yaml` for render parameters
3. **Render**: Click the render button or call `runNano()`
4. **Output**: Rendered images saved as PFM files

### Key Settings

```yaml
# config.yaml
cameraLocation: [0, 0, 100]
lightLocation: [50, 100, 50]
pixelSamples: 10
```

## Project Structure

```
renderer/
├── main.cpp              # Entry point
├── cpu_renderer.cpp      # CPU rendering (macOS)
├── nanovdb.cu            # CUDA rendering (Windows)
├── integrator.cuh        # Path tracing algorithms
├── GLRender.cpp          # OpenGL visualization
├── buffer_interface.h    # Platform abstraction
├── random_interface.h    # RNG abstraction
├── compute_interface.h   # Compute abstraction
└── imgui/                # UI framework
```

## Performance

| Platform | Resolution | Time (1 sample) |
|----------|-----------|-----------------|
| Windows RTX 3080 | 1080x1080 | ~100ms |
| macOS M1 Pro (CPU) | 1080x1080 | ~15 seconds |
| macOS M1 Pro (CPU) | 512x512 | ~3 seconds |

**macOS Performance Tips**:
- Use Release build for 2-3x speedup
- Reduce resolution for faster preview
- Lower pixel samples in config

## Algorithm Details

### Delta Tracking
Efficient null-collision path tracing through heterogeneous media:
1. Sample free-path lengths using majorant extinction
2. Classify collisions as real or null
3. Apply absorption, scattering, or continue

### Henyey-Greenstein Phase Function
Realistic anisotropic scattering with adjustable `g` parameter:
- `g = 0`: Isotropic scattering
- `g > 0`: Forward scattering
- `g < 0`: Backward scattering

## Development

### Adding Features

1. **New Render Modes**: Modify `integrator.cuh`
2. **UI Controls**: Edit ImGui code in `Window.h`
3. **File Formats**: Extend loaders in `main.cpp`

### Platform-Specific Code

Use conditional compilation:
```cpp
#ifdef USE_CUDA
    // CUDA-specific code
#else
    // CPU fallback
#endif
```

## Troubleshooting

### Windows
- **CUDA not found**: Install CUDA Toolkit and update PATH
- **vcpkg errors**: Set `VCPKG_ROOT` environment variable

### macOS
- **OpenGL warnings**: Expected, OpenGL still works
- **Slow rendering**: Normal for CPU, try lower resolution
- **vcpkg errors**: Ensure correct architecture (arm64 vs x64)

See [MACOS_BUILD.md](MACOS_BUILD.md) for detailed troubleshooting.

## Contributing

Contributions welcome! Areas for improvement:
- [ ] Metal compute shaders for macOS GPU
- [ ] OpenMP CPU parallelization
- [ ] Progressive rendering
- [ ] Additional file formats
- [ ] Denoising integration

## License

[Your License Here]

## Acknowledgments

- OpenVDB by DreamWorks Animation
- NanoVDB by NVIDIA
- ImGui by Omar Cornut
- GLFW for windowing

## Contact

[Your Contact Information]

---

**Note**: This is a research/educational project. For production rendering, consider using established renderers like Cycles, Arnold, or Mantra.
