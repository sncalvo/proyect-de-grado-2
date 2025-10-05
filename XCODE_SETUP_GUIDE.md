# Xcode Setup Guide for macOS

## ✅ Status: Working!

The Xcode preset is now fully functional and tested.

## Prerequisites

1. **Xcode** installed from App Store
2. **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```
3. **Xcode First Launch** (must be run once):
   ```bash
   sudo xcodebuild -runFirstLaunch
   ```
   OR open Xcode.app once to complete setup

4. **Dependencies** (via Homebrew):
   ```bash
   brew install openvdb yaml-cpp
   ```

## Quick Start

### Option 1: Open in Xcode (Recommended for Development)

```bash
# Configure
cmake --preset=macos-xcode

# Open in Xcode
open build-xcode/MonteCarloRenderer.xcodeproj
```

Then in Xcode:
1. Select the `mcrenderer` scheme from the dropdown
2. Press `⌘R` to build and run
3. Or `⌘B` to just build

### Option 2: Build from Command Line

```bash
# Configure
cmake --preset=macos-xcode

# Build (Debug)
cmake --build build-xcode --config Debug

# Build (Release)
cmake --build build-xcode --config Release

# Run
./build-xcode/renderer/Debug/mcrenderer
```

## Available CMake Presets

### 1. macos-xcode (Xcode IDE)
```bash
cmake --preset=macos-xcode
```
- **Generator**: Xcode
- **Output**: `build-xcode/MonteCarloRenderer.xcodeproj`
- **Best for**: Development with Xcode IDE
- **Debugging**: Full Xcode debugging support

### 2. macos-ninja (Command Line)
```bash
cmake --preset=macos-ninja
```
- **Generator**: Ninja
- **Output**: `build/mcrenderer`
- **Best for**: Fast command-line builds
- **Debugging**: Can use lldb from terminal

## Xcode Project Structure

```
build-xcode/
├── MonteCarloRenderer.xcodeproj  ← Main project (open this)
├── renderer/
│   └── Debug/
│       └── mcrenderer            ← Executable
├── common/
│   ├── gdt/
│   ├── glad/
│   └── 3rdParty/glfw3/
└── CMakeFiles/
```

## Building in Xcode

### Schemes Available:
- **ALL_BUILD** - Builds everything
- **mcrenderer** - Main executable (use this one)
- **glfw** - GLFW library
- **glad** - OpenGL loader
- **gdt** - GPU Development Tools

### Build Configurations:
- **Debug** - With debug symbols, no optimization
- **Release** - Optimized, no debug symbols
- **MinSizeRel** - Size-optimized
- **RelWithDebInfo** - Optimized with debug symbols

### How to Build:
1. Select `mcrenderer` scheme
2. Choose Debug or Release from scheme menu
3. Product → Build (`⌘B`)
4. Product → Run (`⌘R`)

## Debugging in Xcode

### Enable Breakpoints:
1. Open source files in Xcode
2. Click line number to add breakpoint
3. Run with debugger (`⌘R`)

### Useful Xcode Shortcuts:
- `⌘B` - Build
- `⌘R` - Run
- `⌘.` - Stop
- `⌘/` - Comment/Uncomment
- `⌘F` - Find in file
- `⇧⌘F` - Find in project

### View Variables:
- When paused at breakpoint, hover over variables
- Use the Variables View (bottom panel)
- LLDB console for commands

## Configuration Details

### CMakePresets.json Settings:
```json
{
    "name": "macos-xcode",
    "generator": "Xcode",
    "binaryDir": "${sourceDir}/build-xcode",
    "cacheVariables": {
        "USE_CUDA": "OFF",
        "CMAKE_OSX_DEPLOYMENT_TARGET": "11.0",
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_C_COMPILER": "/usr/bin/clang",
        "CMAKE_CXX_COMPILER": "/usr/bin/clang++",
        "CMAKE_XCODE_GENERATE_SCHEME": "ON"
    }
}
```

### Key Settings:
- **USE_CUDA**: OFF (no CUDA on macOS)
- **CMAKE_OSX_DEPLOYMENT_TARGET**: 11.0 (macOS Big Sur minimum)
- **Compilers**: Apple Clang from Xcode
- **Schemes**: Auto-generated for each target

## Comparison: Xcode vs Ninja

| Feature | Xcode | Ninja |
|---------|-------|-------|
| Build Speed | Moderate | Fast |
| IDE Integration | ✅ Full | ❌ None |
| Debugging | ✅ GUI | Terminal only |
| Code Navigation | ✅ Excellent | ❌ None |
| Build Output | `build-xcode/` | `build/` |
| Best For | Development | CI/CD |

## Troubleshooting

### Issue: "xcodebuild failed to load a required plug-in"
**Solution**: Run Xcode first launch setup
```bash
sudo xcodebuild -runFirstLaunch
```
Or open Xcode.app once to complete setup.

---

### Issue: "No CMAKE_CXX_COMPILER could be found"
**Solution**: Install Xcode Command Line Tools
```bash
xcode-select --install
```

---

### Issue: "OpenVDB not found"
**Solution**: Install via Homebrew
```bash
brew install openvdb yaml-cpp
```

---

### Issue: Build warnings about library versions
**Status**: Harmless - Homebrew libraries are newer than deployment target
**Action**: Ignore or set `CMAKE_OSX_DEPLOYMENT_TARGET` to "13.0" or higher

---

### Issue: Scheme not appearing in Xcode
**Solution**: 
1. Close Xcode
2. Reconfigure: `cmake --preset=macos-xcode`
3. Reopen Xcode project

---

## Clean Build

To start fresh:

```bash
# Remove build directory
rm -rf build-xcode

# Reconfigure
cmake --preset=macos-xcode

# Rebuild
cmake --build build-xcode --config Debug
```

## Performance Tips

### For Faster Builds:
1. Use Release configuration for final builds
2. Enable parallel builds in Xcode preferences
3. Use incremental builds (don't clean every time)

### For Development:
1. Use Debug configuration
2. Enable "Build Active Architecture Only" in Xcode
3. Use Ninja for quick command-line builds

## Advanced: Custom Build Settings

### Add custom compiler flags:
Edit `renderer/CMakeLists.txt`:
```cmake
if(XCODE)
  target_compile_options(mcrenderer PRIVATE -Wall -Wextra)
endif()
```

### Change optimization level:
In Xcode: Build Settings → Optimization Level

### Enable Address Sanitizer:
Product → Scheme → Edit Scheme → Run → Diagnostics → Address Sanitizer

## Summary

✅ **Xcode preset is fully configured and working!**

**Quick Commands:**
```bash
# Setup
cmake --preset=macos-xcode

# Build
cmake --build build-xcode --config Debug

# Open in Xcode
open build-xcode/MonteCarloRenderer.xcodeproj

# Run
./build-xcode/renderer/Debug/mcrenderer
```

**Best Practices:**
- Use Xcode for development and debugging
- Use Ninja for quick command-line builds
- Keep both build directories separate (build-xcode vs build)
- Commit CMakePresets.json to git for team consistency

---

For more information, see:
- `MACOS_BUILD.md` - General macOS build guide
- `BUILD_FIXES_APPLIED.md` - Build issues resolved
- `IMPLEMENTATION_STATUS.md` - Project status
