set(CMAKE_BUILD_TYPE Debug)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

cmake_minimum_required(VERSION 3.10)

# Platform detection and CUDA configuration
option(USE_CUDA "Build with CUDA support" ON)

# Disable CUDA on macOS (not supported)
if(APPLE)
    set(USE_CUDA OFF CACHE BOOL "CUDA not supported on macOS" FORCE)
    message(STATUS "macOS detected - building without CUDA (CPU mode)")
endif()

# Configure project languages based on CUDA availability
if(USE_CUDA)
    project(MonteCarloRenderer LANGUAGES CXX CUDA)
    message(STATUS "Building with CUDA support")
else()
    project(MonteCarloRenderer LANGUAGES CXX)
    message(STATUS "Building without CUDA (CPU mode)")
endif()

set(CMAKE_CXX_STANDARD 17)

# CUDA-specific configuration
if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")

# Configure vcpkg toolchain - use environment variable or platform-specific default
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
elseif(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    message(STATUS "Using vcpkg from VCPKG_ROOT: $ENV{VCPKG_ROOT}")
elseif(WIN32 AND EXISTS "C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake")
    # Fallback to Windows default path if it exists
    set(CMAKE_TOOLCHAIN_FILE "C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    message(STATUS "Using default Windows vcpkg path")
else()
    message(WARNING "vcpkg not found. Dependencies must be installed manually or CMAKE_TOOLCHAIN_FILE must be set.")
endif()

# Configure GLFW as static library
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)    # Optional: disable docs
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)   # Optional: disable tests
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # Optional: disable examples

# Add GLFW from local source
add_subdirectory(common/3rdParty/glfw3)

if (MSVC)
  add_compile_options(/bigobj)
elseif(MINGW)
  add_compile_options(-Wa,-mbig-obj)
endif ()

include_directories(${gdt_dir})
add_subdirectory(${gdt_dir})

include_directories(common)

add_subdirectory(renderer)

add_subdirectory(common/glad)
