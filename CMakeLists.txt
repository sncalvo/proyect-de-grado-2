set(CMAKE_BUILD_TYPE Debug)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

cmake_minimum_required(VERSION 3.5)
project(MonteCarloRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")

set(CMAKE_TOOLCHAIN_FILE "C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake")

# Configure GLFW as static library
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)    # Optional: disable docs
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)   # Optional: disable tests
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # Optional: disable examples

# Add GLFW from local source
add_subdirectory(common/3rdParty/glfw3)

if (MSVC)
  add_compile_options(/bigobj)
else ()
  add_compile_options(-Wa,-mbig-obj)
endif ()

include_directories(${gdt_dir})
add_subdirectory(${gdt_dir})

include_directories(common)

add_subdirectory(renderer)

add_subdirectory(common/glad)
