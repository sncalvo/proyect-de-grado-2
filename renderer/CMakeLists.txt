find_package(OpenGL REQUIRED)

# Common sources for both CUDA and CPU builds
set(COMMON_SOURCES
  main.cpp
  Window.h
  Camera.h
  image.h
  glfWindow/GLFWindow.h
  glfWindow/GLFWindow.cpp
  imgui/imgui.h
  imgui/imgui.cpp
  imgui/imconfig.h
  imgui/imgui_internal.h
  imgui/imgui_widgets.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_impl_glfw.cpp
  imgui/imgui_impl_glfw.h
  imgui/imstb_rectpack.h
  imgui/imstb_textedit.h
  imgui/imstb_truetype.h
  imgui/imgui_impl_opengl3.h
  imgui/imgui_impl_opengl3.cpp
  imgui/imgui_impl_opengl3_loader.h
  GLRender.h
  GLRender.cpp
  settings.h
)

# Conditional sources based on CUDA availability
if(USE_CUDA)
  set(RENDERER_SOURCES
    ${COMMON_SOURCES}
    nanovdb.cu
    ComputePrimitives.cuh
    common.cuh
    integrator.cuh
    ray.cuh
  )
else()
  set(RENDERER_SOURCES
    ${COMMON_SOURCES}
    cpu_renderer.cpp
    buffer_interface.h
    random_interface.h
    compute_interface.h
    platform_macros.h
  )
endif()

add_executable(mcrenderer ${RENDERER_SOURCES})

# Try to find OpenVDB
# On Homebrew (macOS), add the cmake module path
if(APPLE AND EXISTS "/opt/homebrew/Cellar/openvdb")
  file(GLOB OPENVDB_CMAKE_DIR "/opt/homebrew/Cellar/openvdb/*/lib/cmake/OpenVDB")
  list(APPEND CMAKE_MODULE_PATH ${OPENVDB_CMAKE_DIR})
endif()

find_package(OpenVDB REQUIRED)

# Always use OpenVDB
target_compile_definitions(mcrenderer PRIVATE "NANOVDB_USE_OPENVDB")

# Conditional CUDA configuration
if(USE_CUDA)
  target_compile_definitions(mcrenderer PRIVATE "NANOVDB_USE_CUDA")
  target_compile_definitions(mcrenderer PRIVATE "USE_CUDA")
  
  set_target_properties(mcrenderer PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
  )
  
  set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
  if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
    set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
  endif()
  
  set(CMAKE_CUDA_FLAGS "${NANOVDB_CUDA_EXTENDED_LAMBDA} -G -use_fast_math ${CMAKE_CUDA_FLAGS}")
else()
  target_compile_definitions(mcrenderer PRIVATE "USE_CPU")
  
  # Enable compiler optimizations for CPU build
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(mcrenderer PRIVATE -O3 -march=native)
  endif()
endif()

find_package(yaml-cpp CONFIG REQUIRED)

# Link OpenVDB and set include directories
if(TARGET OpenVDB::openvdb)
  target_link_libraries(mcrenderer OpenVDB::openvdb)
elseif(OPENVDB_FOUND)
  target_include_directories(mcrenderer PRIVATE ${OpenVDB_INCLUDE_DIRS})
  target_link_libraries(mcrenderer ${OpenVDB_LIBRARIES})
endif()

target_link_libraries(mcrenderer
  ${OPENGL_gl_LIBRARY}
  yaml-cpp::yaml-cpp
  gdt
  glfw
  glad
)