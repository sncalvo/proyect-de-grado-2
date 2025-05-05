find_package(OpenGL REQUIRED)

set_target_properties(
  PROPERTIES
  # CUDA_SEPARABLE_COMPILATION ON
  # CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

add_executable(mcrenderer
  nanovdb.cu
  ComputePrimitives.cuh
  main.cpp
  common.cuh
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
  integrator.cuh
  ray.cuh
  settings.h
)

find_package(OpenVDB CONFIG REQUIRED)

target_compile_definitions(mcrenderer PRIVATE "NANOVDB_USE_OPENVDB")
target_compile_definitions(mcrenderer PRIVATE "NANOVDB_USE_CUDA")

set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
  set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
endif()

set(CMAKE_CUDA_FLAGS "${NANOVDB_CUDA_EXTENDED_LAMBDA} -G -use_fast_math ${CMAKE_CUDA_FLAGS}")

find_package(yaml-cpp CONFIG REQUIRED)

target_link_libraries(mcrenderer
  ${OPENGL_gl_LIBRARY}
  OpenVDB::openvdb
  gdt
  glfw
  glad
)