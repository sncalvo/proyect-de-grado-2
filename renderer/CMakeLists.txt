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
  integrator.cuh
  ray.cuh
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
find_package(glfw3 CONFIG REQUIRED)

target_link_libraries(mcrenderer
  ${OPENGL_gl_LIBRARY}
  OpenVDB::openvdb
  gdt
  glfw
  imgui
)