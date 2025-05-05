#ifndef WINDOW_H
#define WINDOW_H

#include "glfWindow/GLFWindow.h"

#include "Camera.h"

#include <fstream>
#include <algorithm>
#include <chrono>

#include <cerrno>

#include <glad/glad.h>

#include "GLRender.h"

#if defined(_WIN32)
#include <windows.h>

std::string getExePath() {
  char buffer[MAX_PATH];
  GetModuleFileName(NULL, buffer, MAX_PATH);
  std::string::size_type pos = std::string(buffer).find_last_of("\\/");
  return std::string(buffer).substr(0, pos);
}

#else
#include <unistd.h>

std::string getExePath() {
  char buffer[1024];
  ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (len != -1) {
    buffer[len] = '\0';
    std::string::size_type pos = std::string(buffer).find_last_of("\\/");
    return std::string(buffer).substr(0, pos);
  }
  return "";
}

#endif



namespace MCRenderer {
class SampleWindow : public GLFCameraWindow {
public:
  SampleWindow(const std::string &title,
               const Camera &camera,
               const float worldScale)
      : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
        camera(camera)
         {

    setupEvents();
  }

  virtual void render() override {
    // Update camera transformation matrices
    updateCamera();
    
    // Call the renderer's render method
    if (renderer) {
      renderer->render();
    }
  }

  virtual void draw() override {
    if (fbTexture == 0) {
      glGenTextures(1, &fbTexture);
    }

    glBindTexture(GL_TEXTURE_2D, fbTexture);
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                 texelType, pixels.data());

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_DEPTH_TEST);

    glViewport(0, 0, fbSize.x, fbSize.y);

    renderer->draw();
  }

  virtual void resize(const vec2i &newSize) {
    fbSize = newSize;
    pixels.resize(newSize.x * newSize.y);
  }

  vec2i fbSize;
  GLuint fbTexture{0};
  std::vector<uint32_t> pixels;

  bool testWindow = true;

  void setRenderer(GLRender *renderer) {
    this->renderer = renderer;
  }

private:
  GLRender *renderer;
  const Camera &camera;

  void updateCamera() {
    if (!renderer) return;
    
    // Model matrix (identity)
    float modelMatrix[16] = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f
    };
    
    // View matrix based on camera position and orientation
    vec3f pos = camera.from;
    vec3f at = camera.at;
    vec3f up = camera.up;
    
    // Calculate look direction (z-axis)
    vec3f z = normalize(pos - at);
    vec3f x = normalize(cross(up, z));
    vec3f y = cross(z, x);
    
    float viewMatrix[16] = {
      x.x, y.x, z.x, 0.0f,
      x.y, y.y, z.y, 0.0f,
      x.z, y.z, z.z, 0.0f,
      -dot(x, pos), -dot(y, pos), -dot(z, pos), 1.0f
    };
    
    // Projection matrix
    float fov = 45.0f;
    float aspect = (float)fbSize.x / (float)fbSize.y;
    float zNear = 0.1f;
    float zFar = 100.0f;
    
    float tanHalfFov = tanf((fov / 2.0f) * (3.14159f / 180.0f));
    float f = 1.0f / tanHalfFov;
    
    float projMatrix[16] = {
      f / aspect, 0.0f, 0.0f, 0.0f,
      0.0f, f, 0.0f, 0.0f,
      0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), -1.0f,
      0.0f, 0.0f, (2.0f * zFar * zNear) / (zNear - zFar), 0.0f
    };
    
    // Set the matrices in the renderer
    renderer->setModelMatrix(modelMatrix);
    renderer->setViewMatrix(viewMatrix);
    renderer->setProjectionMatrix(projMatrix);
  }
};
} // namespace MCRenderer
#endif
