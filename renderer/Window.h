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
      : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale)
         {

    setupEvents();
  }

  virtual void render() override {

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
};
} // namespace MCRenderer
#endif
