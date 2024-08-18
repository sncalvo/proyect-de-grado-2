#pragma once

#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

#include "Camera.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <fstream>
#include <algorithm>
#include <chrono>

#include <cerrno>


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

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto &io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(handle, true);
    ImGui_ImplOpenGL3_Init("#version 130");
  }

  virtual void render() override {

  }

  virtual void draw() override {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (testWindow) {
      ImGui::Begin("Test Window", &testWindow);
      ImGui::Text("Hello World");
      // auto &settings = Settings::getInstance();
      // ImGui::Checkbox("Freeze Render", &settings.m_freezeRender);
      // ImGui::SliderFloat(
      //   "Shininess",
      //   &Settings::getInstance().m_roughness,
      //   0.0000001f,
      //   0.02f,
      //   "%.7f"
      // );
      // ImGui::Checkbox("Sample PDF", &settings.m_samplePDF);
      // ImGui::Checkbox("Sample Light", &settings.m_sampleLight);
      // ImGui::Checkbox("Color Weight", &settings.m_colorWeights);
      // ImGui::Checkbox("Debug Print", &settings.m_debugPrint);

      auto light1 = ImGui::Button("Light x0.50", ImVec2(100, 20));
      auto light2 = ImGui::Button("Light x0.75", ImVec2(100, 20));
      auto light3 = ImGui::Button("Light x1.50", ImVec2(100, 20));
      // ImGui::Text("Light Size: %f", Settings::getInstance().m_lightSize);
      auto download = ImGui::Button("Download", ImVec2(100, 20));

      ImGui::End();
    }

    ImGui::Render();

    // Print Frame
    // sample->downloadPixels(pixels.data());

    if (fbTexture == 0) {
      glGenTextures(1, &fbTexture);
    }

    glBindTexture(GL_TEXTURE_2D, fbTexture);
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                 texelType, pixels.data());

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, fbSize.x, fbSize.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
      glTexCoord2f(0.f, 0.f);
      glVertex3f(0.f, 0.f, 0.f);

      glTexCoord2f(0.f, 1.f);
      glVertex3f(0.f, (float)fbSize.y, 0.f);

      glTexCoord2f(1.f, 1.f);
      glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

      glTexCoord2f(1.f, 0.f);
      glVertex3f((float)fbSize.x, 0.f, 0.f);
    }
    glEnd();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  }

  virtual void resize(const vec2i &newSize) {
    fbSize = newSize;
    pixels.resize(newSize.x * newSize.y);
  }

  vec2i fbSize;
  GLuint fbTexture{0};
  std::vector<uint32_t> pixels;

  bool testWindow = true;
};
} // namespace MCRenderer
