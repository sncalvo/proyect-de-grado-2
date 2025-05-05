// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "GLFWindow.h"
#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"
#include "../settings.h"

/*! \namespace osc - Optix Siggraph Course */
namespace MCRenderer {
using namespace gdt;

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Error: %s\n", description);
}

GLFWindow::~GLFWindow() {
  glfwDestroyWindow(handle);
  glfwTerminate();
}

GLFWindow::GLFWindow(const std::string &title) {
  glfwSetErrorCallback(glfw_error_callback);
  // glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);

  if (!glfwInit())
    exit(EXIT_FAILURE);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

  handle = glfwCreateWindow(400, 400, title.c_str(), NULL, NULL);
  if (!handle) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwMakeContextCurrent(handle);

  GLFWwindow *window = glfwGetCurrentContext();
  assert(window != nullptr);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330 core");

  glfwSetWindowUserPointer(handle, this);
  glfwMakeContextCurrent(handle);
  glfwSwapInterval(1);
}

/*! callback for a window resizing event */
static void glfwindow_reshape_cb(GLFWwindow *window, int width, int height) {
  GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
  assert(gw);
  gw->resize(vec2i(width, height));
  // assert(GLFWindow::current);
  //   GLFWindow::current->resize(vec2i(width,height));
}

/*! callback for a key press */
static void glfwindow_key_cb(GLFWwindow *window, int key, int scancode,
                             int action, int mods) {
  GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
  assert(gw);
  if (action == GLFW_PRESS) {
    gw->key(key, mods);
  }
}

/*! callback for _moving_ the mouse to a new position */
static void glfwindow_mouseMotion_cb(GLFWwindow *window, double x, double y) {
  GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
  assert(gw);
  gw->mouseMotion(vec2i((int)x, (int)y));
}


/*! callback for pressing _or_ releasing a mouse button*/
static void glfwindow_mouseButton_cb(GLFWwindow *window, int button, int action,
                                     int mods) {
  GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
  assert(gw);
  // double x, y;
  // glfwGetCursorPos(window,&x,&y);
  gw->mouseButton(button, action, mods);
}

void GLFWindow::setupEvents() {
  //glfwSetFramebufferSizeCallback(handle, glfwindow_reshape_cb);
  //glfwSetMouseButtonCallback(handle, glfwindow_mouseButton_cb);
  //glfwSetKeyCallback(handle, glfwindow_key_cb);
  //glfwSetCursorPosCallback(handle, glfwindow_mouseMotion_cb);
}

void GLFWindow::run(std::function<void()>& lambda) {
  int width, height;
  glfwGetFramebufferSize(handle, &width, &height);
  resize(vec2i(width, height));

  // Create a window called "My First Tool", with a menu bar.
  bool open = true;

  float color[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
  float lightLocation[3] = { 0.f, 0.f, 0.f };
  float cameraLocation[3] = { 0.f, 0.f, 0.f };

  while (!glfwWindowShouldClose(handle)) {
    // Specify the color of the background
    glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
    // Clean the back buffer and assign the new color to it
    glClear(GL_COLOR_BUFFER_BIT);

    render();
    draw();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("My First Tool", &open, ImGuiWindowFlags_MenuBar);
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */ }
            if (ImGui::MenuItem("Save", "Ctrl+S")) { /* Do stuff */ }
            if (ImGui::MenuItem("Close", "Ctrl+W")) { open = false; }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    // Edit a color stored as 4 floats

    // Display contents in a scrolling region
    ImGui::BeginChild("settings");
    ImGui::Text("Light");
    ImGui::InputFloat3("LightLocation", Settings::getInstance().lightLocation, "%.2f");
    ImGui::ColorEdit3("Color", Settings::getInstance().lightColor);

    ImGui::Text("Camera");
    ImGui::InputFloat3("CameraLocation", Settings::getInstance().cameraLocation, "%.2f");
    ImGui::EndChild();

    if (ImGui::Button("Render")) {
      // render
        lambda();
    }
    ImGui::End(); 

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(handle);
    glfwPollEvents();
  }

  // Deletes all ImGUI instances
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

} // namespace MCRenderer
