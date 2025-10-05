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
#include "image.h"

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

  ~SampleWindow() {
    // Clean up texture display resources
    if (quadVAO) glDeleteVertexArrays(1, &quadVAO);
    if (quadVBO) glDeleteBuffers(1, &quadVBO);
    if (textureShader) glDeleteProgram(textureShader);
    if (fbTexture) glDeleteTextures(1, &fbTexture);
  }

  virtual void render() override {
    // Update camera transformation matrices
    updateCamera();
    
    // If progressive rendering is active, render one sample per frame
    if (isRendering && renderOneSampleCallback) {
      renderOneSampleCallback();
    }
    
    // Call the renderer's render method
    if (renderer) {
      renderer->render();
    }
  }

  virtual void draw() override {
    glViewport(0, 0, fbSize.x, fbSize.y);
    
    // If we have pixel data, draw it as a texture
    if (!pixels.empty()) {
      // Initialize texture display resources on first use
      if (!textureShader) {
        initTextureDisplay();
      }

      // Upload texture data
      if (fbTexture == 0) {
        glGenTextures(1, &fbTexture);
      }

      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fbSize.x, fbSize.y, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, pixels.data());
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      // Draw the texture as a fullscreen quad
      glDisable(GL_DEPTH_TEST);
      glUseProgram(textureShader);
      
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glUniform1i(glGetUniformLocation(textureShader, "textureSampler"), 0);
      
      glBindVertexArray(quadVAO);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);
      
      glUseProgram(0);
      glBindTexture(GL_TEXTURE_2D, 0);
    } else {
      // If no image yet, show the 3D wireframe
      glEnable(GL_DEPTH_TEST);
      if (renderer) {
        renderer->draw();
      }
    }
  }

  virtual void resize(const vec2i &newSize) {
    fbSize = newSize;
    pixels.resize(newSize.x * newSize.y);
  }

  vec2i fbSize;
  GLuint fbTexture{0};
  std::vector<uint32_t> pixels;

  bool testWindow = true;
  
  // Progressive rendering state
  bool isRendering = false;
  std::function<void()> renderOneSampleCallback;
  Image* currentImage = nullptr;

  void setRenderer(GLRender *renderer) {
    this->renderer = renderer;
  }
  
  void startProgressiveRender(std::function<void()> oneSampleCallback, Image* img) {
    renderOneSampleCallback = oneSampleCallback;
    currentImage = img;
    isRendering = true;
  }
  
  void stopProgressiveRender() {
    isRendering = false;
    currentImage = nullptr;
  }
  
  bool isProgressiveRendering() const {
    return isRendering;
  }
  
  int getCurrentSample() const {
    return currentImage ? currentImage->getCurrentSample() : 0;
  }
  
  // Override base class methods for progress display
  virtual bool getIsRendering() const override {
    return isProgressiveRendering();
  }
  
  virtual int getCurrentRenderSample() const override {
    return getCurrentSample();
  }

  void loadImageToPixels(Image* image) {
    if (!image) return;
    
    // Get the image data (download from device if necessary)
    float* imageData = image->deviceDownload();
    if (!imageData) return;
    
    int width = image->width();
    int height = image->height();
    
    // Resize pixel buffer if needed
    if (fbSize.x != width || fbSize.y != height) {
      fbSize.x = width;
      fbSize.y = height;
      pixels.resize(width * height);
    }
    
    // Convert float RGB to uint32_t RGBA
    // Image is stored as RGB floats (0-1 range), we need RGBA bytes
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        int imgIdx = idx * 3;
        
        // Clamp and convert float [0,1] to byte [0,255]
        uint8_t r = static_cast<uint8_t>(std::min(1.0f, std::max(0.0f, imageData[imgIdx + 0])) * 255.0f);
        uint8_t g = static_cast<uint8_t>(std::min(1.0f, std::max(0.0f, imageData[imgIdx + 1])) * 255.0f);
        uint8_t b = static_cast<uint8_t>(std::min(1.0f, std::max(0.0f, imageData[imgIdx + 2])) * 255.0f);
        uint8_t a = 255; // Full opacity
        
        // Pack RGBA into uint32_t (ABGR format for OpenGL)
        pixels[idx] = (a << 24) | (b << 16) | (g << 8) | r;
      }
    }
  }

private:
  GLRender *renderer;
  const Camera &camera;
  
  // Resources for displaying the rendered image texture
  GLuint quadVAO = 0;
  GLuint quadVBO = 0;
  GLuint textureShader = 0;

  void initTextureDisplay() {
    // Simple vertex shader for fullscreen quad
    const char* vertexShaderSource = R"(
      #version 330 core
      layout (location = 0) in vec2 aPos;
      layout (location = 1) in vec2 aTexCoord;
      
      out vec2 TexCoord;
      
      void main()
      {
          gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
          TexCoord = aTexCoord;
      }
    )";

    // Simple fragment shader for textured quad
    const char* fragmentShaderSource = R"(
      #version 330 core
      out vec4 FragColor;
      
      in vec2 TexCoord;
      
      uniform sampler2D textureSampler;
      
      void main()
      {
          FragColor = texture(textureSampler, TexCoord);
      }
    )";

    // Compile shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Create shader program
    textureShader = glCreateProgram();
    glAttachShader(textureShader, vertexShader);
    glAttachShader(textureShader, fragmentShader);
    glLinkProgram(textureShader);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Create fullscreen quad (two triangles)
    float quadVertices[] = {
      // positions   // texCoords
      -1.0f,  1.0f,  0.0f, 1.0f,
      -1.0f, -1.0f,  0.0f, 0.0f,
       1.0f, -1.0f,  1.0f, 0.0f,

      -1.0f,  1.0f,  0.0f, 1.0f,
       1.0f, -1.0f,  1.0f, 0.0f,
       1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    
    // Position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    
    // Texture coord attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    
    glBindVertexArray(0);
  }

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
