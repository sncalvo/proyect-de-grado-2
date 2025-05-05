#pragma once

#include <openvdb/openvdb.h>
#include <vector>
#include <string>

namespace MCRenderer
{
  class GLRender
  {
  public:
    GLRender();
    ~GLRender();

    void render();
    void draw();
    void init();

    void setGrid(openvdb::FloatGrid::Ptr grid) {
        m_Grid = grid;
    }
    
    // Methods to set transformation matrices
    void setModelMatrix(const float* matrix);
    void setViewMatrix(const float* matrix);
    void setProjectionMatrix(const float* matrix);

  private:
    openvdb::FloatGrid::Ptr m_Grid;
    unsigned int VAO, VBO, EBO;
    unsigned int shaderProgram;

    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<unsigned int> indices;
    
    // Transformation matrices
    float modelMatrix[16];
    float viewMatrix[16];
    float projectionMatrix[16];

    void getNodeData(openvdb::FloatGrid::TreeType::NodeIter &iter);
    
    // Shader utilities
    unsigned int compileShader(const char* shaderSource, unsigned int shaderType);
    unsigned int createShaderProgram(const char* vertexShaderSource, const char* fragmentShaderSource);
    void checkShaderCompileErrors(unsigned int shader, const std::string& type);
  };
}
