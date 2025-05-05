#include "GLRender.h"

#include <stdexcept>
#include <iostream>
#include <cstring> // For memcpy

#include <glad/glad.h>

namespace MCRenderer
{
  // Default vertex shader source
  const char* defaultVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec3 FragPos;
    out vec3 Normal;
    
    void main()
    {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = aNormal;
        
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
  )";

  // Default fragment shader source
  const char* defaultFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    in vec3 FragPos;
    in vec3 Normal;
    
    uniform vec3 lightPos;
    uniform vec3 objectColor;
    uniform vec3 lightColor;
    
    void main()
    {
        // Ambient light
        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
        
        // Diffuse light
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0) - FragPos);
        float diff = max(dot(Normal, lightDir), 0.0);
        vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
        
        // Final color
        vec3 baseColor = vec3(0.7, 0.8, 0.9);
        vec3 result = (ambient + diffuse) * baseColor;
        FragColor = vec4(result, 1.0);
    }
  )";

  GLRender::GLRender() : VAO(0), VBO(0), EBO(0), shaderProgram(0)
  {
    // Initialize matrices to identity
    float identity[16] = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f
    };
    
    memcpy(modelMatrix, identity, sizeof(float) * 16);
    memcpy(viewMatrix, identity, sizeof(float) * 16);
    memcpy(projectionMatrix, identity, sizeof(float) * 16);
  }

  GLRender::~GLRender()
  {
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (EBO) glDeleteBuffers(1, &EBO);
    if (shaderProgram) glDeleteProgram(shaderProgram);
  }

  void GLRender::setModelMatrix(const float* matrix)
  {
    if (matrix) {
      memcpy(modelMatrix, matrix, sizeof(float) * 16);
    }
  }

  void GLRender::setViewMatrix(const float* matrix)
  {
    if (matrix) {
      memcpy(viewMatrix, matrix, sizeof(float) * 16);
    }
  }

  void GLRender::setProjectionMatrix(const float* matrix)
  {
    if (matrix) {
      memcpy(projectionMatrix, matrix, sizeof(float) * 16);
    }
  }

  void GLRender::render()
  {
    // Use the shader program
    glUseProgram(shaderProgram);
    
    // Set transformation matrices
    int modelLoc = glGetUniformLocation(shaderProgram, "model");
    int viewLoc = glGetUniformLocation(shaderProgram, "view");
    int projLoc = glGetUniformLocation(shaderProgram, "projection");
    
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, modelMatrix);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projectionMatrix);
    
    // Draw the geometry
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }

  void GLRender::draw()
  {
    // Use the shader program
    glUseProgram(shaderProgram);
    
    // Set transformation matrices
    int modelLoc = glGetUniformLocation(shaderProgram, "model");
    int viewLoc = glGetUniformLocation(shaderProgram, "view");
    int projLoc = glGetUniformLocation(shaderProgram, "projection");
    
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, modelMatrix);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projectionMatrix);
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    
    // Draw the geometry
    glBindVertexArray(VAO);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }

  void GLRender::init()
  {
    if (!m_Grid)
    {
      throw std::runtime_error("Grid is not a float grid");
    }

    auto gridVoxelSize = m_Grid->voxelSize();
    vertices.clear();
    normals.clear();
    indices.clear();

    auto currentIndex = 0;
    auto bbox = m_Grid->evalActiveVoxelBoundingBox();

    // fill vertices and indices based on bounding box
    // fill vertices and indices based on bounding box
    auto minCoord = bbox.min();
    auto maxCoord = bbox.max();

    // Extract corner coordinates in voxel space
    float xMin = static_cast<float>(minCoord.x());
    float yMin = static_cast<float>(minCoord.y());
    float zMin = static_cast<float>(minCoord.z());
    float xMax = static_cast<float>(maxCoord.x());
    float yMax = static_cast<float>(maxCoord.y());
    float zMax = static_cast<float>(maxCoord.z());

    // Bottom face (negative y)
    // Vertices
    vertices.push_back(xMin); vertices.push_back(yMin); vertices.push_back(zMin); // 0
    vertices.push_back(xMax); vertices.push_back(yMin); vertices.push_back(zMin); // 1
    vertices.push_back(xMax); vertices.push_back(yMin); vertices.push_back(zMax); // 2
    vertices.push_back(xMin); vertices.push_back(yMin); vertices.push_back(zMax); // 3

    // Normals (same for all vertices in the face)
    for (int i = 0; i < 4; i++) {
        normals.push_back(0.0f); normals.push_back(-1.0f); normals.push_back(0.0f);
    }

    // Bottom face triangles
    indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
    indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);

    currentIndex += 4; // Bottom face has 4 vertices

    // Top face (positive y)
    // Vertices
    vertices.push_back(xMin); vertices.push_back(yMax); vertices.push_back(zMin); // 4
    vertices.push_back(xMax); vertices.push_back(yMax); vertices.push_back(zMin); // 5
    vertices.push_back(xMax); vertices.push_back(yMax); vertices.push_back(zMax); // 6
    vertices.push_back(xMin); vertices.push_back(yMax); vertices.push_back(zMax); // 7

    // Normals
    for (int i = 0; i < 4; i++) {
        normals.push_back(0.0f); normals.push_back(1.0f); normals.push_back(0.0f);
    }

    // Top face triangles
    indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
    indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);

    currentIndex += 4; // Top face has 4 vertices

    // Front face (negative z)
    // Vertices
    vertices.push_back(xMin); vertices.push_back(yMin); vertices.push_back(zMin); // 8
    vertices.push_back(xMax); vertices.push_back(yMin); vertices.push_back(zMin); // 9
    vertices.push_back(xMax); vertices.push_back(yMax); vertices.push_back(zMin); // 10
    vertices.push_back(xMin); vertices.push_back(yMax); vertices.push_back(zMin); // 11

    // Normals
    for (int i = 0; i < 4; i++) {
        normals.push_back(0.0f); normals.push_back(0.0f); normals.push_back(-1.0f);
    }

    // Front face triangles
    indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
    indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);

    currentIndex += 4; // Front face has 4 vertices

    // Back face (positive z)
    // Vertices
    vertices.push_back(xMin); vertices.push_back(yMin); vertices.push_back(zMax); // 12
    vertices.push_back(xMax); vertices.push_back(yMin); vertices.push_back(zMax); // 13
    vertices.push_back(xMax); vertices.push_back(yMax); vertices.push_back(zMax); // 14
    vertices.push_back(xMin); vertices.push_back(yMax); vertices.push_back(zMax); // 15

    // Normals
    for (int i = 0; i < 4; i++) {
        normals.push_back(0.0f); normals.push_back(0.0f); normals.push_back(1.0f);
    }

    // Back face triangles
    indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
    indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);

    currentIndex += 4; // Back face has 4 vertices

    // Left face (negative x)
    // Vertices
    vertices.push_back(xMin); vertices.push_back(yMin); vertices.push_back(zMin); // 16
    vertices.push_back(xMin); vertices.push_back(yMin); vertices.push_back(zMax); // 17
    vertices.push_back(xMin); vertices.push_back(yMax); vertices.push_back(zMax); // 18
    vertices.push_back(xMin); vertices.push_back(yMax); vertices.push_back(zMin); // 19

    // Normals
    for (int i = 0; i < 4; i++) {
        normals.push_back(-1.0f); normals.push_back(0.0f); normals.push_back(0.0f);
    }

    // Left face triangles
    indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
    indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);

    currentIndex += 4; // Left face has 4 vertices

    // Right face (positive x)
    // Vertices
    vertices.push_back(xMax); vertices.push_back(yMin); vertices.push_back(zMin); // 20
    vertices.push_back(xMax); vertices.push_back(yMin); vertices.push_back(zMax); // 21
    vertices.push_back(xMax); vertices.push_back(yMax); vertices.push_back(zMax); // 22
    vertices.push_back(xMax); vertices.push_back(yMax); vertices.push_back(zMin); // 23

    // Normals
    for (int i = 0; i < 4; i++) {
        normals.push_back(1.0f); normals.push_back(0.0f); normals.push_back(0.0f);
    }

    // Right face triangles
    indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
    indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);

    currentIndex += 4; // Right face has 4 vertices
    
    /*
    for (openvdb::FloatGrid::ValueOnCIter iter = m_Grid->cbeginValueOn(); iter; ++iter)
    {
      openvdb::FloatGrid::TreeType::RootNodeType::ChildNodeType::ChildNodeType::ChildNodeType *node = nullptr;
      auto worldPosition = iter.getCoord().asVec3d();
      iter.getNode(node);

      if (node == nullptr)
      {
          continue;
      }

      float size = gridVoxelSize[0];
      float x = worldPosition[0];
      float y = worldPosition[1];
      float z = worldPosition[2];

      // Define vertices and normals for a cube
      
      // Bottom face (negative y)
      // Vertices
      vertices.push_back(x - size/2); vertices.push_back(y - size/2); vertices.push_back(z - size/2); // 0
      vertices.push_back(x + size/2); vertices.push_back(y - size/2); vertices.push_back(z - size/2); // 1
      vertices.push_back(x + size/2); vertices.push_back(y - size/2); vertices.push_back(z + size/2); // 2
      vertices.push_back(x - size/2); vertices.push_back(y - size/2); vertices.push_back(z + size/2); // 3
      
      // Normals (same for all vertices in the face)
      for (int i = 0; i < 4; i++) {
        normals.push_back(0.0f); normals.push_back(-1.0f); normals.push_back(0.0f);
      }
      
      // Bottom face triangles
      indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
      indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);
      
      currentIndex += 4; // Bottom face has 4 vertices
      
      // Top face (positive y)
      // Vertices
      vertices.push_back(x - size/2); vertices.push_back(y + size/2); vertices.push_back(z - size/2); // 4
      vertices.push_back(x + size/2); vertices.push_back(y + size/2); vertices.push_back(z - size/2); // 5
      vertices.push_back(x + size/2); vertices.push_back(y + size/2); vertices.push_back(z + size/2); // 6
      vertices.push_back(x - size/2); vertices.push_back(y + size/2); vertices.push_back(z + size/2); // 7
      
      // Normals
      for (int i = 0; i < 4; i++) {
        normals.push_back(0.0f); normals.push_back(1.0f); normals.push_back(0.0f);
      }
      
      // Top face triangles
      indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
      indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);
      
      currentIndex += 4; // Top face has 4 vertices
      
      // Front face (negative z)
      // Vertices
      vertices.push_back(x - size/2); vertices.push_back(y - size/2); vertices.push_back(z - size/2); // 8
      vertices.push_back(x + size/2); vertices.push_back(y - size/2); vertices.push_back(z - size/2); // 9
      vertices.push_back(x + size/2); vertices.push_back(y + size/2); vertices.push_back(z - size/2); // 10
      vertices.push_back(x - size/2); vertices.push_back(y + size/2); vertices.push_back(z - size/2); // 11
      
      // Normals
      for (int i = 0; i < 4; i++) {
        normals.push_back(0.0f); normals.push_back(0.0f); normals.push_back(-1.0f);
      }
      
      // Front face triangles
      indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
      indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);
      
      currentIndex += 4; // Front face has 4 vertices
      
      // Back face (positive z)
      // Vertices
      vertices.push_back(x - size/2); vertices.push_back(y - size/2); vertices.push_back(z + size/2); // 12
      vertices.push_back(x + size/2); vertices.push_back(y - size/2); vertices.push_back(z + size/2); // 13
      vertices.push_back(x + size/2); vertices.push_back(y + size/2); vertices.push_back(z + size/2); // 14
      vertices.push_back(x - size/2); vertices.push_back(y + size/2); vertices.push_back(z + size/2); // 15
      
      // Normals
      for (int i = 0; i < 4; i++) {
        normals.push_back(0.0f); normals.push_back(0.0f); normals.push_back(1.0f);
      }
      
      // Back face triangles
      indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
      indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);
      
      currentIndex += 4; // Back face has 4 vertices
      
      // Left face (negative x)
      // Vertices
      vertices.push_back(x - size/2); vertices.push_back(y - size/2); vertices.push_back(z - size/2); // 16
      vertices.push_back(x - size/2); vertices.push_back(y - size/2); vertices.push_back(z + size/2); // 17
      vertices.push_back(x - size/2); vertices.push_back(y + size/2); vertices.push_back(z + size/2); // 18
      vertices.push_back(x - size/2); vertices.push_back(y + size/2); vertices.push_back(z - size/2); // 19
      
      // Normals
      for (int i = 0; i < 4; i++) {
        normals.push_back(-1.0f); normals.push_back(0.0f); normals.push_back(0.0f);
      }
      
      // Left face triangles
      indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
      indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);
      
      currentIndex += 4; // Left face has 4 vertices
      
      // Right face (positive x)
      // Vertices
      vertices.push_back(x + size/2); vertices.push_back(y - size/2); vertices.push_back(z - size/2); // 20
      vertices.push_back(x + size/2); vertices.push_back(y - size/2); vertices.push_back(z + size/2); // 21
      vertices.push_back(x + size/2); vertices.push_back(y + size/2); vertices.push_back(z + size/2); // 22
      vertices.push_back(x + size/2); vertices.push_back(y + size/2); vertices.push_back(z - size/2); // 23
      
      // Normals
      for (int i = 0; i < 4; i++) {
        normals.push_back(1.0f); normals.push_back(0.0f); normals.push_back(0.0f);
      }
      
      // Right face triangles
      indices.push_back(currentIndex + 0); indices.push_back(currentIndex + 1); indices.push_back(currentIndex + 2);
      indices.push_back(currentIndex + 2); indices.push_back(currentIndex + 3); indices.push_back(currentIndex + 0);
      
      currentIndex += 4; // Right face has 4 vertices
    }*/

    std::cout << "Added Vertices" << std::endl;

    // Create shader program
    shaderProgram = createShaderProgram(defaultVertexShaderSource, defaultFragmentShaderSource);

    // OpenGL initialization
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    // check for errors
    if (glGetError() != 0)
    {
        std::cerr << "Error generating vertex array" << std::endl;
    }

    // Create and bind VBO for vertices
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);
    
    // Create and bind VBO for normals
    unsigned int normalVBO;
    glGenBuffers(1, &normalVBO);
    glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
    
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);

    // Element buffer object
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);

    vertices.clear();
  }

  void GLRender::getNodeData(openvdb::FloatGrid::TreeType::NodeIter &iter)
  {
    openvdb::FloatGrid::TreeType::RootNodeType::ChildNodeType::ChildNodeType::ChildNodeType *node = nullptr;
    iter.getNode(node);
    auto origin = node->origin();
    auto dim = node->dim();

    printf("Origin: %f, %f, %f\n", origin[0], origin[1], origin[2]);
  }

  unsigned int GLRender::compileShader(const char* shaderSource, unsigned int shaderType)
  {
    unsigned int shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSource, NULL);
    glCompileShader(shader);
    
    // Check for shader compile errors
    checkShaderCompileErrors(shader, shaderType == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT");
    
    return shader;
  }

  unsigned int GLRender::createShaderProgram(const char* vertexShaderSource, const char* fragmentShaderSource)
  {
    // Compile shaders
    unsigned int vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
    unsigned int fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
    
    // Link shaders
    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    // Check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    
    // Delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
  }

  void GLRender::checkShaderCompileErrors(unsigned int shader, const std::string& type)
  {
    int success;
    char infoLog[1024];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    
    if (!success) {
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
    }
  }
}