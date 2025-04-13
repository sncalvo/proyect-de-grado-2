#include "GLRender.h"

#include <stdexcept>
#include <iostream>

#include <glad/glad.h>

namespace MCRenderer
{
  GLRender::GLRender()
  {
  }

  GLRender::~GLRender()
  {
  }

  void GLRender::render()
  {
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }

  void GLRender::draw()
  {
  }

  void GLRender::init()
  {
    if (!m_Grid)
    {
      throw std::runtime_error("Grid is not a float grid");
    }

    auto gridVoxelSize = m_Grid->voxelSize();
    vertices.clear();
    indices.clear();

    auto currentIndex = 0;

    // for (openvdb::FloatGrid::ValueOnCIter iter = m_Grid->cbeginValueOn(); iter; ++iter)
    // {
    //   openvdb::FloatGrid::TreeType::RootNodeType::ChildNodeType::ChildNodeType::ChildNodeType *node = nullptr;
    //   auto worldPosition = iter.getCoord().asVec3d();

    //   // std::cout << "Grid " << iter.getCoord() << " = " << *iter << " " << iter.getBoundingBox() << ", depth: " << iter.getDepth() << ", origin: " << worldPosition << ", size: " << gridVoxelSize << std::endl;

    //   float size = gridVoxelSize[0];
    //   float x = worldPosition[0];
    //   float y = worldPosition[1];
    //   float z = worldPosition[2];

    //   // Define the 8 vertices of a cube
    //   // Vertex 0: bottom-left-back
    //   vertices.push_back(x - size / 2);
    //   vertices.push_back(y - size / 2);
    //   vertices.push_back(z - size / 2);
    //   // ... [Same vertex definitions as before]
    //   // Vertex 7: top-left-front
    //   vertices.push_back(x - size / 2);
    //   vertices.push_back(y + size / 2);
    //   vertices.push_back(z + size / 2);

    //   // Define the 12 triangles (same as before)
    //   // Front face
    //   indices.push_back(currentIndex + 4);
    //   indices.push_back(currentIndex + 5);
    //   indices.push_back(currentIndex + 6);
    //   // ... [Same indices definitions as before]
      
    //   currentIndex += 8;
    // }

    VAO = 0;
    VBO = 0;
    EBO = 0;

    // OpenGL initialization
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    // check for errors
    if (glGetError() != 0)
    {
        std::cerr << "Error generating vertex array" << std::endl;
    }

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), 0);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), 0);

    glBindVertexArray(0);
  }

  void GLRender::getNodeData(openvdb::FloatGrid::TreeType::NodeIter &iter)
  {
    openvdb::FloatGrid::TreeType::RootNodeType::ChildNodeType::ChildNodeType::ChildNodeType *node = nullptr;
    iter.getNode(node);
    auto origin = node->origin();
    auto dim = node->dim();

    printf("Origin: %f, %f, %f\n", origin[0], origin[1], origin[2]);
  }
}