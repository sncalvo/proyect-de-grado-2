#pragma once

#include <openvdb/openvdb.h>
#include <vector>

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

  private:
    openvdb::FloatGrid::Ptr m_Grid;
    unsigned int VAO, VBO, EBO;

    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    void getNodeData(openvdb::FloatGrid::TreeType::NodeIter &iter);
  };
}
