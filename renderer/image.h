#ifndef H_IMAGE

#define H_IMAGE

#include <nanovdb/util/CudaDeviceBuffer.h>

#include <fstream>

using BufferT = nanovdb::CudaDeviceBuffer;

class Image
{
private:
    BufferT m_buffer;

    int m_width;
    int m_height;

    float* m_uploadPtr = nullptr;
    float* m_downloadPtr = nullptr;

  bool _isLittleEndian() {
      static int x = 1;
      return reinterpret_cast<uint8_t*>(&x)[0] == 1;
  }
public:
    Image(int width, int height)
        : m_width(width), m_height(height)
    {
        m_buffer.init(width * height * 3 * sizeof(float), true);
    }

    float* deviceUpload()
    {
        if (m_uploadPtr == nullptr)
        {
            m_buffer.deviceUpload();
            m_uploadPtr = reinterpret_cast<float*>(m_buffer.deviceData());
        }

        return m_uploadPtr;
    }

    float* deviceDownload()
    {
        if (m_downloadPtr == nullptr)
        {
            m_buffer.deviceDownload();
            m_downloadPtr = (float*)m_buffer.data();
        }

        return m_downloadPtr;
    }

    void save(const std::string& filename)
    {
        auto image = deviceDownload();

        if (image == nullptr) {
            throw std::runtime_error("Unable to download image from device");
        }

        float scale = 1.0f;
        if (_isLittleEndian())
            scale = -scale;

        std::fstream fs(filename, std::ios::out | std::ios::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        fs << "PF\n"
            << width() << "\n"
            << height() << "\n"
            << scale << "\n";

        for (int i = 0; i < width() * height() * 3; i += 3) {
            float r = image[i];
            float g = image[i + 1];
            float b = image[i + 2];

            fs.write((char*)&r, sizeof(float));
            fs.write((char*)&g, sizeof(float));
            fs.write((char*)&b, sizeof(float));
        }
    }

    void clear()
    {
        m_buffer.clear();
        m_buffer.init(m_width * m_height * 3 * sizeof(float), true);

        m_uploadPtr = nullptr;
        m_downloadPtr = nullptr;
    }

    int width() const { return m_width; }
    int height() const { return m_height; }
};

#endif // H_IMAGE