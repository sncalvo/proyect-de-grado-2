#ifndef H_IMAGE

#define H_IMAGE

#include "buffer_interface.h"

#include <fstream>

// BufferT is now defined in buffer_interface.h

class Image
{
private:
    BufferT m_buffer;
    BufferT m_accumBuffer;  // Accumulation buffer for progressive rendering

    int m_width;
    int m_height;
    int m_currentSample;    // Track current sample count

    float* m_uploadPtr = nullptr;
    float* m_downloadPtr = nullptr;
    float* m_accumUploadPtr = nullptr;
    float* m_accumDownloadPtr = nullptr;

  bool _isLittleEndian() {
      static int x = 1;
      return reinterpret_cast<uint8_t*>(&x)[0] == 1;
  }
public:
    Image(int width, int height)
        : m_width(width), m_height(height), m_currentSample(0)
    {
        m_buffer.init(width * height * 3 * sizeof(float), true);
        m_accumBuffer.init(width * height * 3 * sizeof(float), true);
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

    // Accumulation buffer access (for progressive rendering)
    float* accumBufferUpload()
    {
        if (m_accumUploadPtr == nullptr)
        {
            m_accumBuffer.deviceUpload();
            m_accumUploadPtr = reinterpret_cast<float*>(m_accumBuffer.deviceData());
        }

        return m_accumUploadPtr;
    }

    float* accumBufferDownload()
    {
        if (m_accumDownloadPtr == nullptr)
        {
            m_accumBuffer.deviceDownload();
            m_accumDownloadPtr = (float*)m_accumBuffer.data();
        }

        return m_accumDownloadPtr;
    }

    // Accumulate one sample into the accumulation buffer and update the display buffer
    void accumulateSample()
    {
        m_currentSample++;
        
        // Get both buffers
        float* displayData = deviceDownload();
        float* accumData = accumBufferDownload();
        
        if (!displayData || !accumData) return;
        
        int numElements = m_width * m_height * 3;
        
        // Add current sample to accumulation buffer
        for (int i = 0; i < numElements; i++) {
            accumData[i] += displayData[i];
        }
        
        // Update display buffer with averaged result
        float invSamples = 1.0f / float(m_currentSample);
        for (int i = 0; i < numElements; i++) {
            displayData[i] = accumData[i] * invSamples;
        }
    }

    // Reset for new render
    void resetAccumulation()
    {
        m_currentSample = 0;
        
        // Clear accumulation buffer
        float* accumData = accumBufferDownload();
        if (accumData) {
            int numElements = m_width * m_height * 3;
            for (int i = 0; i < numElements; i++) {
                accumData[i] = 0.0f;
            }
        }
    }

    int getCurrentSample() const { return m_currentSample; }

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
        
        m_accumBuffer.clear();
        m_accumBuffer.init(m_width * m_height * 3 * sizeof(float), true);

        m_uploadPtr = nullptr;
        m_downloadPtr = nullptr;
        m_accumUploadPtr = nullptr;
        m_accumDownloadPtr = nullptr;
        
        m_currentSample = 0;
    }

    int width() const { return m_width; }
    int height() const { return m_height; }
};

#endif // H_IMAGE
