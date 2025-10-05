#pragma once

#ifdef USE_CUDA
    #include <nanovdb/util/CudaDeviceBuffer.h>
    using BufferT = nanovdb::CudaDeviceBuffer;
#else
    #include <vector>
    #include <cstring>
    #include <cstdint>
    
    // CPU-based buffer implementation that mimics CudaDeviceBuffer interface
    class HostBuffer {
    public:
        HostBuffer() = default;
        ~HostBuffer() = default;
        
        // Static create method (mimics CudaDeviceBuffer::create)
        template<typename T = HostBuffer>
        static HostBuffer create(size_t size, const T* pool = nullptr, bool dummy = false) {
            HostBuffer buffer;
            buffer.init(size, false);
            return buffer;
        }
        
        // Initialize buffer with given size
        void init(size_t size, bool zero = false) {
            m_size = size;
            m_data.resize(size);
            if (zero) {
                std::memset(m_data.data(), 0, size);
            }
        }
        
        // Clear the buffer
        void clear() {
            m_data.clear();
            m_size = 0;
        }
        
        // Get host data pointer
        void* data() {
            return m_data.data();
        }
        
        const void* data() const {
            return m_data.data();
        }
        
        // For CPU builds, device data is same as host data
        void* deviceData() {
            return m_data.data();
        }
        
        const void* deviceData() const {
            return m_data.data();
        }
        
        // Upload to device (no-op for CPU)
        void deviceUpload(void* stream = nullptr, bool sync = true) {
            // No-op: data is already in accessible memory
        }
        
        // Download from device (no-op for CPU)
        void deviceDownload(void* stream = nullptr, bool sync = true) {
            // No-op: data is already in accessible memory
        }
        
        // Get buffer size
        size_t size() const {
            return m_size;
        }
        
        // Check if buffer is empty
        bool empty() const {
            return m_data.empty();
        }
        
    private:
        std::vector<uint8_t> m_data;
        size_t m_size = 0;
    };
    
    using BufferT = HostBuffer;
#endif
