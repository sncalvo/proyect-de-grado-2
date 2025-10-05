#pragma once

#ifdef USE_CUDA
    #include "ComputePrimitives.cuh"
#else
    #include <cstring>
    #include <iostream>
    #include "random_interface.h"
    
    // CPU-based compute primitives that mimic CUDA interface
    
    inline void computeSync(bool useCuda, const char* file, int line) {
        // No-op for CPU - execution is already synchronous
    }
    
    inline void computeFill(bool useCuda, void* data, uint8_t value, size_t size) {
        std::memset(data, value, size);
    }
    
    inline void computeDownload(bool useCuda, void* dst, const void* src, size_t size) {
        std::memcpy(dst, src, size);
    }
    
    inline void computeCopy(bool useCuda, void* dst, const void* src, size_t size) {
        std::memcpy(dst, src, size);
    }
    
    // CPU implementation of parallel foreach
    template<typename FunctorT, typename... Args>
    inline void computeForEach(bool useCuda, int numItems, int blockSize, 
                               const char* file, int line, 
                               const FunctorT& op, Args... args) {
        if (numItems == 0)
            return;
        
        // Allocate random states for each "thread"
        // For CPU, we'll use a reasonable number of states
        const int numStates = std::min(numItems, 128);
        std::vector<RandomState> states(numStates);
        
        // Initialize random states
        for (int i = 0; i < numStates; ++i) {
            random_init(&states[i], 123456ULL, i, 0);
        }
        
        // Sequential execution (can be parallelized with OpenMP later)
        for (int i = 0; i < numItems; ++i) {
            // Use modulo to map items to random states
            RandomState* state = &states[i % numStates];
            op(i, i + 1, state, args...);
        }
    }
    
    // Define __hostdev__ for CPU builds
    #define __hostdev__ inline
    #define __device__ inline
    #define __host__ inline
    
#endif
