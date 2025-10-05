#pragma once

#ifdef USE_CUDA
    #include <curand.h>
    #include <curand_kernel.h>
    
    using RandomState = curandState;
    
    // CUDA random functions are already defined by curand
    
#else
    #include <random>
    
    // CPU-based random number generator that mimics curandState interface
    struct RandomState {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist{0.0f, 1.0f};
        
        RandomState() : generator(0), dist(0.0f, 1.0f) {}
    };
    
    // Initialize random state (mimics curand_init)
    inline void random_init(RandomState* state, unsigned long long seed, 
                           unsigned long long sequence, unsigned long long offset) {
        state->generator.seed(static_cast<unsigned int>(seed + sequence));
    }
    
    // Generate uniform random float in [0, 1) (mimics curand_uniform)
    inline float random_uniform(RandomState* state) {
        return state->dist(state->generator);
    }
    
    // For compatibility with CUDA code
    #define curand_init(seed, sequence, offset, state) random_init(state, seed, sequence, offset)
    #define curand_uniform(state) random_uniform(state)
    
#endif
