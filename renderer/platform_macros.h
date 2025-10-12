#pragma once

// Platform-specific macros for CUDA/CPU compatibility

#ifdef USE_CUDA
    // CUDA build - use native CUDA decorators
    // __device__, __host__, __global__, __hostdev__ are already defined by CUDA
    
#else
    // CPU build - define CUDA decorators as inline or empty
    
    #ifndef __device__
        #define __device__
    #endif
    
    #ifndef __host__
        #define __host__
    #endif
    
    #ifndef __hostdev__
        #define __hostdev__
    #endif
    
    #ifndef __global__
        // __global__ should not be used in CPU code
        #define __global__ static inline void ERROR_GLOBAL_NOT_SUPPORTED_ON_CPU
    #endif
    
    // Additional CUDA keywords that might be used
    #ifndef __shared__
        #define __shared__
    #endif
    
    #ifndef __constant__
        #define __constant__
    #endif
    
#endif

// Common utility macros
#define DEVICE_FUNC __device__
#define HOST_FUNC __host__
#define HOST_DEVICE_FUNC __hostdev__
