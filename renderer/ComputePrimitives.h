// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ComputePrimitives.h
/// @brief A collection of parallel compute primitives

#pragma once

#if defined(NANOVDB_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <utility>
#include <tuple>

#include "ComputePrimitives.h"

// forward compatibility for C++14 Standard Library
namespace cxx14 {
template<std::size_t...>
struct index_sequence
{
};

template<std::size_t N, std::size_t... Is>
struct make_index_sequence : make_index_sequence<N - 1, N - 1, Is...>
{
};

template<std::size_t... Is>
struct make_index_sequence<0, Is...> : index_sequence<Is...>
{
};
} // namespace cxx14

static inline bool checkCUDA(cudaError_t result, const char* file, const int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime API error " << result << " in file " << file << ", line " << line << " : " << cudaGetErrorString(result) << ".\n";
        return false;
    }
    return true;
}

#define NANOVDB_CUDA_SAFE_CALL(x) checkCUDA(x, __FILE__, __LINE__)

static inline void checkErrorCUDA(cudaError_t result, const char* file, const int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime API error " << result << " in file " << file << ", line " << line << " : " << cudaGetErrorString(result) << ".\n";
        exit(1);
    }
}

#define NANOVDB_CUDA_CHECK_ERROR(result, file, line) checkErrorCUDA(result, file, line)

template<typename Fn, typename... Args>
class ApplyFunc
{
public:
    ApplyFunc(int count, int blockSize, const Fn& fn, Args... args)
        : mCount(count)
        , mBlockSize(blockSize)
        , mArgs(args...)
        , mFunc(fn)
    {
    }

    template<std::size_t... Is>
    void call(int start, int end, cxx14::index_sequence<Is...>) const
    {
        mFunc(start, end, std::get<Is>(mArgs)...);
    }

    void operator()(int i) const
    {
        int start = i * mBlockSize;
        int end = i * mBlockSize + mBlockSize;
        if (end > mCount)
            end = mCount;
        call(start, end, cxx14::make_index_sequence<sizeof...(Args)>());
    }

private:
    int                 mCount;
    int                 mBlockSize;
    Fn                  mFunc;
    std::tuple<Args...> mArgs;
};

template<int WorkPerThread, typename FnT, typename... Args>
__global__ void parallelForKernel(int numItems, FnT f, Args... args)
{
    for (int j=0;j<WorkPerThread;++j)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x + j * blockDim.x * gridDim.x;
        if (i < numItems)
            f(i, i + 1, args...);
    }
}


inline void computeSync(bool useCuda, const char* file, int line)
{
    if (useCuda) {
        NANOVDB_CUDA_CHECK_ERROR(cudaDeviceSynchronize(), file, line);
    }
}

inline void computeFill(bool useCuda, void* data, uint8_t value, size_t size)
{
    if (useCuda) {
        cudaMemset(data, value, size);
    }
}

template<typename FunctorT, typename... Args>
inline void computeForEach(bool useCuda, int numItems, int blockSize, const char* file, int line, const FunctorT& op, Args... args)
{
    if (numItems == 0)
        return;

    if (useCuda) {
        static const int WorkPerThread = 1;
        int blockCount = ((numItems/WorkPerThread) + (blockSize - 1)) / blockSize;
        parallelForKernel<WorkPerThread, FunctorT, Args...><<<blockCount, blockSize, 0, 0>>>(numItems, op, args...);
        NANOVDB_CUDA_CHECK_ERROR(cudaGetLastError(), __FILE__, __LINE__);
    }
}

inline void computeDownload(bool useCuda, void* dst, const void* src, size_t size)
{
    if (useCuda) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
}

inline void computeCopy(bool useCuda, void* dst, const void* src, size_t size)
{
    if (useCuda) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
}
