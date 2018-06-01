// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file cuda.hpp
 *   
 *  \brief Interface to CUDA related functions.
 */
#ifndef __CUDA_HPP__
#define __CUDA_HPP__

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <nvToolsExt.h>

#include <complex>
#include <vector>
#include <stdio.h>

inline void stack_backtrace()
{
    void *array[10];
    char **strings;
    int size = backtrace(array, 10);
    strings = backtrace_symbols(array, size);
    printf ("Stack backtrace:\n");
    for (int i = 0; i < size; i++) {
        printf ("%s\n", strings[i]);
    }
    raise(SIGQUIT);
}

#ifdef NDEBUG
#define CALL_CUDA(func__, args__)                                                                                  \
{                                                                                                                  \
    cudaError_t error = func__ args__;                                                                             \
    if (error != cudaSuccess) {                                                                                    \
        char nm[1024];                                                                                             \
        gethostname(nm, 1024);                                                                                     \
        printf("hostname: %s\n", nm);                                                                              \
        printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__, cudaGetErrorString(error)); \
        stack_backtrace();                                                                                         \
    }                                                                                                              \
}
#else
#define CALL_CUDA(func__, args__)                                                                                  \
{                                                                                                                  \
    cudaError_t error;                                                                                             \
    func__ args__;                                                                                                 \
    cudaDeviceSynchronize();                                                                                       \
    error = cudaGetLastError();                                                                                    \
    if (error != cudaSuccess) {                                                                                    \
        char nm[1024];                                                                                             \
        gethostname(nm, 1024);                                                                                     \
        printf("hostname: %s\n", nm);                                                                              \
        printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__, cudaGetErrorString(error)); \
        stack_backtrace();                                                                                         \
    }                                                                                                              \
}
#endif

namespace acc {

/// return the gpu id
inline int &device_id()
{
	static int dev_id_;
	return dev_id_;
}

/// set the gpu id
inline int set_device_id(int dev_id)
{
       device_id() = dev_id;
       cudaSetDevice(device_id());
}

/// run calculations on the initial gpu
inline int set_device()
{
	cudaSetDevice(device_id());
}
/// Vector of CUDA streams.
inline std::vector<cudaStream_t>& streams()
{
    static std::vector<cudaStream_t> streams_;
    return streams_;
}

/// Return a single CUDA stream.
inline cudaStream_t stream(int id__)
{
    return (id__ == -1) ? NULL : streams()[id__];
}

/// Reset device.
inline void reset()
{
    CALL_CUDA(cudaDeviceReset, ());
}

/// Synchronize device.
inline void sync()
{
    CALL_CUDA(cudaDeviceSynchronize, ());
}

// Get free memory in bytes.
inline size_t get_free_mem()
{
    size_t free, total;
    CALL_CUDA(cudaMemGetInfo, (&free, &total));

    return free;
}

/// Get the number of devices.
inline int num_devices()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
}

inline void print_device_info(int device_id__)
{
    cudaDeviceProp devprop;
     
    CALL_CUDA(cudaGetDeviceProperties, (&devprop, device_id__));
    
    printf("  name                             : %s \n",       devprop.name);
    printf("  major                            : %i \n",       devprop.major);
    printf("  minor                            : %i \n",       devprop.minor);
    printf("  asyncEngineCount                 : %i \n",       devprop.asyncEngineCount);
    printf("  canMapHostMemory                 : %i \n",       devprop.canMapHostMemory);
    printf("  clockRate                        : %i kHz \n",   devprop.clockRate);
    printf("  concurrentKernels                : %i \n",       devprop.concurrentKernels);
    printf("  ECCEnabled                       : %i \n",       devprop.ECCEnabled);
    printf("  l2CacheSize                      : %i kB \n",    devprop.l2CacheSize >> 10);
    printf("  maxGridSize                      : %i %i %i \n", devprop.maxGridSize[0], devprop.maxGridSize[1], devprop.maxGridSize[2]);
    printf("  maxThreadsDim                    : %i %i %i \n", devprop.maxThreadsDim[0], devprop.maxThreadsDim[1], devprop.maxThreadsDim[2]);
    printf("  maxThreadsPerBlock               : %i \n",       devprop.maxThreadsPerBlock);
    printf("  maxThreadsPerMultiProcessor      : %i \n",       devprop.maxThreadsPerMultiProcessor);
    printf("  memoryBusWidth                   : %i bits \n",  devprop.memoryBusWidth);
    printf("  memoryClockRate                  : %i kHz \n",   devprop.memoryClockRate);
    printf("  memPitch                         : %zi \n",      devprop.memPitch);
    printf("  multiProcessorCount              : %i \n",       devprop.multiProcessorCount);
    printf("  regsPerBlock                     : %i \n",       devprop.regsPerBlock);
    printf("  sharedMemPerBlock                : %li kB \n",   devprop.sharedMemPerBlock >> 10);
    printf("  totalConstMem                    : %li kB \n",   devprop.totalConstMem >> 10);
    printf("  totalGlobalMem                   : %li kB \n",   devprop.totalGlobalMem >> 10);
    printf("  available memory                 : %li kB \n",   get_free_mem() >> 10);
}

/// Get number of streams.
inline int num_streams()
{
    return static_cast<int>(streams().size());
}

/// Create CUDA streams.
inline void create_streams(int num_streams__)
{
    streams() = std::vector<cudaStream_t>(num_streams__);

    //for (int i = 0; i < num_streams; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    for (int i = 0; i < num_streams(); i++) {
        CALL_CUDA(cudaStreamCreate, (&streams()[i]));
    }
}

/// Destroy CUDA streams.
inline void destroy_streams()
{
    for (int i = 0; i < num_streams(); i++) {
        CALL_CUDA(cudaStreamDestroy, (stream(i)));
    }
}

/// Synchronize a single stream.
inline void sync_stream(int stream_id__)
{
    CALL_CUDA(cudaStreamSynchronize, (stream(stream_id__)));
}

/// Copy memory inside a device.
template <typename T>
inline void copy(T* target__, T const* source__, size_t n__)
{
    CALL_CUDA(cudaMemcpy, (target__, source__, n__ * sizeof(T), cudaMemcpyDeviceToDevice));
}

/// Copy memory from host to device.
template <typename T>
inline void copyin(T* target__, T const* source__, size_t n__)
{
    CALL_CUDA(cudaMemcpy, (target__, source__, n__ * sizeof(T), cudaMemcpyHostToDevice));
}

/// Asynchronous copy from host to device.
template <typename T>
inline void copyin(T* target__, T const* source__, size_t n__, int stream_id__)
{
    CALL_CUDA(cudaMemcpyAsync, (target__, source__, n__ * sizeof(T), cudaMemcpyHostToDevice, stream(stream_id__)));
}

/// 2D copy to the device.
template <typename T>
inline void copyin(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__)
{
    CALL_CUDA(cudaMemcpy2D, (target__, ld1__ * sizeof(T), source__, ld2__ * sizeof(T), nrow__ * sizeof(T), ncol__, cudaMemcpyHostToDevice));
}

/// Asynchronous 2D copy to the device.
template <typename T>
inline void copyin(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__, int stream_id__)
{
    CALL_CUDA(cudaMemcpy2DAsync, (target__, ld1__ * sizeof(T), source__, ld2__ * sizeof(T), nrow__ * sizeof(T), ncol__, cudaMemcpyHostToDevice, stream(stream_id__)));
}

/// Copy memory from device to host.
template <typename T>
inline void copyout(T* target__, T const* source__, size_t n__)
{
    CALL_CUDA(cudaMemcpy, (target__, source__, n__ * sizeof(T), cudaMemcpyDeviceToHost));
}

/// Asynchronous copy from device to host.
template <typename T>
inline void copyout(T* target__, T const* source__, size_t n__, int stream_id__)
{
    CALL_CUDA(cudaMemcpyAsync, (target__, source__, n__ * sizeof(T), cudaMemcpyDeviceToHost, stream(stream_id__)));
}

/// 2D copy from device to host.
template <typename T>
inline void copyout(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__)
{
    CALL_CUDA(cudaMemcpy2D, (target__, ld1__ * sizeof(T), source__, ld2__ * sizeof(T), nrow__ * sizeof(T), ncol__, cudaMemcpyDeviceToHost));
}

/// Asynchronous 2D copy from device to host.
template <typename T>
inline void copyout(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__, int stream_id__)
{
    CALL_CUDA(cudaMemcpy2D, (target__, ld1__ * sizeof(T), source__, ld2__ * sizeof(T), nrow__ * sizeof(T), ncol__, cudaMemcpyDeviceToHost, stream(stream_id__)));
}

/// Zero the device memory.
template <typename T>
inline void zero(T* ptr__, size_t n__)
{
    CALL_CUDA(cudaMemset, (ptr__, 0, n__ * sizeof(T)));
}

/// Zero the 2D block of device memory.
template <typename T>
inline void zero(T* ptr__, int ld__, int nrow__, int ncol__)
{
    CALL_CUDA(cudaMemset2D, (ptr__, ld__ * sizeof(T), 0, nrow__ * sizeof(T), ncol__));
}

/// Allocate memory on the GPU.
template <typename T>
inline T* allocate(size_t size__) {
    T* ptr;
    CALL_CUDA(cudaMalloc, (&ptr, size__ * sizeof(T)));
    return ptr;
}

/// Deallocate GPU memory.
inline void deallocate(void* ptr__)
{
    CALL_CUDA(cudaFree, (ptr__));
}

/// Allocate pinned memory on the host.
template <typename T>
inline T* allocate_host(size_t size__) {
    T* ptr;
    CALL_CUDA(cudaMallocHost, (&ptr, size__ * sizeof(T)));
    return ptr;
}

/// Deallocate host memory.
inline void deallocate_host(void* ptr__)
{
    CALL_CUDA(cudaFreeHost, (ptr__));
}

inline void begin_range_marker(const char* label__)
{
    nvtxRangePushA(label__);
}

inline void end_range_marker()
{
    nvtxRangePop();
}

template <typename T>
inline void register_host(T* ptr__, size_t size__)
{
    assert(ptr__);
    
    CALL_CUDA(cudaHostRegister, (ptr__, size__ * sizeof(T), cudaHostRegisterMapped));
}

inline void unregister_host(void* ptr)
{
    CALL_CUDA(cudaHostUnregister, (ptr));
}

inline void check_last_error()
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error != cudaSuccess\n");
    }
}

inline bool check_device_ptr(void const* ptr__)
{
    set_device();
    cudaPointerAttributes attr;
    cudaError_t error = cudaPointerGetAttributes(&attr, ptr__);
    cudaGetLastError();
    if (error != cudaSuccess) {
        return false;
    }
    if (attr.memoryType == cudaMemoryTypeDevice) {
        return true;
    }
    return false;
}

} // namespace acc

extern "C" void scale_matrix_columns_gpu(int nrow, int ncol, void* mtrx, double* a);

extern "C" void scale_matrix_rows_gpu(int nrow, int ncol, void* mtrx, double const* v);

extern "C" void scale_matrix_elements_gpu(cuDoubleComplex* ptr__,
                                          int ld__,
                                          int nrow__,
                                          int ncol__,
                                          double beta__);

#endif
