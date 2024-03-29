/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file acc.hpp
 *
 *  \brief Interface to accelerators API.
 *
 */

#ifndef __ACC_HPP__
#define __ACC_HPP__

#if defined(SIRIUS_CUDA)
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <cuComplex.h>
#endif

#if defined(SIRIUS_ROCM)
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#endif

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>

#include <complex>
#include <vector>
#include <stdio.h>

namespace sirius {

#if defined(SIRIUS_CUDA)
#define GPU_PREFIX(x) cuda##x
#elif defined(SIRIUS_ROCM)
#define GPU_PREFIX(x) hip##x
#endif

#if defined(SIRIUS_CUDA)
using acc_stream_t = cudaStream_t;
#elif defined(SIRIUS_ROCM)
using acc_stream_t = hipStream_t;
#else
using acc_stream_t = void*;
#endif

#if defined(SIRIUS_CUDA)
using acc_error_t = cudaError_t;
#elif defined(SIRIUS_ROCM)
using acc_error_t = hipError_t;
#else
using acc_error_t = void;
#endif

#if defined(SIRIUS_CUDA)
using acc_complex_float_t  = cuFloatComplex;
using acc_complex_double_t = cuDoubleComplex;
#define make_accDoubleComplex make_cuDoubleComplex
#define make_accFloatComplex make_cuFloatComplex
#define accCadd cuCadd
#define accCsub cuCsub
#define accCmul cuCmul
#define accCdiv cuCdiv
#define accConj cuConj
#define accCaddf cuCaddf
#define accCsubf cuCsubf
#define accCmulf cuCmulf
#define accCdivf cuCdivf
#define accConjf cuConjf
#define ACC_DYNAMIC_SHARED(type, var) extern __shared__ type var[];

#elif defined(SIRIUS_ROCM)
using acc_complex_float_t  = hipFloatComplex;
using acc_complex_double_t = hipDoubleComplex;
#define make_accDoubleComplex make_hipDoubleComplex
#define make_accFloatComplex make_hipFloatComplex
#define accCadd hipCadd
#define accCsub hipCsub
#define accCmul hipCmul
#define accCdiv hipCdiv
#define accConj hipConj
#define accCaddf hipCaddf
#define accCsubf hipCsubf
#define accCmulf hipCmulf
#define accCdivf hipCdivf
#define accConjf hipConjf
#define ACC_DYNAMIC_SHARED(type, var) HIP_DYNAMIC_SHARED(type, var)
#endif

/// helper type traits to template kernel function
#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
template <typename T>
struct GPU_Complex;

template <>
struct GPU_Complex<double>
{
    using type = acc_complex_double_t;
};

template <>
struct GPU_Complex<float>
{
    using type = acc_complex_float_t;
};

template <typename T>
using gpu_complex_type = typename GPU_Complex<T>::type;
#endif

/// Namespace for accelerator-related functions.
namespace acc {

/// Helper class to wrap stream id (integer number).
class stream_id
{
  private:
    int id_;

  public:
    explicit stream_id(int id__)
        : id_(id__)
    {
    }
    inline int
    operator()() const
    {
        return id_;
    }
};

inline void
stack_backtrace()
{
    void* array[10];
    char** strings;
    int size = backtrace(array, 10);
    strings  = backtrace_symbols(array, size);
    std::printf("Stack backtrace:\n");
    for (int i = 0; i < size; i++) {
        std::printf("%s\n", strings[i]);
    }
    raise(SIGQUIT);
}

/// Get the number of devices.
int
num_devices();

#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
#define CALL_DEVICE_API(func__, args__)                                                                                \
    {                                                                                                                  \
        if (acc::num_devices()) {                                                                                      \
            acc_error_t error;                                                                                         \
            error = GPU_PREFIX(func__) args__;                                                                         \
            if (error != GPU_PREFIX(Success)) {                                                                        \
                char nm[1024];                                                                                         \
                gethostname(nm, 1024);                                                                                 \
                std::printf("hostname: %s\n", nm);                                                                     \
                std::printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__,                    \
                            GPU_PREFIX(GetErrorString)(error));                                                        \
                stack_backtrace();                                                                                     \
            }                                                                                                          \
        }                                                                                                              \
    }
#else
#define CALL_DEVICE_API(func__, args__)
#endif

/// Set the GPU id.
inline void
set_device_id(int id__)
{
    if (num_devices() > 0) {
        CALL_DEVICE_API(SetDevice, (id__));
    }
}

/// Get current device ID.
inline int
get_device_id()
{
    int id{0};
    CALL_DEVICE_API(GetDevice, (&id));
    return id;
}

/// Vector of device streams.
std::vector<acc_stream_t>&
streams();

/// Return a single device stream.
inline acc_stream_t
stream(stream_id sid__)
{
    assert(sid__() < int(streams().size()));
    return (sid__() == -1) ? NULL : streams()[sid__()];
}

/// Get number of streams.
inline int
num_streams()
{
    return static_cast<int>(streams().size());
}

/// Create CUDA streams.
inline void
create_streams(int num_streams__)
{
    streams() = std::vector<acc_stream_t>(num_streams__);

    // for (int i = 0; i < num_streams; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    for (int i = 0; i < num_streams(); i++) {
        CALL_DEVICE_API(StreamCreate, (&streams()[i]));
    }
}

/// Destroy CUDA streams.
inline void
destroy_streams()
{
    for (int i = 0; i < num_streams(); i++) {
        CALL_DEVICE_API(StreamDestroy, (stream(stream_id(i))));
    }
}

/// Synchronize a single stream.
inline void
sync_stream(stream_id sid__)
{
    CALL_DEVICE_API(StreamSynchronize, (stream(sid__)));
}

/// Reset device.
inline void
reset()
{
#ifdef SIRIUS_CUDA
    CALL_DEVICE_API(ProfilerStop, ());
#endif
    CALL_DEVICE_API(DeviceReset, ());
}

/// Synchronize device.
inline void
sync()
{
    CALL_DEVICE_API(DeviceSynchronize, ());
}

// Get free memory in bytes.
inline size_t
get_free_mem()
{
    size_t free{0};
#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
    size_t total{0};
    CALL_DEVICE_API(MemGetInfo, (&free, &total));
#endif
    return free;
}

inline void
print_device_info(int device_id__, std::ostream& out__)
{
#if defined(SIRIUS_CUDA)
    cudaDeviceProp devprop;
#elif defined(SIRIUS_ROCM)
    hipDeviceProp_t devprop;
#endif

    CALL_DEVICE_API(GetDeviceProperties, (&devprop, device_id__));

#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
    out__ << "  name                             : " << std::string(devprop.name) << std::endl
          << "  major                            : " << devprop.major << std::endl
          << "  minor                            : " << devprop.minor << std::endl
          << "  clockRate                        : " << devprop.clockRate << " kHz" << std::endl
          << "  memoryClockRate                  : " << devprop.memoryClockRate << " kHz" << std::endl
          << "  memoryBusWidth                   : " << devprop.memoryBusWidth << " bits" << std::endl
          << "  sharedMemPerBlock                : " << (devprop.sharedMemPerBlock >> 10) << " kB" << std::endl
          << "  totalConstMem                    : " << (devprop.totalConstMem >> 10) << " kB" << std::endl
          << "  totalGlobalMem                   : " << (devprop.totalGlobalMem >> 10) << " kB" << std::endl
          << "  available memory                 : " << (get_free_mem() >> 10) << " kB" << std::endl
          << "  l2CacheSize                      : " << (devprop.l2CacheSize >> 10) << " kB" << std::endl
          << "  warpSize                         : " << devprop.warpSize << std::endl
          << "  regsPerBlock                     : " << devprop.regsPerBlock << std::endl
          << "  canMapHostMemory                 : " << devprop.canMapHostMemory << std::endl
          << "  concurrentKernels                : " << devprop.concurrentKernels << std::endl
          << "  maxGridSize                      : " << devprop.maxGridSize[0] << " " << devprop.maxGridSize[1] << " "
          << devprop.maxGridSize[2] << std::endl
          << "  maxThreadsDim                    : " << devprop.maxThreadsDim[0] << " " << devprop.maxThreadsDim[1]
          << " " << devprop.maxThreadsDim[2] << std::endl
          << "  maxThreadsPerBlock               : " << devprop.maxThreadsPerBlock << std::endl
          << "  maxThreadsPerMultiProcessor      : " << devprop.maxThreadsPerMultiProcessor << std::endl
          << "  multiProcessorCount              : " << devprop.multiProcessorCount << std::endl
          << "  pciBusID                         : " << devprop.pciBusID << std::endl
          << "  pciDeviceID                      : " << devprop.pciDeviceID << std::endl
          << "  pciDomainID                      : " << devprop.pciDomainID << std::endl;
#if defined(SIRIUS_CUDA)
    out__ << "  regsPerMultiprocessor            : " << devprop.regsPerMultiprocessor << std::endl
          << "  asyncEngineCount                 : " << devprop.asyncEngineCount << std::endl
          << "  ECCEnabled                       : " << devprop.ECCEnabled << std::endl
          << "  memPitch                         : " << devprop.memPitch << std::endl;
#endif
    // this is cuda10
    // printf("  uuid                             : ");
    // for (int s = 0; s < 16; s++) {
    //     std::printf("%#2x ", (unsigned char)devprop.uuid.bytes[s]);
    // }
    // printf("\n");
#endif
}

/// Copy memory inside a device.
template <typename T>
inline void
copy(T* target__, T const* source__, size_t n__)
{
    assert(source__ != nullptr);
    assert(target__ != nullptr);
    CALL_DEVICE_API(Memcpy, (target__, source__, n__ * sizeof(T), GPU_PREFIX(MemcpyDeviceToDevice)));
}

/// 2D copy inside a device.
template <typename T>
inline void
copy(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__)
{
    CALL_DEVICE_API(Memcpy2D, (target__, ld1__ * sizeof(T), source__, ld2__ * sizeof(T), nrow__ * sizeof(T), ncol__,
                               GPU_PREFIX(MemcpyDeviceToDevice)));
}

/// Copy memory from host to device.
template <typename T>
inline void
copyin(T* target__, T const* source__, size_t n__)
{
    CALL_DEVICE_API(Memcpy, (target__, source__, n__ * sizeof(T), GPU_PREFIX(MemcpyHostToDevice)));
}

/// Asynchronous copy from host to device.
template <typename T>
inline void
copyin(T* target__, T const* source__, size_t n__, stream_id sid__)
{
    CALL_DEVICE_API(MemcpyAsync, (target__, source__, n__ * sizeof(T), GPU_PREFIX(MemcpyHostToDevice), stream(sid__)));
}

/// 2D copy to the device.
template <typename T>
inline void
copyin(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__)
{
    CALL_DEVICE_API(Memcpy2D, (target__, ld1__ * sizeof(T), source__, ld2__ * sizeof(T), nrow__ * sizeof(T), ncol__,
                               GPU_PREFIX(MemcpyHostToDevice)));
}

/// Asynchronous 2D copy to the device.
template <typename T>
inline void
copyin(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__, stream_id sid__)
{
    CALL_DEVICE_API(Memcpy2DAsync, (target__, ld1__ * sizeof(T), source__, ld2__ * sizeof(T), nrow__ * sizeof(T),
                                    ncol__, GPU_PREFIX(MemcpyHostToDevice), stream(sid__)));
}

/// Copy memory from device to host.
template <typename T>
inline void
copyout(T* target__, T const* source__, size_t n__)
{
    CALL_DEVICE_API(Memcpy, (target__, source__, n__ * sizeof(T), GPU_PREFIX(MemcpyDeviceToHost)));
}

/// Asynchronous copy from device to host.
template <typename T>
inline void
copyout(T* target__, T const* source__, size_t n__, stream_id sid__)
{
    CALL_DEVICE_API(MemcpyAsync, (target__, source__, n__ * sizeof(T), GPU_PREFIX(MemcpyDeviceToHost), stream(sid__)));
}

/// 2D copy from device to host.
template <typename T>
inline void
copyout(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__)
{
    CALL_DEVICE_API(Memcpy2D, (target__, ld1__ * sizeof(T), source__, ld2__ * sizeof(T), nrow__ * sizeof(T), ncol__,
                               GPU_PREFIX(MemcpyDeviceToHost)));
}

/// Asynchronous 2D copy from device to host.
template <typename T>
inline void
copyout(T* target__, int ld1__, T const* source__, int ld2__, int nrow__, int ncol__, stream_id sid__)
{
    CALL_DEVICE_API(Memcpy2DAsync, (target__, ld1__ * sizeof(T), source__, ld2__ * sizeof(T), nrow__ * sizeof(T),
                                    ncol__, GPU_PREFIX(MemcpyDeviceToHost), stream(sid__)));
}

/// Zero the device memory.
template <typename T>
inline void
zero(T* ptr__, size_t n__)
{
    CALL_DEVICE_API(Memset, (ptr__, 0, n__ * sizeof(T)));
}

template <typename T>
inline void
zero(T* ptr__, size_t n__, stream_id sid__)
{
    CALL_DEVICE_API(MemsetAsync, (ptr__, 0, n__ * sizeof(T), stream(sid__)));
}

/// Zero the 2D block of device memory.
template <typename T>
inline void
zero(T* ptr__, int ld__, int nrow__, int ncol__)
{
    CALL_DEVICE_API(Memset2D, (ptr__, ld__ * sizeof(T), 0, nrow__ * sizeof(T), ncol__));
}

/// Allocate memory on the GPU.
template <typename T>
inline T*
allocate(size_t size__)
{
    T* ptr{nullptr};
#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
    // CALL_DEVICE_API(Malloc, (&ptr, size__ * sizeof(T)));
    if (acc::num_devices()) {
        acc_error_t error;
        error = GPU_PREFIX(Malloc)(&ptr, size__ * sizeof(T));
        if (error != GPU_PREFIX(Success)) {
            std::printf("Device memory allocation of %li MB failed; available memory %li MB\n",
                        (size__ * sizeof(T)) >> 20, get_free_mem() >> 20);
            stack_backtrace();
        }
    }
#endif
    return ptr;
}

/// Deallocate GPU memory.
inline void
deallocate(void* ptr__)
{
    CALL_DEVICE_API(Free, (ptr__));
}

/// Allocate pinned memory on the host.
template <typename T>
inline T*
allocate_host(size_t size__)
{
    T* ptr{nullptr};
#if defined(SIRIUS_CUDA)
    CALL_DEVICE_API(MallocHost, (&ptr, size__ * sizeof(T)));
#endif
#if defined(SIRIUS_ROCM)
    CALL_DEVICE_API(HostMalloc, (&ptr, size__ * sizeof(T)));
#endif
    return ptr;
}

/// Deallocate host memory.
inline void
deallocate_host(void* ptr__)
{
#if defined(SIRIUS_CUDA)
    CALL_DEVICE_API(FreeHost, (ptr__));
#endif
#if defined(SIRIUS_ROCM)
    CALL_DEVICE_API(HostFree, (ptr__));
#endif
}

#if defined(SIRIUS_CUDA)
inline void
begin_range_marker(const char* label__)
{
    nvtxRangePushA(label__);
}

inline void
end_range_marker()
{
    nvtxRangePop();
}

template <typename T>
inline void
register_host(T* ptr__, size_t size__)
{
    assert(ptr__);

    CALL_DEVICE_API(HostRegister, (ptr__, size__ * sizeof(T), cudaHostRegisterMapped));
}

inline void
unregister_host(void* ptr)
{
    CALL_DEVICE_API(HostUnregister, (ptr));
}

inline bool
check_last_error()
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::printf("CUDA error != cudaSuccess\n");
        return true;
    }
    return false;
}

inline bool
check_device_ptr(void const* ptr__)
{
    cudaPointerAttributes attr;
    cudaError_t error = cudaPointerGetAttributes(&attr, ptr__);
    // cudaGetLastError();
    if (error != cudaSuccess) {
        return false;
    }
    if (attr.devicePointer) {
        return true;
    }
    return false;
}

#endif

} // namespace acc

#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
extern "C" {
void
scale_matrix_rows_gpu(int nrow, int ncol, acc_complex_double_t* mtrx, double const* v);
void
scale_matrix_elements_gpu(acc_complex_double_t* ptr__, int ld__, int nrow__, int ncol__, double beta__);
}
#endif

} // namespace sirius

#if defined(SIRIUS_GPU)

#endif

#endif // __ACC_HPP__
