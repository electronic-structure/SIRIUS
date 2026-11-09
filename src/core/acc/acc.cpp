/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file acc.cpp
 *
 *  \brief Definition of the functions for the acc:: namespace.
 *
 */
#include <atomic>
#include <iomanip>
#include <sstream>
#include "acc.hpp"
#if defined(SIRIUS_CUDA)
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#endif

namespace sirius {

namespace acc {

int
num_devices()
{
#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
    static std::atomic<int> count(-1);
    if (count.load(std::memory_order_relaxed) == -1) {
        int c;
        if (GPU_PREFIX(GetDeviceCount)(&c) != GPU_PREFIX(Success)) {
            count.store(0, std::memory_order_relaxed);
        } else {
            count.store(c, std::memory_order_relaxed);
        }
    }
    return count.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}

std::vector<acc_stream_t>&
streams()
{
    static std::vector<acc_stream_t> streams_;
    return streams_;
}

void
reset()
{
#ifdef SIRIUS_CUDA
    CALL_DEVICE_API(ProfilerStop, ());
#endif
    CALL_DEVICE_API(DeviceReset, ());
}

void
print_device_info(int device_id__, std::ostream& out__)
{
#if defined(SIRIUS_CUDA)
    cudaDeviceProp devprop;
#elif defined(SIRIUS_ROCM)
    hipDeviceProp_t devprop;
#endif

    CALL_DEVICE_API(GetDeviceProperties, (&devprop, device_id__));

#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
    int clockRate{-1};
    int memoryClockRate{-1};

#if CUDA_VERSION < 13000 || defined(SIRIUS_ROCM)
    clockRate       = devprop.clockRate;
    memoryClockRate = devprop.memoryClockRate;
#else
    // Cf. https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/#changes_to_cudadeviceprop
    CALL_DEVICE_API(DeviceGetAttribute, (&clockRate, cudaDevAttrClockRate, device_id__));
    CALL_DEVICE_API(DeviceGetAttribute, (&memoryClockRate, cudaDevAttrMemoryClockRate, device_id__));
#endif

    out__ << "  name                             : " << std::string(devprop.name) << std::endl
          << "  major                            : " << devprop.major << std::endl
          << "  minor                            : " << devprop.minor << std::endl
          << "  clockRate                        : " << clockRate << " kHz" << std::endl
          << "  memoryClockRate                  : " << memoryClockRate << " kHz" << std::endl
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
#endif
}

std::string
get_uuid(int device_id__)
{
#if defined(SIRIUS_CUDA)
    cudaDeviceProp devprop;
    CALL_DEVICE_API(GetDeviceProperties, (&devprop, device_id__));
    auto const& uuid_bytes = devprop.uuid.bytes;
#elif defined(SIRIUS_ROCM)
    hipUUID uuid;
    CALL_DEVICE_API(DeviceGetUuid, (&uuid, device_id__));
    auto const& uuid_bytes = uuid.bytes;
#endif

    std::stringstream s;
#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
    for (int i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            s << '-';
        }
        s << std::hex << std::setw(2) << std::setfill('0') << (int)uuid_bytes[i];
    }
#endif
    return s.str();
}

#if defined(SIRIUS_CUDA)
void
begin_range_marker(const char* label__)
{
    nvtxRangePushA(label__);
}

void
end_range_marker()
{
    nvtxRangePop();
}
#endif

} // namespace acc

} // namespace sirius
