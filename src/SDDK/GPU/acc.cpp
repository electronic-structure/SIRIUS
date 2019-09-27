/** \file acc.cpp
 *
 *  \brief Definition of the functions for the acc:: namespace.
 *
 */

#include "acc.hpp"

namespace acc {

int num_devices()
{
#if defined(__CUDA) || defined(__ROCM)
    static int count{-1};
    if (count == -1) {
        if (GPU_PREFIX(GetDeviceCount)(&count) != GPU_PREFIX(Success)) {
            count = 0;
        }
    }
    return count;
#else
    return 0;
#endif
}

std::vector<acc_stream_t>& streams()
{
    static std::vector<acc_stream_t> streams_;
    return streams_;
}

} // namespace sddk
