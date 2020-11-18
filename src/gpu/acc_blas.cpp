#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
#include "acc_blas.hpp"

namespace accblas {

::acc::blas::handle_t&
null_stream_handle()
{
    static ::acc::blas::handle_t null_stream_handle_;
    return null_stream_handle_;
}

std::vector<::acc::blas::handle_t>&
stream_handles()
{
    static std::vector<::acc::blas::handle_t> stream_handles_;
    return stream_handles_;
}

#if defined(SIRIUS_CUDA)
namespace xt {
cublasXtHandle_t&
cublasxt_handle()
{
    static cublasXtHandle_t handle;
    return handle;
}
} // namespace xt
#endif
}

#endif
