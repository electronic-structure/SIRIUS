#if defined __CUDA || __ROCM
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

namespace xt {
cublasXtHandle_t&
cublasxt_handle()
{
    static cublasXtHandle_t handle;
    return handle;
}
} // namespace xt
}  // xt

#endif
