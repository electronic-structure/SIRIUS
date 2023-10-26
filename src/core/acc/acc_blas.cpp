#if defined(SIRIUS_CUDA) || defined(SIRIUS_ROCM)
#include "acc_blas.hpp"

namespace sirius {

namespace acc {

namespace blas {

acc::blas_api::handle_t&
null_stream_handle()
{
    static acc::blas_api::handle_t null_stream_handle_;
    return null_stream_handle_;
}

std::vector<acc::blas_api::handle_t>&
stream_handles()
{
    static std::vector<acc::blas_api::handle_t> stream_handles_;
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
} // blas

} // acc

} // sirius

#endif
