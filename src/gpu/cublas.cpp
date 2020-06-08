#ifdef __CUDA
#include "cublas.hpp"

namespace cublas {

cublasHandle_t&
null_stream_handle()
{
    static cublasHandle_t null_stream_handle_;
    return null_stream_handle_;
}

/// Store the cublas handlers associated with cuda streams.
std::vector<cublasHandle_t>&
stream_handles()
{
    static std::vector<cublasHandle_t> stream_handles_;
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

} // namespace cublas
#endif
