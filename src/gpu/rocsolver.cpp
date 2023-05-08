#ifdef SIRIUS_ROCM

#include "rocsolver.hpp"
#include "acc_blas.hpp"

namespace rocsolver {

::acc::blas::handle_t&
rocsolver_handle()
{
    return ::accblas::null_stream_handle();
}

} // namespace rocsolver
#endif
