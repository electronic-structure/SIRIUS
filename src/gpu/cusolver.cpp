#ifdef SIRIUS_CUDA
#include "cusolver.hpp"

namespace cusolver {

cusolverDnHandle_t&
cusolver_handle()
{
    static cusolverDnHandle_t handle;
    return handle;
}

void
create_handle()
{
    CALL_CUSOLVER(cusolverDnCreate, (&cusolver_handle()));
}

void
destroy_handle()
{
    CALL_CUSOLVER(cusolverDnDestroy, (cusolver_handle()));
}


} // namespace cusolver
#endif
