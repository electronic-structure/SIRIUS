/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifdef SIRIUS_CUDA
#include "cusolver.hpp"

namespace sirius {
namespace acc {
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
} // namespace acc
} // namespace sirius

#endif
