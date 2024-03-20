/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file eigensolver.cpp
 *
 *  \brief Contains implementation of eigensolver factory.
 */

#include "eigensolver.hpp"
#include "eigenproblem.hpp"

namespace sirius {

namespace la {

std::unique_ptr<Eigensolver>
Eigensolver_factory(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);

    Eigensolver* ptr = nullptr;
    switch (get_ev_solver_t(name__)) {
        case ev_solver_t::lapack: {
            ptr = new Eigensolver_lapack();
            break;
        }
#if defined(SIRIUS_SCALAPACK)
        case ev_solver_t::scalapack: {
            ptr = new Eigensolver_scalapack();
            break;
        }
#endif
#if defined(SIRIUS_DLAF)
        case ev_solver_t::dlaf: {
            ptr = new Eigensolver_dlaf();
            break;
        }
#endif
#if defined(SIRIUS_ELPA)
        case ev_solver_t::elpa: {
            if (name__ == "elpa1") {
                ptr = new Eigensolver_elpa(1);
            } else {
                ptr = new Eigensolver_elpa(2);
            }
            break;
        }
#endif
#if defined(SIRIUS_MAGMA)
        case ev_solver_t::magma: {
            ptr = new Eigensolver_magma();
            break;
        }
        case ev_solver_t::magma_gpu: {
            ptr = new Eigensolver_magma_gpu();
            break;
        }
#endif
#if defined(SIRIUS_CUDA)
        case ev_solver_t::cusolver: {
            ptr = new Eigensolver_cuda();
            break;
        }
#endif
        default: {
            RTE_THROW("not compiled with the selected eigen-solver");
        }
    }
    return std::unique_ptr<Eigensolver>(ptr);
}

} // namespace la

} // namespace sirius
