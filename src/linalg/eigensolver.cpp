// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file eigensolver.cpp
 *
 *  \brief Contains implementation of eigensolver factory.
 */

#include "eigensolver.hpp"
#include "eigenproblem.hpp"

std::unique_ptr<Eigensolver> Eigensolver_factory(std::string name__, memory_pool* mpd__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);

    Eigensolver* ptr;
    switch (get_ev_solver_t(name__)) {
        case ev_solver_t::lapack: {
            ptr = new Eigensolver_lapack();
            break;
        }
        case ev_solver_t::scalapack: {
            ptr = new Eigensolver_scalapack();
            break;
        }
        case ev_solver_t::elpa: {
            if (name__ == "elpa1") {
                ptr = new Eigensolver_elpa(1);
            } else {
                ptr = new Eigensolver_elpa(2);
            }
            break;
        }
        case ev_solver_t::magma: {
            ptr = new Eigensolver_magma();
            break;
        }
        case ev_solver_t::magma_gpu: {
            ptr = new Eigensolver_magma_gpu();
            break;
        }
        case ev_solver_t::cusolver: {
            ptr = new Eigensolver_cuda(mpd__);
            break;
        }
        default: {
            TERMINATE("not implemented");
        }
    }
    return std::unique_ptr<Eigensolver>(ptr);
}

