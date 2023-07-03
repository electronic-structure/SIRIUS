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

/** \file band.cpp
 *
 *  \brief Contains implementation of sirius::Band class.
 */

#include "band.hpp"
#include "context/simulation_context.hpp"
#include "k_point/k_point_set.hpp"
#include "utils/profiler.hpp"

namespace sirius {

/// Constructor
Band::Band(Simulation_context& ctx__)
    : ctx_(ctx__)
    , unit_cell_(ctx__.unit_cell())
    , blacs_grid_(ctx__.blacs_grid())
{
    if (!ctx_.initialized()) {
        RTE_THROW("Simulation_context is not initialized");
    }
}

template <typename T>
void
Band::initialize_subspace(K_point_set& kset__, Hamiltonian0<T>& H0__) const
{
    PROFILE("sirius::Band::initialize_subspace");

    int N{0};

    if (ctx_.cfg().iterative_solver().init_subspace() == "lcao") {
        /* get the total number of atomic-centered orbitals */
        N = unit_cell_.num_ps_atomic_wf().first;
    }

    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__.get<T>(ik);
        auto Hk = H0__(*kp);
        if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
            ::sirius::initialize_subspace<T, T>(Hk, N);
        } else {
            ::sirius::initialize_subspace<T, std::complex<T>>(Hk, N);
        }
    }

    /* reset the energies for the iterative solver to do at least two steps */
    for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
        for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
            for (int i = 0; i < ctx_.num_bands(); i++) {
                kset__.get<T>(ik)->band_energy(i, ispn, 0);
                kset__.get<T>(ik)->band_occupancy(i, ispn, ctx_.max_occupancy());
            }
        }
    }
}

template
void
Band::initialize_subspace<double>(K_point_set& kset__, Hamiltonian0<double>& H0__) const;
#if defined(USE_FP32)
template
void
Band::initialize_subspace<float>(K_point_set& kset__, Hamiltonian0<float>& H0__) const;
#endif

}
