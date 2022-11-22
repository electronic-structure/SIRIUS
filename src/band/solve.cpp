// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file solve.cpp
 *
 *   \brief Contains interfaces to the sirius::Band solvers.
 */
#include "band.hpp"
#include "davidson.hpp"

namespace sirius {

template <typename T>
void
Band::solve_full_potential(Hamiltonian_k<T>& Hk__, double itsol_tol__) const
{
    print_memory_usage(ctx_.out(), FILE_LINE);
    if (ctx_.cfg().control().use_second_variation()) {
        /* solve non-magnetic Hamiltonian (so-called first variation) */
        auto& itso = ctx_.cfg().iterative_solver();
        if (itso.type() == "exact") {
            diag_full_potential_first_variation_exact(Hk__);
        } else if (itso.type() == "davidson") {
            diag_full_potential_first_variation_davidson(Hk__, itsol_tol__);
        }
        /* generate first-variational states */
        Hk__.kp().generate_fv_states();
        /* solve magnetic Hamiltonian */
        diag_full_potential_second_variation(Hk__);
        /* generate spinor wave-functions */
        Hk__.kp().generate_spinor_wave_functions();
    } else {
        throw std::runtime_error("not implemented");
        //diag_full_potential_single_variation();
    }
    print_memory_usage(ctx_.out(), FILE_LINE);
}

template
void
Band::solve_full_potential<double>(Hamiltonian_k<double>& Hk__, double itsol_tol__) const;

#if defined(USE_FP32)
template<>
void
Band::solve_full_potential<float>(Hamiltonian_k<float>& Hk__, double itsol_tol__) const
{
    RTE_THROW("FP32 is not implemented for FP-LAPW");
}
#endif

template <typename T, typename F>
int
Band::solve_pseudo_potential(Hamiltonian_k<T>& Hk__, double itsol_tol__, double empy_tol__) const
{
    print_memory_usage(ctx_.out(), FILE_LINE);

    int niter{0};

    auto& itso = ctx_.cfg().iterative_solver();
    if (itso.type() == "exact") {
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                diag_pseudo_potential_exact<T, F>(ispn, Hk__);
            }
        } else {
            STOP();
        }
    } else if (itso.type() == "davidson") {
        auto& kp = Hk__.kp();

        auto tolerance = [&](int j__, int ispn__) -> double {

            /* tolerance for occupied states */
            double tol      = itsol_tol__;
            /* if band is empty, make tolerance larger (in most cases we don't need high precision on
             * unoccupied  states) */
            if (std::abs(kp.band_occupancy(j__, ispn__)) < ctx_.min_occupancy() * ctx_.max_occupancy()) {
                tol += empy_tol__;
            }

            return tol;
        };

        std::stringstream s;
        std::ostream* out = (kp.comm().rank() == 0) ? &std::cout : &s;

        auto result = davidson<T, F, davidson_evp_t::hamiltonian>(Hk__, wf::num_bands(ctx_.num_bands()),
                wf::num_mag_dims(ctx_.num_mag_dims()), kp.spinor_wave_functions(), tolerance,
                itso.residual_tolerance(), itso.num_steps(), itso.locking(), itso.subspace_size(),
                itso.converge_by_energy(), itso.extra_ortho(), *out, 0);
        niter = result.niter;
        for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                kp.band_energy(j, ispn, result.eval(j, ispn));
            }
        }
    } else {
        RTE_THROW("unknown iterative solver type");
    }

    ///* check residuals */
    //if (ctx_.cfg().control().verification() >= 2) {
    //    check_residuals<T>(Hk__);
    //    check_wave_functions<T>(Hk__);
    //}

    print_memory_usage(ctx_.out(), FILE_LINE);

    return niter;
}

template <typename T, typename F>
void
Band::solve(K_point_set& kset__, Hamiltonian0<T>& H0__, double itsol_tol__) const
{
    PROFILE("sirius::Band::solve");

    print_memory_usage(ctx_.out(),FILE_LINE);

    double empy_tol{itsol_tol__};
    if (ctx_.cfg().iterative_solver().type() == "davidson") {
        empy_tol = std::max(itsol_tol__ * ctx_.cfg().settings().itsol_tol_ratio(),
                                   ctx_.cfg().iterative_solver().empty_states_tolerance());
        ctx_.out(2, __func__) << "iterative solver tolerance (occupied, empty): " << itsol_tol__ << " "
                              << itsol_tol__ + empy_tol << std::endl;
    }

    int num_dav_iter{0};
    /* solve secular equation and generate wave functions */
    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__.get<T>(ik);

        auto Hk = H0__(*kp);
        if (ctx_.full_potential()) {
            solve_full_potential<T>(Hk, itsol_tol__);
        } else {
            if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
                num_dav_iter += solve_pseudo_potential<T, F>(Hk, itsol_tol__, empy_tol);
            } else {
                num_dav_iter += solve_pseudo_potential<T, std::complex<F>>(Hk, itsol_tol__, empy_tol);
            }
        }
    }
    kset__.comm().allreduce(&num_dav_iter, 1);
    ctx_.num_itsol_steps(num_dav_iter);
    if (!ctx_.full_potential()) {
        ctx_.out(2, __func__) << "average number of iterations: "
                              << static_cast<double>(num_dav_iter) / kset__.num_kpoints() << std::endl;
    }

    /* synchronize eigen-values */
    kset__.sync_band<T, sync_band_t::energy>();

    if (ctx_.verbosity() >= 2) {
        std::stringstream s;
        s << "Lowest band energie" << std::endl;
        int nbnd = std::min(ctx_.cfg().control().num_bands_to_print(), ctx_.num_bands());
        for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
            s << "ik: " << std::setw(2) << ik;
            for (int j = 0; j < nbnd; j++) {
                s << utils::ffmt(12, 6) << kset__.get<T>(ik)->band_energy(j, 0);
            }
            if (ctx_.num_mag_dims() == 1) {
                s << std::endl << "         ";
                for (int j = 0; j < nbnd; j++) {
                    s << utils::ffmt(12, 6) << kset__.get<T>(ik)->band_energy(j, 1);
                }
            }
            s << std::endl;
        }
        ctx_.message(2, __func__, s);
    }
    print_memory_usage(ctx_.out(), FILE_LINE);
}

template
void
Band::solve<double, double>(K_point_set& kset__, Hamiltonian0<double>& H0__, double itsol_tol__) const;

#if defined(USE_FP32)
template
void
Band::solve<float, float>(K_point_set& kset__, Hamiltonian0<float>& H0__, double itsol_tol__) const;

template
void
Band::solve<float, double>(K_point_set& kset__, Hamiltonian0<float>& H0__, double itsol_tol__) const;
#endif

} // namespace
