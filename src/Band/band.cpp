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
#include "simulation_context.hpp"
#include "K_point/k_point_set.hpp"
#include "SDDK/wf_trans.hpp"
#include "SDDK/wf_inner.hpp"
#include "utils/profiler.hpp"

namespace sirius {

/// Constructor
Band::Band(Simulation_context& ctx__)
    : ctx_(ctx__)
    , unit_cell_(ctx__.unit_cell())
    , blacs_grid_(ctx__.blacs_grid())
{
    if (!ctx_.initialized()) {
        TERMINATE("Simulation_context is not initialized");
    }
}

template <typename T>
void
Band::set_subspace_mtrx(int N__, int n__, Wave_functions& phi__, Wave_functions& op_phi__, dmatrix<T>& mtrx__,
                        dmatrix<T>* mtrx_old__) const
{
    PROFILE("sirius::Band::set_subspace_mtrx");

    assert(n__ != 0);
    if (mtrx_old__ && mtrx_old__->size()) {
        assert(&mtrx__.blacs_grid() == &mtrx_old__->blacs_grid());
    }

    /* copy old N x N distributed matrix */
    if (N__ > 0) {
        splindex<splindex_t::block_cyclic> spl_row(N__, mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().rank_row(),
                                       mtrx__.bs_row());
        splindex<splindex_t::block_cyclic> spl_col(N__, mtrx__.blacs_grid().num_ranks_col(), mtrx__.blacs_grid().rank_col(),
                                       mtrx__.bs_col());

        if (mtrx_old__) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spl_col.local_size(); i++) {
                std::copy(&(*mtrx_old__)(0, i), &(*mtrx_old__)(0, i) + spl_row.local_size(), &mtrx__(0, i));
            }
        }

        if (ctx_.control().print_checksum_) {
            double_complex cs(0, 0);
            for (int i = 0; i < spl_col.local_size(); i++) {
                for (int j = 0; j < spl_row.local_size(); j++) {
                    cs += mtrx__(j, i);
                }
            }
            mtrx__.blacs_grid().comm().allreduce(&cs, 1);
            if (ctx_.comm_band().rank() == 0) {
                utils::print_checksum("subspace_mtrx_old", cs);
            }
        }
    }

    /* <{phi,phi_new}|Op|phi_new> */
    inner(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), (ctx_.num_mag_dims() == 3) ? 2 : 0, phi__, 0, N__ + n__,
          op_phi__, N__, n__, mtrx__, 0, N__);

    /* restore lower part */
    if (N__ > 0) {
        if (mtrx__.blacs_grid().comm().size() == 1) {
            #pragma omp parallel for
            for (int i = 0; i < N__; i++) {
                for (int j = N__; j < N__ + n__; j++) {
                    mtrx__(j, i) = utils::conj(mtrx__(i, j));
                }
            }
        } else {
            linalg(linalg_t::scalapack).tranc(n__, N__, mtrx__, 0, N__, mtrx__, N__, 0);
        }
    }

    if (ctx_.control().print_checksum_) {
        splindex<splindex_t::block_cyclic> spl_row(N__ + n__, mtrx__.blacs_grid().num_ranks_row(),
                                                   mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
        splindex<splindex_t::block_cyclic> spl_col(N__ + n__, mtrx__.blacs_grid().num_ranks_col(),
                                                   mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());
        double_complex cs(0, 0);
        for (int i = 0; i < spl_col.local_size(); i++) {
            for (int j = 0; j < spl_row.local_size(); j++) {
                cs += mtrx__(j, i);
            }
        }
        mtrx__.blacs_grid().comm().allreduce(&cs, 1);
        if (ctx_.comm_band().rank() == 0) {
            utils::print_checksum("subspace_mtrx", cs);
        }
    }

    /* kill any numerical noise */
    mtrx__.make_real_diag(N__ + n__);

    /* save new matrix */
    if (mtrx_old__) {
        splindex<splindex_t::block_cyclic> spl_row(N__ + n__, mtrx__.blacs_grid().num_ranks_row(),
                                                   mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
        splindex<splindex_t::block_cyclic> spl_col(N__ + n__, mtrx__.blacs_grid().num_ranks_col(),
                                                   mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < spl_col.local_size(); i++) {
            std::copy(&mtrx__(0, i), &mtrx__(0, i) + spl_row.local_size(), &(*mtrx_old__)(0, i));
        }
    }
}

void
Band::initialize_subspace(K_point_set& kset__, Hamiltonian0& H0__) const
{
    PROFILE("sirius::Band::initialize_subspace");

    int N{0};

    if (ctx_.iterative_solver_input().init_subspace_ == "lcao") {
        /* get the total number of atomic-centered orbitals */
        N = unit_cell_.num_ps_atomic_wf();
    }

    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__[ik];
        auto Hk = H0__(*kp);
        if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
            initialize_subspace<double>(Hk, N);
        } else {
            initialize_subspace<double_complex>(Hk, N);
        }
    }

    /* reset the energies for the iterative solver to do at least two steps */
    for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
        for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
            for (int i = 0; i < ctx_.num_bands(); i++) {
                kset__[ik]->band_energy(i, ispn, 0);
                kset__[ik]->band_occupancy(i, ispn, ctx_.max_occupancy());
            }
        }
    }
}

template <typename T>
void Band::initialize_subspace(Hamiltonian_k& Hk__, int num_ao__) const
{
    PROFILE("sirius::Band::initialize_subspace|kp");

    if (ctx_.control().verification_ >= 1) {
        auto eval = diag_S_davidson<T>(Hk__);
        if (eval[0] <= 0) {
            std::stringstream s;
            s << "[sirius::Band::initialize_subspace] S-operator matrix is not positive definite\n"
              << "  lowest eigen-value: " << eval[0];
            TERMINATE(s);
        }
    }

    /* number of non-zero spin components */
    const int num_sc = (ctx_.num_mag_dims() == 3) ? 2 : 1;

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_bands();

    /* number of basis functions */
    int num_phi = std::max(num_ao__, num_bands / num_sc);

    int num_phi_tot = num_phi * num_sc;

    auto& mp = ctx_.mem_pool(ctx_.host_memory_t());

    ctx_.print_memory_usage(__FILE__, __LINE__);

    /* initial basis functions */
    Wave_functions phi(mp, Hk__.kp().gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);
    for (int ispn = 0; ispn < num_sc; ispn++) {
        phi.pw_coeffs(ispn).prime().zero();
    }

    PROFILE_START("sirius::Band::initialize_subspace|kp|wf");

    /* generate the initial atomic wavefunctions */
    std::vector<int> atoms(ctx_.unit_cell().num_atoms());
    std::iota(atoms.begin(), atoms.end(), 0);
    Hk__.kp().generate_atomic_wave_functions(atoms, [&](int iat){return &ctx_.unit_cell().atom_type(iat).indexb_wfs();},
                                             ctx_.atomic_wf_ri(), phi);

    /* fill remaining wave-functions with pseudo-random guess */
    assert(Hk__.kp().num_gkvec() > num_phi + 10);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_phi - num_ao__; i++) {
        for (int igk_loc = 0; igk_loc < Hk__.kp().num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = Hk__.kp().idxgk(igk_loc);
            if (igk == i + 1) {
                phi.pw_coeffs(0).prime(igk_loc, num_ao__ + i) = 1.0;
            }
            if (igk == i + 2) {
                phi.pw_coeffs(0).prime(igk_loc, num_ao__ + i) = 0.5;
            }
            if (igk == i + 3) {
                phi.pw_coeffs(0).prime(igk_loc, num_ao__ + i) = 0.25;
            }
        }
    }

    std::vector<double> tmp(4096);
    for (int i = 0; i < 4096; i++) {
        tmp[i] = utils::random<double>();
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_phi; i++) {
        for (int igk_loc = Hk__.kp().gkvec().skip_g0(); igk_loc < Hk__.kp().num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = Hk__.kp().idxgk(igk_loc);
            phi.pw_coeffs(0).prime(igk_loc, i) += tmp[igk & 0xFFF] * 1e-5;
        }
    }

    if (ctx_.num_mag_dims() == 3) {
        /* make pure spinor up- and dn- wave functions */
        phi.copy_from(device_t::CPU, num_phi, phi, 0, 0, 1, num_phi);
    }
    PROFILE_STOP("sirius::Band::initialize_subspace|kp|wf");

    /* allocate wave-functions */
    Wave_functions hphi(mp, Hk__.kp().gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);
    Wave_functions ophi(mp, Hk__.kp().gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);
    /* temporary wave-functions required as a storage during orthogonalization */
    Wave_functions wf_tmp(mp, Hk__.kp().gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);

    int bs = ctx_.cyclic_block_size();

    auto& gen_solver = ctx_.gen_evp_solver();

    dmatrix<T> hmlt(mp, num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> ovlp(mp, num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> evec(mp, num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs);

    std::vector<double> eval(num_bands);

    ctx_.print_memory_usage(__FILE__, __LINE__);

    if (is_device_memory(ctx_.preferred_memory_t())) {
        auto& mpd = ctx_.mem_pool(memory_t::device);

        for (int ispn = 0; ispn < num_sc; ispn++) {
            phi.pw_coeffs(ispn).allocate(mpd);
            phi.pw_coeffs(ispn).copy_to(memory_t::device, 0, num_phi_tot);
        }

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            Hk__.kp().spinor_wave_functions().pw_coeffs(ispn).allocate(mpd);
        }

        for (int ispn = 0; ispn < num_sc; ispn++) {
            hphi.pw_coeffs(ispn).allocate(mpd);
            ophi.pw_coeffs(ispn).allocate(mpd);
            wf_tmp.pw_coeffs(ispn).allocate(mpd);
        }
        evec.allocate(mpd);
        hmlt.allocate(mpd);
        ovlp.allocate(mpd);
    }

    ctx_.print_memory_usage(__FILE__, __LINE__);

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < num_sc; ispn++) {
            auto cs = phi.checksum_pw(get_device_t(ctx_.preferred_memory_t()), ispn, 0, num_phi_tot);
            if (Hk__.kp().comm().rank() == 0) {
                std::stringstream s;
                s << "initial_phi" << ispn;
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    Hk__.kp().copy_hubbard_orbitals_on_device();

    for (int ispn_step = 0; ispn_step < ctx_.num_spin_dims(); ispn_step++) {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        Hk__.apply_h_s<T>(spin_range((ctx_.num_mag_dims() == 3) ? 2 : ispn_step), 0, num_phi_tot, phi, &hphi, &ophi);

        /* do some checks */
        if (ctx_.control().verification_ >= 1) {

            set_subspace_mtrx<T>(0, num_phi_tot, phi, ophi, ovlp);
            if (ctx_.control().verification_ >= 2 && ctx_.control().verbosity_ >= 2) {
                ovlp.serialize("overlap", num_phi_tot);
            }

            double max_diff = check_hermitian(ovlp, num_phi_tot);
            if (max_diff > 1e-12) {
                std::stringstream s;
                s << "overlap matrix is not hermitian, max_err = " << max_diff;
                TERMINATE(s);
            }
            std::vector<double> eo(num_phi_tot);
            auto& std_solver = ctx_.std_evp_solver();
            if (std_solver.solve(num_phi_tot, num_phi_tot, ovlp, eo.data(), evec)) {
                std::stringstream s;
                s << "error in diagonalization";
                TERMINATE(s);
            }
            Hk__.kp().message(1, __function_name__, "minimum eigen-value of the overlap matrix: %18.12f\n", eo[0]);
            if (eo[0] < 0) {
                TERMINATE("overlap matrix is not positively defined");
            }
        }

        /* setup eigen-value problem */
        set_subspace_mtrx<T>(0, num_phi_tot, phi, hphi, hmlt);
        set_subspace_mtrx<T>(0, num_phi_tot, phi, ophi, ovlp);

        if (ctx_.control().verification_ >= 2 && ctx_.control().verbosity_ >= 2) {
            hmlt.serialize("hmlt", num_phi_tot);
            ovlp.serialize("ovlp", num_phi_tot);
        }

        /* solve generalized eigen-value problem with the size N and get lowest num_bands eigen-vectors */
        if (gen_solver.solve(num_phi_tot, num_bands, hmlt, ovlp, eval.data(), evec)) {
            std::stringstream s;
            s << "[sirius::Band::initialize_subspace] error in diagonalziation";
            TERMINATE(s);
        }

        if (ctx_.control().print_checksum_) {
            auto cs = evec.checksum();
            evec.blacs_grid().comm().allreduce(&cs, 1);
            double cs1{0};
            for (int i = 0; i < num_bands; i++) {
                cs1 += eval[i];
            }
            if (Hk__.kp().comm().rank() == 0) {
                utils::print_checksum("evec", cs);
                utils::print_checksum("eval", cs1);
            }
        }
        for (int i = 0; i < num_bands; i++) {
            Hk__.kp().message(3, __function_name__, "eval[%i]=%20.16f\n", i, eval[i]);
        }

        /* compute wave-functions */
        /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
        transform<T>(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), (ctx_.num_mag_dims() == 3) ? 2 : ispn_step,
                     {&phi}, 0, num_phi_tot, evec, 0, 0, {&Hk__.kp().spinor_wave_functions()}, 0, num_bands);

        for (int j = 0; j < num_bands; j++) {
            Hk__.kp().band_energy(j, ispn_step, eval[j]);
        }
    }

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            auto cs = Hk__.kp().spinor_wave_functions().checksum_pw(get_device_t(ctx_.preferred_memory_t()), ispn, 0, num_bands);
            std::stringstream s;
            s << "initial_spinor_wave_functions_" << ispn;
            if (Hk__.kp().comm().rank() == 0) {
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    if (is_device_memory(ctx_.preferred_memory_t())) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            Hk__.kp().spinor_wave_functions().pw_coeffs(ispn).copy_to(memory_t::host, 0, num_bands);
            Hk__.kp().spinor_wave_functions().pw_coeffs(ispn).deallocate(memory_t::device);
        }
    }

    Hk__.kp().release_hubbard_orbitals_on_device();

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            auto cs = Hk__.kp().spinor_wave_functions().checksum_pw(device_t::CPU, ispn, 0, num_bands);
            std::stringstream s;
            s << "initial_spinor_wave_functions_" << ispn;
            if (Hk__.kp().comm().rank() == 0) {
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    /* check residuals */
    if (ctx_.control().verification_ >= 2) {
        check_residuals<T>(Hk__);
        check_wave_functions<T>(Hk__);
    }

    ctx_.print_memory_usage(__FILE__, __LINE__);
}

template <typename T>
void Band::check_residuals(Hamiltonian_k& Hk__) const
{
    auto& kp = Hk__.kp();
    kp.message(1, __function_name__, "%s", "checking residuals\n");

    const bool nc_mag = (ctx_.num_mag_dims() == 3);
    const int num_sc = nc_mag ? 2 : 1;

    auto& psi = kp.spinor_wave_functions();
    Wave_functions hpsi(kp.gkvec_partition(), ctx_.num_bands(), ctx_.preferred_memory_t(), num_sc);
    Wave_functions spsi(kp.gkvec_partition(), ctx_.num_bands(), ctx_.preferred_memory_t(), num_sc);
    Wave_functions res(kp.gkvec_partition(), ctx_.num_bands(), ctx_.preferred_memory_t(), num_sc);

    if (is_device_memory(ctx_.preferred_memory_t())) {
        auto& mpd = ctx_.mem_pool(memory_t::device);
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            psi.pw_coeffs(ispn).allocate(mpd);
            psi.pw_coeffs(ispn).copy_to(memory_t::device, 0, ctx_.num_bands());
        }
        for (int i = 0; i < num_sc; i++) {
            res.pw_coeffs(i).allocate(mpd);
            hpsi.pw_coeffs(i).allocate(mpd);
            spsi.pw_coeffs(i).allocate(mpd);
        }

        kp.copy_hubbard_orbitals_on_device();
    }
    /* compute residuals */
    for (int ispin_step = 0; ispin_step < ctx_.num_spin_dims(); ispin_step++) {
        /* apply Hamiltonian and S operators to the wave-functions */
        Hk__.apply_h_s<T>(spin_range(nc_mag ? 2 : ispin_step), 0, ctx_.num_bands(), psi, &hpsi, &spsi);

        for (int ispn = 0; ispn < num_sc; ispn++) {
            if (is_device_memory(ctx_.preferred_memory_t())) {
                hpsi.copy_to(spin_range(ispn), memory_t::host, 0, ctx_.num_bands());
                spsi.copy_to(spin_range(ispn), memory_t::host, 0, ctx_.num_bands());
            }
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < ctx_.num_bands(); j++) {
                for (int ig = 0; ig < kp.num_gkvec_loc(); ig++) {
                    res.pw_coeffs(ispn).prime(ig, j) = hpsi.pw_coeffs(ispn).prime(ig, j) -
                                                       spsi.pw_coeffs(ispn).prime(ig, j) *
                                                       kp.band_energy(j, ispin_step);
                }
            }
        }
        /* get the norm */
        auto l2norm = res.l2norm(device_t::CPU, nc_mag ? spin_range(2) : spin_range(0), ctx_.num_bands());

        for (int j = 0; j < ctx_.num_bands(); j++) {
            Hk__.kp().message(1, __function_name__, "band: %3i, residual l2norm: %18.12f\n", j, l2norm[j]);
        }
    }
    if (is_device_memory(ctx_.preferred_memory_t())) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            psi.pw_coeffs(ispn).deallocate(memory_t::device);
        }
    }

    kp.release_hubbard_orbitals_on_device();
}

/// Check wave-functions for orthonormalization.
template <typename T>
void Band::check_wave_functions(Hamiltonian_k& Hk__) const
{
    auto& kp = Hk__.kp();
    kp.message(1, __function_name__, "%s", "checking wave-functions\n");

    if (!ctx_.full_potential()) {

        dmatrix<T> ovlp(ctx_.num_bands(), ctx_.num_bands(), ctx_.blacs_grid(), ctx_.cyclic_block_size(), ctx_.cyclic_block_size());

        const bool nc_mag = (ctx_.num_mag_dims() == 3);
        const int num_sc = nc_mag ? 2 : 1;

        auto& psi = kp.spinor_wave_functions();
        Wave_functions spsi(kp.gkvec_partition(), ctx_.num_bands(), ctx_.preferred_memory_t(), num_sc);

        if (is_device_memory(ctx_.preferred_memory_t())) {
            auto& mpd = ctx_.mem_pool(memory_t::device);
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                psi.pw_coeffs(ispn).allocate(mpd);
                psi.pw_coeffs(ispn).copy_to(memory_t::device, 0, ctx_.num_bands());
            }
            for (int i = 0; i < num_sc; i++) {
                spsi.pw_coeffs(i).allocate(mpd);
            }
            ovlp.allocate(memory_t::device);
        }

        Hk__.kp().copy_hubbard_orbitals_on_device();

        /* compute residuals */
        for (int ispin_step = 0; ispin_step < ctx_.num_spin_dims(); ispin_step++) {
            /* apply Hamiltonian and S operators to the wave-functions */
            Hk__.apply_h_s<T>(spin_range(nc_mag ? 2 : ispin_step), 0, ctx_.num_bands(), psi, nullptr, &spsi);
            inner(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), nc_mag ? 2 : ispin_step, psi, 0, ctx_.num_bands(),
                  spsi, 0, ctx_.num_bands(), ovlp, 0, 0);

            double diff = check_identity(ovlp, ctx_.num_bands());

            if (diff > 1e-12) {
                kp.message(1, __function_name__, "overlap matrix is not identity, maximum error : %20.12f\n", diff);
            } else {
                kp.message(1, __function_name__, "%s", "OK! Wave functions are orthonormal.\n");
            }
        }
        if (is_device_memory(ctx_.preferred_memory_t())) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                psi.pw_coeffs(ispn).deallocate(memory_t::device);
            }
        }

        Hk__.kp().release_hubbard_orbitals_on_device();
    }
}

template
void
Band::set_subspace_mtrx<double>(int N__, int n__, Wave_functions& phi__, Wave_functions& op_phi__,
                                dmatrix<double>& mtrx__, dmatrix<double>* mtrx_old__) const;

template
void
Band::set_subspace_mtrx<double_complex>(int N__, int n__, Wave_functions& phi__, Wave_functions& op_phi__,
                                        dmatrix<double_complex>& mtrx__, dmatrix<double_complex>* mtrx_old__) const;

}
