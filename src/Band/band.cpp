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
 *   \brief Contains implementation of sirius::Band class.
 */

#include "Band.hpp"

namespace sirius {

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
            tranc(n__, N__, mtrx__, 0, N__, mtrx__, N__, 0);
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
Band::initialize_subspace(K_point_set& kset__, Hamiltonian& H__) const
{
    PROFILE("sirius::Band::initialize_subspace");

    int N{0};

    if (ctx_.iterative_solver_input().init_subspace_ == "lcao") {
        /* get the total number of atomic-centered orbitals */
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            int n{0};
            for (int i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
                n += (2 * std::abs(atom_type.ps_atomic_wf(i).first) + 1);
            }
            N += atom_type.num_atoms() * n;
        }

        if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 2) {
            printf("number of atomic orbitals: %i\n", N);
        }
    }

    H__.prepare();
    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__[ik];
        if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
            initialize_subspace<double>(kp, H__, N);
        } else {
            initialize_subspace<double_complex>(kp, H__, N);
        }
    }
    H__.dismiss();

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
void Band::initialize_subspace(K_point* kp__, Hamiltonian& H__, int num_ao__) const
{
    PROFILE("sirius::Band::initialize_subspace|kp");

    if (ctx_.control().verification_ >= 1) {
        auto eval = diag_S_davidson<T>(*kp__, H__);
        if (eval[0] <= 0) {
            std::stringstream s;
            s << "S-operator matrix is not positive definite\n"
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
    Wave_functions phi(mp, kp__->gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);
    for (int ispn = 0; ispn < num_sc; ispn++) {
        phi.pw_coeffs(ispn).prime().zero();
    }

    utils::timer t1("sirius::Band::initialize_subspace|kp|wf");
    /* get proper lmax */
    int lmax{0};

    /* generate the initial atomic wavefunctions */

    int offset = 0;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom_type = unit_cell_.atom(ia).type();
        lmax = std::max(lmax, atom_type.lmax_ps_atomic_wf());
        // generate the atomic wave functions
        kp__->generate_atomic_wave_functions(atom_type.indexb_wfc(), ia, offset, false, phi);
        offset += atom_type.indexb_wfc().size();
    }

    lmax = std::max(lmax, unit_cell_.lmax());

    /* fill remaining wave-functions with pseudo-random guess */
    assert(kp__->num_gkvec() > num_phi + 10);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_phi - num_ao__; i++) {
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->idxgk(igk_loc);
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
        for (int igk_loc = kp__->gkvec().skip_g0(); igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->idxgk(igk_loc);
            phi.pw_coeffs(0).prime(igk_loc, i) += tmp[igk & 0xFFF] * 1e-5;
        }
    }

    if (ctx_.num_mag_dims() == 3) {
        /* make pure spinor up- and dn- wave functions */
        phi.copy_from(device_t::CPU, num_phi, phi, 0, 0, 1, num_phi);
    }
    t1.stop();

    ctx_.fft_coarse().prepare(kp__->gkvec_partition());
    H__.local_op().prepare(kp__->gkvec_partition());

    /* allocate wave-functions */
    Wave_functions hphi(mp, kp__->gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);
    Wave_functions ophi(mp, kp__->gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);
    /* temporary wave-functions required as a storage during orthogonalization */
    Wave_functions wf_tmp(mp, kp__->gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);

    int bs = ctx_.cyclic_block_size();

    auto& gen_solver = ctx_.gen_evp_solver();

    dmatrix<T> hmlt(mp, num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> ovlp(mp, num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> evec(mp, num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs);

    std::vector<double> eval(num_bands);

    ctx_.print_memory_usage(__FILE__, __LINE__);

    kp__->beta_projectors().prepare();

    if (is_device_memory(ctx_.preferred_memory_t())) {
        auto& mpd = ctx_.mem_pool(memory_t::device);

        for (int ispn = 0; ispn < num_sc; ispn++) {
            phi.pw_coeffs(ispn).allocate(mpd);
            phi.pw_coeffs(ispn).copy_to(memory_t::device, 0, num_phi_tot);
        }

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__->spinor_wave_functions().pw_coeffs(ispn).allocate(mpd);
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
            if (kp__->comm().rank() == 0) {
                std::stringstream s;
                s << "initial_phi" << ispn;
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    for (int ispn_step = 0; ispn_step < ctx_.num_spin_dims(); ispn_step++) {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        H__.apply_h_s<T>(kp__, (ctx_.num_mag_dims() == 3) ? 2 : ispn_step, 0, num_phi_tot, phi, &hphi, &ophi);

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
            if (kp__->comm().rank() == 0) {
                printf("[verification] minimum eigen-value of the overlap matrix: %18.12f\n", eo[0]);
            }
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
            s << "error in diagonalziation";
            TERMINATE(s);
        }

        if (ctx_.control().print_checksum_) {
            auto cs = evec.checksum();
            evec.blacs_grid().comm().allreduce(&cs, 1);
            double cs1{0};
            for (int i = 0; i < num_bands; i++) {
                cs1 += eval[i];
            }
            if (kp__->comm().rank() == 0) {
                utils::print_checksum("evec", cs);
                utils::print_checksum("eval", cs1);
            }
        }

        if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
            for (int i = 0; i < num_bands; i++) {
                printf("eval[%i]=%20.16f\n", i, eval[i]);
            }
        }

        /* compute wave-functions */
        /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
        transform<T>(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), (ctx_.num_mag_dims() == 3) ? 2 : ispn_step,
                     {&phi}, 0, num_phi_tot, evec, 0, 0, {&kp__->spinor_wave_functions()}, 0, num_bands);

        for (int j = 0; j < num_bands; j++) {
            kp__->band_energy(j, ispn_step, eval[j]);
        }
    }

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            auto cs = kp__->spinor_wave_functions().checksum_pw(get_device_t(ctx_.preferred_memory_t()), ispn, 0, num_bands);
            std::stringstream s;
            s << "initial_spinor_wave_functions_" << ispn;
            if (kp__->comm().rank() == 0) {
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    if (is_device_memory(ctx_.preferred_memory_t())) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__->spinor_wave_functions().pw_coeffs(ispn).copy_to(memory_t::host, 0, num_bands);
            kp__->spinor_wave_functions().pw_coeffs(ispn).deallocate(memory_t::device);
        }
    }

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            auto cs = kp__->spinor_wave_functions().checksum_pw(device_t::CPU, ispn, 0, num_bands);
            std::stringstream s;
            s << "initial_spinor_wave_functions_" << ispn;
            if (kp__->comm().rank() == 0) {
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    /* check residuals */
    if (ctx_.control().verification_ >= 2) {
        check_residuals<T>(*kp__, H__);
        check_wave_functions<T>(*kp__, H__);
    }

    kp__->beta_projectors().dismiss();
    ctx_.fft_coarse().dismiss();

    ctx_.print_memory_usage(__FILE__, __LINE__);
}

/// Compute r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
static void compute_res(device_t            pu__,
                        int                 ispn__,
                        int                 num_bands__,
                        mdarray<double, 1>& eval__,
                        Wave_functions&     hpsi__,
                        Wave_functions&     opsi__,
                        Wave_functions&     res__)
{
    auto spins = get_spins(ispn__);

    for (int ispn: spins) {
        switch (pu__) {
            case device_t::CPU: {
                /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
                #pragma omp parallel for
                for (int i = 0; i < num_bands__; i++) {
                    for (int ig = 0; ig < res__.pw_coeffs(ispn).num_rows_loc(); ig++) {
                        res__.pw_coeffs(ispn).prime(ig, i) = hpsi__.pw_coeffs(ispn).prime(ig, i) -
                            eval__[i] * opsi__.pw_coeffs(ispn).prime(ig, i);
                    }
                    if (res__.has_mt()) {
                        for (int j = 0; j < res__.mt_coeffs(ispn).num_rows_loc(); j++) {
                            res__.mt_coeffs(ispn).prime(j, i) = hpsi__.mt_coeffs(ispn).prime(j, i) -
                                eval__[i] * opsi__.mt_coeffs(ispn).prime(j, i);
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
#if defined(__GPU)
                compute_residuals_gpu(hpsi__.pw_coeffs(ispn).prime().at(memory_t::device),
                                      opsi__.pw_coeffs(ispn).prime().at(memory_t::device),
                                      res__.pw_coeffs(ispn).prime().at(memory_t::device),
                                      res__.pw_coeffs(ispn).num_rows_loc(),
                                      num_bands__,
                                      eval__.at(memory_t::device));
                if (res__.has_mt()) {
                    compute_residuals_gpu(hpsi__.mt_coeffs(ispn).prime().at(memory_t::device),
                                          opsi__.mt_coeffs(ispn).prime().at(memory_t::device),
                                          res__.mt_coeffs(ispn).prime().at(memory_t::device),
                                          res__.mt_coeffs(ispn).num_rows_loc(),
                                          num_bands__,
                                          eval__.at(memory_t::device));
                }
#endif
            }
        }
    }
}

/// Apply preconditioner to the residuals.
static void apply_p(device_t            pu__,
                    int                 ispn__,
                    int                 num_bands__,
                    Wave_functions&     res__,
                    mdarray<double, 2>& h_diag__,
                    mdarray<double, 1>& o_diag__,
                    mdarray<double, 1>& eval__)
{
    int s0 = (ispn__ == 2) ? 0 : ispn__;
    int s1 = (ispn__ == 2) ? 1 : ispn__;

    for (int ispn = s0; ispn <= s1; ispn++) {
        switch (pu__) {
            case device_t::CPU: {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < num_bands__; i++) {
                    for (int ig = 0; ig < res__.pw_coeffs(ispn).num_rows_loc(); ig++) {
                        double p = h_diag__(ig, ispn) - o_diag__[ig] * eval__[i];
                        p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                        res__.pw_coeffs(ispn).prime(ig, i) /= p;
                    }
                    if (res__.has_mt()) {
                        for (int j = 0; j < res__.mt_coeffs(ispn).num_rows_loc(); j++) {
                            double p = h_diag__(res__.pw_coeffs(ispn).num_rows_loc() + j, ispn) -
                                       o_diag__[res__.pw_coeffs(ispn).num_rows_loc() + j] * eval__[i];
                            p = 0.5 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                            res__.mt_coeffs(ispn).prime(j, i) /= p;
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
#if defined(__GPU)
                apply_preconditioner_gpu(res__.pw_coeffs(ispn).prime().at(memory_t::device),
                                         res__.pw_coeffs(ispn).num_rows_loc(),
                                         num_bands__,
                                         eval__.at(memory_t::device),
                                         h_diag__.at(memory_t::device, 0, ispn),
                                         o_diag__.at(memory_t::device));
                if (res__.has_mt()) {
                    apply_preconditioner_gpu(res__.mt_coeffs(ispn).prime().at(memory_t::device),
                                             res__.mt_coeffs(ispn).num_rows_loc(),
                                             num_bands__,
                                             eval__.at(memory_t::device),
                                             h_diag__.at(memory_t::device, res__.pw_coeffs(ispn).num_rows_loc(), ispn),
                                             o_diag__.at(memory_t::device, res__.pw_coeffs(ispn).num_rows_loc()));
                }
                break;
#endif
            }
        }
    }
}

/// Normalize residuals.
/** This not strictly necessary as the wave-function orthoronormalization can take care of this.
 *  However, normalization of residuals is harmless and gives a better numerical stability. */
static void normalize_res(device_t            pu__,
                          int                 ispn__,
                          int                 num_bands__,
                          Wave_functions&     res__,
                          mdarray<double, 1>& p_norm__)
{
    auto spins = get_spins(ispn__);

    for (int ispn: spins) {
        switch (pu__) {
            case device_t::CPU: {
            #pragma omp parallel for schedule(static)
                for (int i = 0; i < num_bands__; i++) {
                    for (int ig = 0; ig < res__.pw_coeffs(ispn).num_rows_loc(); ig++) {
                        res__.pw_coeffs(ispn).prime(ig, i) *= p_norm__[i];
                    }
                    if (res__.has_mt()) {
                        for (int j = 0; j < res__.mt_coeffs(ispn).num_rows_loc(); j++) {
                            res__.mt_coeffs(ispn).prime(j, i) *= p_norm__[i];
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
                #ifdef __GPU
                scale_matrix_columns_gpu(res__.pw_coeffs(ispn).num_rows_loc(), num_bands__,
                                         (acc_complex_double_t*)res__.pw_coeffs(ispn).prime().at(memory_t::device),
                                         p_norm__.at(memory_t::device));

                if (res__.has_mt()) {
                    scale_matrix_columns_gpu(res__.mt_coeffs(ispn).num_rows_loc(),
                                             num_bands__,
                                             (acc_complex_double_t *)res__.mt_coeffs(ispn).prime().at(memory_t::device),
                                             p_norm__.at(memory_t::device));
                }
                #endif
                break;
            }
        }
    }
}

mdarray<double, 1>
Band::residuals_aux(K_point*             kp__,
                    int                  ispn__,
                    int                  num_bands__,
                    std::vector<double>& eval__,
                    Wave_functions&      hpsi__,
                    Wave_functions&      opsi__,
                    Wave_functions&      res__,
                    mdarray<double, 2>&  h_diag__,
                    mdarray<double, 1>&  o_diag__) const
{
    PROFILE("sirius::Band::residuals_aux");

    assert(num_bands__ != 0);

    auto pu = get_device_t(ctx_.preferred_memory_t());

    mdarray<double, 1> eval(eval__.data(), num_bands__, "residuals_aux::eval");
    if (pu == device_t::GPU) {
        eval.allocate(memory_t::device).copy_to(memory_t::device);
    }

    /* compute residuals */
    compute_res(pu, ispn__, num_bands__, eval, hpsi__, opsi__, res__);

    /* compute norm */
    auto res_norm = res__.l2norm(pu, spin_range(ispn__), num_bands__);

    apply_p(pu, ispn__, num_bands__, res__, h_diag__, o_diag__, eval);

    auto p_norm = res__.l2norm(pu, spin_range(ispn__), num_bands__);
    for (int i = 0; i < num_bands__; i++) {
        p_norm[i] = 1.0 / p_norm[i];
    }
    if (pu == device_t::GPU) {
        p_norm.copy_to(memory_t::device);
    }

    /* normalize preconditioned residuals */
    normalize_res(pu, ispn__, num_bands__, res__, p_norm);

    if (ctx_.control().verbosity_ >= 5) {
        auto n_norm = res__.l2norm(pu, spin_range(ispn__), num_bands__);
        if (kp__->comm().rank() == 0) {
            for (int i = 0; i < num_bands__; i++) {
                printf("norms of residual %3i: %18.14f %24.14f %18.14f", i, res_norm[i], p_norm[i], n_norm[i]);
                if (res_norm[i] > ctx_.iterative_solver_input().residual_tolerance_) {
                    printf(" +");
                }
                printf("\n");
            }
        }
    }

   return res_norm;
}

template <typename T>
int Band::residuals(K_point*             kp__,
                    int                  ispn__,
                    int                  N__,
                    int                  num_bands__,
                    std::vector<double>& eval__,
                    std::vector<double>& eval_old__,
                    dmatrix<T>&          evec__,
                    Wave_functions&      hphi__,
                    Wave_functions&      ophi__,
                    Wave_functions&      hpsi__,
                    Wave_functions&      opsi__,
                    Wave_functions&      res__,
                    mdarray<double, 2>&  h_diag__,
                    mdarray<double, 1>&  o_diag__,
                    double               eval_tolerance__,
                    double               norm_tolerance__) const
{
    PROFILE("sirius::Band::residuals");

    assert(N__ != 0);

    auto& itso = ctx_.iterative_solver_input();
    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    auto spins = get_spins(ispn__);

    int n{0};
    if (converge_by_energy) {

        /* main trick here: first estimate energy difference, and only then compute unconverged residuals */
        auto get_ev_idx = [&](double tol__)
        {
            std::vector<int> ev_idx;
            int s = ispn__ == 2 ? 0 : ispn__;
            for (int i = 0; i < num_bands__; i++) {
                double o1 = std::abs(kp__->band_occupancy(i, s) / ctx_.max_occupancy());
                double o2 = std::abs(1 - o1);

                double tol = o1 * tol__ + o2 * (tol__ + itso.empty_states_tolerance_);
                if (std::abs(eval__[i] - eval_old__[i]) > tol) {
                    ev_idx.push_back(i);
                }
            }
            return ev_idx;
        };

        auto ev_idx = get_ev_idx(eval_tolerance__);

        n = static_cast<int>(ev_idx.size());

        if (n) {
            std::vector<double> eval_tmp(n);

            int bs = ctx_.cyclic_block_size();
            dmatrix<T> evec_tmp(N__, n, ctx_.blacs_grid(), bs, bs);
            int num_rows_local = evec_tmp.num_rows_local();
            for (int j = 0; j < n; j++) {
                eval_tmp[j] = eval__[ev_idx[j]];
                if (ctx_.blacs_grid().comm().size() == 1) {
                    /* do a local copy */
                    std::copy(&evec__(0, ev_idx[j]), &evec__(0, ev_idx[j]) + num_rows_local, &evec_tmp(0, j));
                } else {
                    auto pos_src  = evec__.spl_col().location(ev_idx[j]);
                    auto pos_dest = evec_tmp.spl_col().location(j);
                    /* do MPI send / recieve */
                    if (pos_src.rank == kp__->comm_col().rank()) {
                        kp__->comm_col().isend(&evec__(0, pos_src.local_index), num_rows_local, pos_dest.rank, ev_idx[j]);
                    }
                    if (pos_dest.rank == kp__->comm_col().rank()) {
                       kp__->comm_col().recv(&evec_tmp(0, pos_dest.local_index), num_rows_local, pos_src.rank, ev_idx[j]);
                    }
                }
            }
            if (ctx_.processing_unit() == device_t::GPU && evec__.blacs_grid().comm().size() == 1) {
                evec_tmp.allocate(memory_t::device);
            }
            /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} and O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
            transform<T>(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), ispn__, {&hphi__, &ophi__}, 0, N__,
                         evec_tmp, 0, 0, {&hpsi__, &opsi__}, 0, n);

            /* print checksums */
            if (ctx_.control().print_checksum_ && n != 0) {
                for (int ispn: spins) {
                    auto cs1 = hpsi__.checksum(get_device_t(hpsi__.preferred_memory_t()), ispn, 0, n);
                    auto cs2 = opsi__.checksum(get_device_t(opsi__.preferred_memory_t()), ispn, 0, n);
                    if (kp__->comm().rank() == 0) {
                        std::stringstream s;
                        s.str("");
                        s << "hpsi_" << ispn;
                        utils::print_checksum(s.str(), cs1);
                        s.str("");
                        s << "opsi_" << ispn;
                        utils::print_checksum(s.str(), cs2);
                    }
                }
            }

            auto res_norm = residuals_aux(kp__, ispn__, n, eval_tmp, hpsi__, opsi__, res__, h_diag__, o_diag__);

            int nmax = n;
            n = 0;
            for (int i = 0; i < nmax; i++) {
                /* take the residual if it's norm is above the threshold */
                if (res_norm[i] > norm_tolerance__) {
                    /* shift unconverged residuals to the beginning of array */
                    if (n != i) {
                        for (int ispn: spins) {
                            res__.copy_from(res__, 1, ispn, i, ispn, n);
                        }
                    }
                    n++;
                }
            }
            if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
                printf("initial and final number of residuals : %i %i\n", nmax, n);
            }
        }
    } else { /* compute all residuals first */
        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} and O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        transform<T>(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), ispn__, {&hphi__, &ophi__}, 0, N__,
                     evec__, 0, 0, {&hpsi__, &opsi__}, 0, num_bands__);

        auto res_norm = residuals_aux(kp__, ispn__, num_bands__, eval__, hpsi__, opsi__, res__, h_diag__, o_diag__);

        for (int i = 0; i < num_bands__; i++) {
            /* take the residual if its norm is above the threshold */
            if (res_norm[i] > norm_tolerance__) {
                /* shift unconverged residuals to the beginning of array */
                if (n != i) {
                    for (int ispn: spins) {
                        res__.copy_from(res__, 1, ispn, i, ispn, n);
                    }
                }
                n++;
            }
        }
        if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
            printf("number of residuals : %i\n", n);
        }
    }

    /* prevent numerical noise */
    /* this only happens for real wave-functions (Gamma-point case), non-magnetic or collinear magnetic */
    if (std::is_same<T, double>::value && kp__->comm().rank() == 0 && n != 0) {
        assert(ispn__ == 0 || ispn__ == 1);
        if (is_device_memory(res__.preferred_memory_t())) {
#if defined(__GPU)
            make_real_g0_gpu(res__.pw_coeffs(ispn__).prime().at(memory_t::device), res__.pw_coeffs(ispn__).prime().ld(), n);
#endif
        } else {
            for (int i = 0; i < n; i++) {
                res__.pw_coeffs(ispn__).prime(0, i) = res__.pw_coeffs(ispn__).prime(0, i).real();
            }
        }
    }

    /* print checksums */
    if (ctx_.control().print_checksum_ && n != 0) {
        for (int ispn: spins) {
            auto cs = res__.checksum(get_device_t(res__.preferred_memory_t()), ispn, 0, n);
            if (kp__->comm().rank() == 0) {
                std::stringstream s;
                s << "res_" << ispn;
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    return n;
}

template <typename T>
void Band::check_residuals(K_point& kp__, Hamiltonian& H__) const
{
    if (kp__.comm().rank() == 0) {
        printf("checking residuals\n");
    }

    const bool nc_mag = (ctx_.num_mag_dims() == 3);
    const int num_sc = nc_mag ? 2 : 1;

    auto& psi = kp__.spinor_wave_functions();
    Wave_functions hpsi(kp__.gkvec_partition(), ctx_.num_bands(), ctx_.preferred_memory_t(), num_sc);
    Wave_functions spsi(kp__.gkvec_partition(), ctx_.num_bands(), ctx_.preferred_memory_t(), num_sc);
    Wave_functions res(kp__.gkvec_partition(), ctx_.num_bands(), ctx_.preferred_memory_t(), num_sc);

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
    }
    kp__.beta_projectors().prepare();
    /* compute residuals */
    for (int ispin_step = 0; ispin_step < ctx_.num_spin_dims(); ispin_step++) {
        if (nc_mag) {
            /* apply Hamiltonian and S operators to the wave-functions */
            H__.apply_h_s<T>(&kp__, 2, 0, ctx_.num_bands(), psi, &hpsi, &spsi);
        } else {
            /* apply Hamiltonian and S operators to the wave-functions */
            H__.apply_h_s<T>(&kp__, ispin_step, 0, ctx_.num_bands(), psi, &hpsi, &spsi);
        }

        for (int ispn = 0; ispn < num_sc; ispn++) {
            if (is_device_memory(ctx_.preferred_memory_t())) {
                hpsi.copy_to(spin_range(ispn), memory_t::host, 0, ctx_.num_bands());
                spsi.copy_to(spin_range(ispn), memory_t::host, 0, ctx_.num_bands());
            }
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < ctx_.num_bands(); j++) {
                for (int ig = 0; ig < kp__.num_gkvec_loc(); ig++) {
                    res.pw_coeffs(ispn).prime(ig, j) = hpsi.pw_coeffs(ispn).prime(ig, j) -
                                                       spsi.pw_coeffs(ispn).prime(ig, j) *
                                                       kp__.band_energy(j, ispin_step);
                }
            }
        }
        /* get the norm */
        auto l2norm = res.l2norm(device_t::CPU, nc_mag ? spin_range(2) : spin_range(0), ctx_.num_bands());

        if (kp__.comm().rank() == 0) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                printf("band: %3i, residual l2norm: %18.12f\n", j, l2norm[j]);
            }
        }
    }
    kp__.beta_projectors().dismiss();
    if (is_device_memory(ctx_.preferred_memory_t())) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            psi.pw_coeffs(ispn).deallocate(memory_t::device);
        }
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


template
int
Band::residuals<double>(K_point*             kp__,
                int                  ispn__,
                int                  N__,
                int                  num_bands__,
                std::vector<double>& eval__,
                std::vector<double>& eval_old__,
                dmatrix<double>&     evec__,
                Wave_functions&      hphi__,
                Wave_functions&      ophi__,
                Wave_functions&      hpsi__,
                Wave_functions&      opsi__,
                Wave_functions&      res__,
                mdarray<double, 2>&  h_diag__,
                mdarray<double, 1>&  o_diag__,
                double               eval_tolerance__,
                double               norm_tolerance__) const;

template
int
Band::residuals<double_complex>(K_point*             kp__,
                int                  ispn__,
                int                  N__,
                int                  num_bands__,
                std::vector<double>& eval__,
                std::vector<double>& eval_old__,
                dmatrix<double_complex>&     evec__,
                Wave_functions&      hphi__,
                Wave_functions&      ophi__,
                Wave_functions&      hpsi__,
                Wave_functions&      opsi__,
                Wave_functions&      res__,
                mdarray<double, 2>&  h_diag__,
                mdarray<double, 1>&  o_diag__,
                double               eval_tolerance__,
                double               norm_tolerance__) const;

}

