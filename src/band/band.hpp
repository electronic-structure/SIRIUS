// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file band.hpp
 *
 *   \brief Contains declaration and partial implementation of sirius::Band class.
 */

#ifndef __BAND_HPP__
#define __BAND_HPP__

#include "SDDK/memory.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "k_point/k_point_set.hpp"
#include "SDDK/wave_functions.hpp"

namespace sirius {

/// Setup and solve the eigen value problem.
class Band // TODO: Band class is lightweight and in principle can be converted to a namespace
{
  private:
    /// Simulation context.
    Simulation_context& ctx_;

    /// Alias for the unit cell.
    Unit_cell& unit_cell_;

    /// BLACS grid for distributed linear algebra operations.
    la::BLACS_grid const& blacs_grid_;

    /// Solve the first-variational (non-magnetic) problem with exact diagonalization.
    /** This is only used by the LAPW method. */
    void diag_full_potential_first_variation_exact(Hamiltonian_k<double>& Hk__) const;

    /// Solve the first-variational (non-magnetic) problem with iterative Davidson diagonalization.
    void diag_full_potential_first_variation_davidson(Hamiltonian_k<double>& Hk__, double itsol_tol__) const;

    /// Solve second-variational problem.
    void diag_full_potential_second_variation(Hamiltonian_k<double>& Hk__) const;

    /// Get singular components of the LAPW overlap matrix.
    /** Singular components are the eigen-vectors with a very small eigen-value. */
    void get_singular_components(Hamiltonian_k<double>& Hk__, double itsol_tol__) const;

    /// Exact (not iterative) diagonalization of the Hamiltonian.
    template <typename T, typename F>
    void diag_pseudo_potential_exact(int ispn__, Hamiltonian_k<T>& Hk__) const;

    /// Diagonalize S operator to check for the negative eigen-values.
    template <typename T>
    sddk::mdarray<real_type<T>, 1> diag_S_davidson(Hamiltonian_k<real_type<T>>& Hk__) const;

  public:
    /// Constructor
    Band(Simulation_context& ctx__);

    /** Compute \f$ O_{ii'} = \langle \phi_i | \hat O | \phi_{i'} \rangle \f$ operator matrix
     *  for the subspace spanned by the wave-functions \f$ \phi_i \f$. The matrix is always returned
     *  in the CPU pointer because most of the standard math libraries start from the CPU. */
    template <typename T, typename F>
    void set_subspace_mtrx(int N__, int n__, int num_locked__, wf::Wave_functions<T>& phi__,
                           wf::Wave_functions<T>& op_phi__, la::dmatrix<F>& mtrx__,
                           la::dmatrix<F>* mtrx_old__ = nullptr) const
    {
        PROFILE("sirius::Band::set_subspace_mtrx");

        RTE_ASSERT(n__ != 0);
        if (mtrx_old__ && mtrx_old__->size()) {
            RTE_ASSERT(&mtrx__.blacs_grid() == &mtrx_old__->blacs_grid());
        }

        /* copy old N - num_locked x N - num_locked distributed matrix */
        if (N__ > 0) {
            sddk::splindex<sddk::splindex_t::block_cyclic> spl_row(N__ - num_locked__,
                    mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
            sddk::splindex<sddk::splindex_t::block_cyclic> spl_col(N__ - num_locked__,
                    mtrx__.blacs_grid().num_ranks_col(), mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());

            if (mtrx_old__) {
                if (spl_row.local_size()) {
                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < spl_col.local_size(); i++) {
                        std::copy(&(*mtrx_old__)(0, i), &(*mtrx_old__)(0, i) + spl_row.local_size(), &mtrx__(0, i));
                    }
                }
            }

            if (ctx_.print_checksum()) {
                auto cs = mtrx__.checksum(N__ - num_locked__, N__ - num_locked__);
                if (ctx_.comm_band().rank() == 0) {
                    utils::print_checksum("subspace_mtrx_old", cs, RTE_OUT(std::cout));
                }
            }
        }

        /*  [--- num_locked -- | ------ N - num_locked ---- | ---- n ----] */
        /*  [ ------------------- N ------------------------| ---- n ----] */

        auto mem = ctx_.processing_unit() == sddk::device_t::CPU ? sddk::memory_t::host : sddk::memory_t::device;
        /* <{phi,phi_new}|Op|phi_new> */
        inner(ctx_.spla_context(), mem, ctx_.num_mag_dims() == 3 ? wf::spin_range(0, 2) : wf::spin_range(0), phi__,
                wf::band_range(num_locked__, N__ + n__), op_phi__, wf::band_range(N__, N__ + n__),
                mtrx__, 0, N__ - num_locked__);

        /* restore lower part */
        if (N__ > 0) {
            if (mtrx__.blacs_grid().comm().size() == 1) {
                #pragma omp parallel for
                for (int i = 0; i < N__ - num_locked__; i++) {
                    for (int j = N__ - num_locked__; j < N__ + n__ - num_locked__; j++) {
                        mtrx__(j, i) = utils::conj(mtrx__(i, j));
                    }
                }
            } else {
                la::wrap(la::lib_t::scalapack)
                    .tranc(n__, N__ - num_locked__, mtrx__, 0, N__ - num_locked__, mtrx__, N__ - num_locked__, 0);
            }
        }

        if (ctx_.print_checksum()) {
            sddk::splindex<sddk::splindex_t::block_cyclic> spl_row(N__ + n__ - num_locked__,
                    mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
            sddk::splindex<sddk::splindex_t::block_cyclic> spl_col(N__ + n__ - num_locked__,
                    mtrx__.blacs_grid().num_ranks_col(), mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());
            auto cs = mtrx__.checksum(N__ + n__ - num_locked__, N__ + n__ - num_locked__);
            if (ctx_.comm_band().rank() == 0) {
                utils::print_checksum("subspace_mtrx", cs, RTE_OUT(std::cout));
            }
        }

        /* remove any numerical noise */
        mtrx__.make_real_diag(N__ + n__ - num_locked__);

        /* save new matrix */
        if (mtrx_old__) {
            sddk::splindex<sddk::splindex_t::block_cyclic> spl_row(N__ + n__ - num_locked__,
                    mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
            sddk::splindex<sddk::splindex_t::block_cyclic> spl_col(N__ + n__ - num_locked__,
                    mtrx__.blacs_grid().num_ranks_col(), mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());

            if (spl_row.local_size()) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < spl_col.local_size(); i++) {
                    std::copy(&mtrx__(0, i), &mtrx__(0, i) + spl_row.local_size(), &(*mtrx_old__)(0, i));
                }
            }
        }
    }

    /// Solve the band eigen-problem for pseudopotential case.
    template <typename T, typename F>
    int solve_pseudo_potential(Hamiltonian_k<T>& Hk__, double itsol_tol__, double empy_tol__) const;

    /// Solve the band eigen-problem for full-potential case.
    template <typename T>
    void solve_full_potential(Hamiltonian_k<T>& Hk__, double itsol_tol__) const;

    /// Check the residuals of wave-functions.
    template <typename T>
    void check_residuals(Hamiltonian_k<real_type<T>>& Hk__) const;

    /// Check wave-functions for orthonormalization.
    template <typename T>
    void check_wave_functions(Hamiltonian_k<real_type<T>>& Hk__) const;

    /// Solve \f$ \hat H \psi = E \psi \f$ and find eigen-states of the Hamiltonian.
    template <typename T, typename F>
    void solve(K_point_set& kset__, Hamiltonian0<T>& H0__, double itsol_tol__) const;

    /// Initialize the subspace for the entire k-point set.
    template <typename T>
    void initialize_subspace(K_point_set& kset__, Hamiltonian0<T>& H0__) const;
};

/// Initialize the wave-functions subspace at a given k-point.
/** If the number of atomic orbitals is smaller than the number of bands, the rest of the initial wave-functions
 *  are created from the random numbers. */
template <typename T, typename F>
inline void initialize_subspace(Hamiltonian_k<T>& Hk__, int num_ao__)
{
    PROFILE("sirius::Band::initialize_subspace|kp");

    //if (ctx_.cfg().control().verification() >= 2) {
    //    auto eval = diag_S_davidson<T>(Hk__);
    //    if (eval[0] <= 0) {
    //        std::stringstream s;
    //        s << "S-operator matrix is not positive definite\n"
    //          << "  lowest eigen-value: " << eval[0];
    //        WARNING(s);
    //    } else {
    //        ctx_.message(1, __function_name__, "S-matrix is OK! Minimum eigen-value: %18.12f\n", eval[0]);
    //    }
    //}
    //
    auto& ctx = Hk__.H0().ctx();

    auto pcs = env::print_checksum();

    /* number of non-zero spin components */
    const int num_sc = (ctx.num_mag_dims() == 3) ? 2 : 1;

    /* short notation for number of target wave-functions */
    int num_bands = ctx.num_bands();

    /* number of basis functions */
    int num_phi = std::max(num_ao__, num_bands / num_sc);

    int num_phi_tot = num_phi * num_sc;

    auto& mp = get_memory_pool(ctx.host_memory_t());

    print_memory_usage(ctx.out(), FILE_LINE);

    /* initial basis functions */
    wf::Wave_functions<T> phi(Hk__.kp().gkvec_sptr(), wf::num_mag_dims(ctx.num_mag_dims() == 3 ? 3 : 0),
            wf::num_bands(num_phi_tot), sddk::memory_t::host);

    for (int ispn = 0; ispn < num_sc; ispn++) {
        phi.zero(sddk::memory_t::host, wf::spin_index(ispn), wf::band_range(0, num_phi_tot));
    }

    /* generate the initial atomic wavefunctions */
    std::vector<int> atoms(ctx.unit_cell().num_atoms());
    std::iota(atoms.begin(), atoms.end(), 0);
    Hk__.kp().generate_atomic_wave_functions(atoms, [&](int iat){return &ctx.unit_cell().atom_type(iat).indexb_wfs();},
                                             ctx.ps_atomic_wf_ri(), phi);

    /* generate some random noise */
    std::vector<T> tmp(4096);
    utils::rnd(true);
    for (int i = 0; i < 4096; i++) {
        tmp[i] = 1e-5 * utils::random<T>();
    }
    PROFILE_START("sirius::Band::initialize_subspace|kp|wf");
    /* fill remaining wave-functions with pseudo-random guess */
    RTE_ASSERT(Hk__.kp().num_gkvec() > num_phi + 10);
    #pragma omp parallel
    {
        for (int i = 0; i < num_phi - num_ao__; i++) {
            #pragma omp for schedule(static) nowait
            for (int igk_loc = 0; igk_loc < Hk__.kp().num_gkvec_loc(); igk_loc++) {
                /* global index of G+k vector */
                int igk = Hk__.kp().gkvec().offset() + igk_loc; //Hk__.kp().idxgk(igk_loc);
                if (igk == i + 1) {
                    phi.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(num_ao__ + i)) = 1.0;
                }
                if (igk == i + 2) {
                    phi.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(num_ao__ + i)) = 0.5;
                }
                if (igk == i + 3) {
                    phi.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(num_ao__ + i)) = 0.25;
                }
            }
        }
        /* add random noise */
        for (int i = 0; i < num_phi; i++) {
            #pragma omp for schedule(static) nowait
            for (int igk_loc = Hk__.kp().gkvec().skip_g0(); igk_loc < Hk__.kp().num_gkvec_loc(); igk_loc++) {
                /* global index of G+k vector */
                int igk = Hk__.kp().gkvec().offset() + igk_loc;
                phi.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(i)) += tmp[igk & 0xFFF];
            }
        }
    }

    if (ctx.num_mag_dims() == 3) {
        /* make pure spinor up- and dn- wave functions */
        wf::copy(sddk::memory_t::host, phi, wf::spin_index(0), wf::band_range(0, num_phi), phi, wf::spin_index(1),
                wf::band_range(num_phi, num_phi_tot));
    }
    PROFILE_STOP("sirius::Band::initialize_subspace|kp|wf");

    /* allocate wave-functions */
    wf::Wave_functions<T> hphi(Hk__.kp().gkvec_sptr(), wf::num_mag_dims(ctx.num_mag_dims() == 3 ? 3 : 0),
            wf::num_bands(num_phi_tot), sddk::memory_t::host);
    wf::Wave_functions<T> ophi(Hk__.kp().gkvec_sptr(), wf::num_mag_dims(ctx.num_mag_dims() == 3 ? 3 : 0),
            wf::num_bands(num_phi_tot), sddk::memory_t::host);
    /* temporary wave-functions required as a storage during orthogonalization */
    wf::Wave_functions<T> wf_tmp(Hk__.kp().gkvec_sptr(), wf::num_mag_dims(ctx.num_mag_dims() == 3 ? 3 : 0),
            wf::num_bands(num_phi_tot), sddk::memory_t::host);

    int bs = ctx.cyclic_block_size();

    auto& gen_solver = ctx.gen_evp_solver();

    la::dmatrix<F> hmlt(num_phi_tot, num_phi_tot, ctx.blacs_grid(), bs, bs, mp);
    la::dmatrix<F> ovlp(num_phi_tot, num_phi_tot, ctx.blacs_grid(), bs, bs, mp);
    la::dmatrix<F> evec(num_phi_tot, num_phi_tot, ctx.blacs_grid(), bs, bs, mp);

    std::vector<real_type<F>> eval(num_bands);

    print_memory_usage(ctx.out(), FILE_LINE);

    auto mem = ctx.processing_unit() == sddk::device_t::CPU ? sddk::memory_t::host : sddk::memory_t::device;

    std::vector<wf::device_memory_guard> mg;
    mg.emplace_back(Hk__.kp().spinor_wave_functions().memory_guard(mem, wf::copy_to::host));
    mg.emplace_back(phi.memory_guard(mem, wf::copy_to::device));
    mg.emplace_back(hphi.memory_guard(mem));
    mg.emplace_back(ophi.memory_guard(mem));
    mg.emplace_back(wf_tmp.memory_guard(mem));

    if (is_device_memory(mem)) {
        auto& mpd = get_memory_pool(mem);
        evec.allocate(mpd);
        hmlt.allocate(mpd);
        ovlp.allocate(mpd);
    }

    print_memory_usage(ctx.out(), FILE_LINE);

    if (pcs) {
        for (int ispn = 0; ispn < num_sc; ispn++) {
            auto cs = phi.checksum(mem, wf::spin_index(ispn), wf::band_range(0, num_phi_tot));
            if (Hk__.kp().comm().rank() == 0) {
                std::stringstream s;
                s << "initial_phi" << ispn;
                utils::print_checksum(s.str(), cs, RTE_OUT(std::cout));
            }
        }
    }

    for (int ispn_step = 0; ispn_step < ctx.num_spinors(); ispn_step++) {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        Hk__.template apply_h_s<F>(ctx.num_mag_dims() == 3 ? wf::spin_range(0, 2) : wf::spin_range(ispn_step),
            wf::band_range(0, num_phi_tot), phi, &hphi, &ophi);

        /* do some checks */
    //    if (ctx_.cfg().control().verification() >= 1) {

    //        set_subspace_mtrx<T>(0, num_phi_tot, 0, phi, ophi, ovlp);
    //        if (ctx_.cfg().control().verification() >= 2 && ctx_.verbosity() >= 2) {
    //            auto s = ovlp.serialize("overlap", num_phi_tot, num_phi_tot);
    //            if (Hk__.kp().comm().rank() == 0) {
    //                ctx_.out() << s.str() << std::endl;
    //            }
    //        }

    //        double max_diff = check_hermitian(ovlp, num_phi_tot);
    //        if (max_diff > 1e-12) {
    //            std::stringstream s;
    //            s << "overlap matrix is not hermitian, max_err = " << max_diff;
    //            WARNING(s);
    //        }
    //        std::vector<real_type<T>> eo(num_phi_tot);
    //        auto& std_solver = ctx_.std_evp_solver();
    //        if (std_solver.solve(num_phi_tot, num_phi_tot, ovlp, eo.data(), evec)) {
    //            std::stringstream s;
    //            s << "error in diagonalization";
    //            WARNING(s);
    //        }
    //        Hk__.kp().message(1, __function_name__, "minimum eigen-value of the overlap matrix: %18.12f\n", eo[0]);
    //        if (eo[0] < 0) {
    //            WARNING("overlap matrix is not positively defined");
    //        }
    //    }

        /* setup eigen-value problem */
        Band(ctx).set_subspace_mtrx(0, num_phi_tot, 0, phi, hphi, hmlt);
        Band(ctx).set_subspace_mtrx(0, num_phi_tot, 0, phi, ophi, ovlp);

        if (pcs) {
            auto cs1 = hmlt.checksum(num_phi_tot, num_phi_tot);
            auto cs2 = ovlp.checksum(num_phi_tot, num_phi_tot);
            if (Hk__.kp().comm().rank() == 0) {
                utils::print_checksum("hmlt", cs1, RTE_OUT(std::cout));
                utils::print_checksum("ovlp", cs2, RTE_OUT(std::cout));
            }
        }

    //    if (ctx_.cfg().control().verification() >= 2 && ctx_.verbosity() >= 2) {
    //        auto s1 = hmlt.serialize("hmlt", num_phi_tot, num_phi_tot);
    //        auto s2 = hmlt.serialize("ovlp", num_phi_tot, num_phi_tot);
    //        if (Hk__.kp().comm().rank() == 0) {
    //            ctx_.out() << s1.str() << std::endl << s2.str() << std::endl;
    //        }
    //    }

        /* solve generalized eigen-value problem with the size N and get lowest num_bands eigen-vectors */
        if (gen_solver.solve(num_phi_tot, num_bands, hmlt, ovlp, eval.data(), evec)) {
            RTE_THROW("error in diagonalization");
        }

    //    if (ctx_.print_checksum()) {
    //        auto cs = evec.checksum(num_phi_tot, num_bands);
    //        real_type<T> cs1{0};
    //        for (int i = 0; i < num_bands; i++) {
    //            cs1 += eval[i];
    //        }
    //        if (Hk__.kp().comm().rank() == 0) {
    //            utils::print_checksum("evec", cs);
    //            utils::print_checksum("eval", cs1);
    //        }
    //    }
        {
            rte::ostream out(Hk__.kp().out(3), std::string(__func__));
            for (int i = 0; i < num_bands; i++) {
                out << "eval[" << i << "]=" << eval[i] << std::endl;
            }
        }

        /* compute wave-functions */
        /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
        for (int ispn = 0; ispn < num_sc; ispn++) {
            wf::transform(ctx.spla_context(), mem, evec, 0, 0, 1.0, phi,
                    wf::spin_index(num_sc == 2 ? ispn : 0), wf::band_range(0, num_phi_tot), 0.0,
                    Hk__.kp().spinor_wave_functions(), wf::spin_index(num_sc == 2 ? ispn : ispn_step),
                    wf::band_range(0, num_bands));
        }

        for (int j = 0; j < num_bands; j++) {
            Hk__.kp().band_energy(j, ispn_step, eval[j]);
        }
    }

    if (pcs) {
        for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
            auto cs = Hk__.kp().spinor_wave_functions().checksum(mem, wf::spin_index(ispn),
                    wf::band_range(0, num_bands));
            std::stringstream s;
            s << "initial_spinor_wave_functions_" << ispn;
            if (Hk__.kp().comm().rank() == 0) {
                utils::print_checksum(s.str(), cs, RTE_OUT(std::cout));
            }
        }
    }

    ///* check residuals */
    //if (ctx_.cfg().control().verification() >= 2) {
    //    check_residuals<T>(Hk__);
    //    check_wave_functions<T>(Hk__);
    //}

    //ctx_.print_memory_usage(__FILE__, __LINE__);
}


}

#endif // __BAND_HPP__
