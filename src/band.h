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

/** \file band.h
 *
 *   \brief Contains declaration and partial implementation of sirius::Band class.
 */

#ifndef __BAND_H__
#define __BAND_H__

#include "periodic_function.h"
#include "k_point_set.h"
#include "local_operator.h"
#include "non_local_operator.h"
#include "hubbard.hpp"
#include "Hamiltonian.h"

namespace sirius {

// TODO: Band problem is a mess and needs more formal organizaiton. We have different basis functions.
//       We can do first- and second-variation or a full variation. We can do iterative or exact diagonalization.
//       This has to be organized.

/// Setup and solve the eigen value problem.
class Band
{
  private:
    /// Simulation context.
    Simulation_context& ctx_;

    /// Alias for the unit cell.
    Unit_cell& unit_cell_;

    /// BLACS grid for distributed linear algebra operations.
    BLACS_grid const& blacs_grid_;

    /// Solve the band diagonalziation problem with single (full) variation.
    inline int solve_with_single_variation(K_point& kp__, Hamiltonian& hamiltonian__) const;

    /// Solve the band diagonalziation problem with second variation approach.
    /** This is only used by the FP-LAPW method. */
    inline void solve_with_second_variation(K_point& kp__, Hamiltonian& hamiltonian__) const;

    /// Solve the first-variational (non-magnetic) problem with exact diagonalization.
    /** This is only used by the LAPW method. */
    inline void diag_fv_exact(K_point* kp__, Hamiltonian& hamiltonian__) const;

    /// Solve the first-variational (non-magnetic) problem with iterative Davidson diagonalization.
    inline void diag_fv_davidson(K_point* kp__, Hamiltonian& hamiltonian__) const;

    /// Get singular components of the LAPW overlap matrix.
    /** Singular components are the eigen-vectors with a very small eigen-value. */
    inline void get_singular_components(K_point* kp__, Hamiltonian& H__) const;

    /// Exact (not iterative) diagonalization of the Hamiltonian.
    template <typename T>
    inline void diag_pseudo_potential_exact(K_point* kp__, int ispn__, Hamiltonian& H__) const;

    /// Iterative Davidson diagonalization.
    template <typename T>
    inline int diag_pseudo_potential_davidson(K_point* kp__, Hamiltonian& H__) const;
    /// RMM-DIIS diagonalization.
    template <typename T>
    inline void diag_pseudo_potential_rmm_diis(K_point* kp__, int ispn__, Hamiltonian& H__) const;

    template <typename T>
    inline void
    diag_pseudo_potential_chebyshev(K_point* kp__, int ispn__, Hamiltonian& H__, P_operator<T>& p_op__) const;

    /// Auxiliary function used internally by residuals() function.
    inline mdarray<double, 1> residuals_aux(K_point* kp__,
                                            int ispn__,
                                            int num_bands__,
                                            std::vector<double>& eval__,
                                            Wave_functions& hpsi__,
                                            Wave_functions& opsi__,
                                            Wave_functions& res__,
                                            mdarray<double, 2>& h_diag__,
                                            mdarray<double, 1>& o_diag__) const;

    template <typename T>
    int residuals_common(K_point* kp__,
                         int ispn__,
                         int N__,
                         int num_bands__,
                         std::vector<double>& eval__,
                         std::vector<double>& eval_old__,
                         dmatrix<T>& evec__,
                         Wave_functions& hphi__,
                         Wave_functions& ophi__,
                         Wave_functions& hpsi__,
                         Wave_functions& opsi__,
                         Wave_functions& res__,
                         mdarray<double, 2>& h_diag__,
                         mdarray<double, 1>& o_diag__) const;

    /// Compute residuals.
    template <typename T>
    inline int residuals(K_point* kp__,
                         int ispn__,
                         int N__,
                         int num_bands__,
                         std::vector<double>& eval__,
                         std::vector<double>& eval_old__,
                         dmatrix<T>& evec__,
                         Wave_functions& hphi__,
                         Wave_functions& ophi__,
                         Wave_functions& hpsi__,
                         Wave_functions& opsi__,
                         Wave_functions& res__,
                         mdarray<double, 2>& h_diag__,
                         mdarray<double, 1>& o_diag__) const;

    /** Compute \f$ O_{ii'} = \langle \phi_i | \hat O | \phi_{i'} \rangle \f$ operator matrix
     *  for the subspace spanned by the wave-functions \f$ \phi_i \f$. The matrix is always returned
     *  in the CPU pointer because most of the standard math libraries start from the CPU. */
    template <typename T>
    inline void set_subspace_mtrx(int N__,
                                  int n__,
                                  Wave_functions& phi__,
                                  Wave_functions& op_phi__,
                                  dmatrix<T>& mtrx__,
                                  dmatrix<T>& mtrx_old__) const
    {
        PROFILE("sirius::Band::set_subspace_mtrx");

        assert(n__ != 0);
        if (mtrx_old__.size()) {
            assert(&mtrx__.blacs_grid() == &mtrx_old__.blacs_grid());
        }

        /* copy old N x N distributed matrix */
        if (N__ > 0) {
            splindex<block_cyclic> spl_row(N__, mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().rank_row(),
                                           mtrx__.bs_row());
            splindex<block_cyclic> spl_col(N__, mtrx__.blacs_grid().num_ranks_col(), mtrx__.blacs_grid().rank_col(),
                                           mtrx__.bs_col());

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spl_col.local_size(); i++) {
                std::copy(&mtrx_old__(0, i), &mtrx_old__(0, i) + spl_row.local_size(), &mtrx__(0, i));
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
                    print_checksum("subspace_mtrx_old", cs);
                }
            }
        }

        /* <{phi,phi_new}|Op|phi_new> */
        inner(ctx_.processing_unit(), (ctx_.num_mag_dims() == 3) ? 2 : 0, phi__, 0, N__ + n__, op_phi__, N__, n__,
              mtrx__, 0, N__);

        /* restore lower part */
        if (N__ > 0) {
            if (mtrx__.blacs_grid().comm().size() == 1) {
                #pragma omp parallel for
                for (int i = 0; i < N__; i++) {
                    for (int j = N__; j < N__ + n__; j++) {
                        mtrx__(j, i) = type_wrapper<T>::bypass(std::conj(mtrx__(i, j)));
                    }
                }
            } else {
                linalg<CPU>::tranc(n__, N__, mtrx__, 0, N__, mtrx__, N__, 0);
            }
        }

        if (ctx_.control().print_checksum_) {
            splindex<block_cyclic> spl_row(N__ + n__, mtrx__.blacs_grid().num_ranks_row(),
                                           mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
            splindex<block_cyclic> spl_col(N__ + n__, mtrx__.blacs_grid().num_ranks_col(),
                                           mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());
            double_complex cs(0, 0);
            for (int i = 0; i < spl_col.local_size(); i++) {
                for (int j = 0; j < spl_row.local_size(); j++) {
                    cs += mtrx__(j, i);
                }
            }
            mtrx__.blacs_grid().comm().allreduce(&cs, 1);
            if (ctx_.comm_band().rank() == 0) {
                print_checksum("subspace_mtrx", cs);
            }
        }

        /* kill any numerical noise */
        mtrx__.make_real_diag(N__ + n__);

        /* save new matrix */
        if (mtrx_old__.size()) {
            splindex<block_cyclic> spl_row(N__ + n__, mtrx__.blacs_grid().num_ranks_row(),
                                           mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
            splindex<block_cyclic> spl_col(N__ + n__, mtrx__.blacs_grid().num_ranks_col(),
                                           mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spl_col.local_size(); i++) {
                std::copy(&mtrx__(0, i), &mtrx__(0, i) + spl_row.local_size(), &mtrx_old__(0, i));
            }
        }
    }

    /// Diagonalize a pseudo-potential Hamiltonian.
    template <typename T>
    int diag_pseudo_potential(K_point* kp__, Hamiltonian& H__) const
    {
        PROFILE("sirius::Band::diag_pseudo_potential");

        H__.local_op().prepare(kp__->gkvec_partition());
        ctx_.fft_coarse().prepare(kp__->gkvec_partition());
        H__.create_d_and_q_operator<T>();
        H__.initialize_D_and_Q_operators<T>();

        int niter{0};

        auto& itso = ctx_.iterative_solver_input();
        if (itso.type_ == "exact") {
            if (ctx_.num_mag_dims() != 3) {
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    diag_pseudo_potential_exact<double_complex>(kp__, ispn, H__);
                }
            } else {
                STOP();
            }
        } else if (itso.type_ == "davidson") {
            niter = diag_pseudo_potential_davidson<T>(kp__, H__);
        } else if (itso.type_ == "rmm-diis") {
            if (ctx_.num_mag_dims() != 3) {
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    diag_pseudo_potential_rmm_diis<T>(kp__, ispn, H__);
                }
            } else {
                STOP();
            }
        } else if (itso.type_ == "chebyshev") {
            P_operator<T> p_op(ctx_, kp__->p_mtrx());
            if (ctx_.num_mag_dims() != 3) {
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    diag_pseudo_potential_chebyshev<T>(kp__, ispn, H__, p_op);
                }
            } else {
                STOP();
            }
        } else {
            TERMINATE("unknown iterative solver type");
        }

        ctx_.fft_coarse().dismiss();
        return niter;
    }

  public:
    /// Constructor
    Band(Simulation_context& ctx__)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
        , blacs_grid_(ctx__.blacs_grid())
    {
    }

    /// Solve second-variational problem.
    inline void diag_sv(K_point* kp, Hamiltonian& hamiltonian__) const;

    /// Solve \f$ \hat H \psi = E \psi \f$ and find eigen-states of the Hamiltonian.
    inline void solve_for_kset(K_point_set& kset__, Hamiltonian& hamiltonian__, bool precompute__) const;

    /// Initialize the subspace for the entire k-point set.
    inline void initialize_subspace(K_point_set& kset__, Hamiltonian& hamiltonian__) const;

    /// Initialize the wave-functions subspace.
    template <typename T>
    inline void initialize_subspace(K_point* kp__, Hamiltonian& hamiltonian__, int num_ao__) const;
};

#include "Band/residuals.hpp"
#include "Band/diag_full_potential.hpp"
#include "Band/diag_pseudo_potential.hpp"
#include "Band/initialize_subspace.hpp"
#include "Band/solve.hpp"
}

#endif // __BAND_H__
