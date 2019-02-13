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

#include "periodic_function.hpp"
#include "K_point/k_point_set.hpp"
#include "Hamiltonian/hamiltonian.hpp"

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
    BLACS_grid const& blacs_grid_;

    inline void solve_full_potential(K_point& kp__, Hamiltonian& hamiltonian__) const;

    /// Solve the first-variational (non-magnetic) problem with exact diagonalization.
    /** This is only used by the LAPW method. */
    inline void diag_full_potential_first_variation_exact(K_point& kp__, Hamiltonian& hamiltonian__) const;

    /// Solve the first-variational (non-magnetic) problem with iterative Davidson diagonalization.
    inline void diag_full_potential_first_variation_davidson(K_point& kp__, Hamiltonian& hamiltonian__) const;

    /// Solve second-variational problem.
    inline void diag_full_potential_second_variation(K_point& kp, Hamiltonian& hamiltonian__) const;

    /// Get singular components of the LAPW overlap matrix.
    /** Singular components are the eigen-vectors with a very small eigen-value. */
    inline void get_singular_components(K_point& kp__, Hamiltonian& H__) const;

    template <typename T>
    inline int solve_pseudo_potential(K_point& kp__, Hamiltonian& hamiltonian__) const;

    /// Diagonalize a pseudo-potential Hamiltonian.
    template <typename T>
    int diag_pseudo_potential(K_point* kp__, Hamiltonian& H__) const;

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

    template <typename T>
    void check_residuals(K_point& kp__, Hamiltonian& H__) const;

    /// Check wave-functions for orthonormalization.
    template <typename T>
    void check_wave_functions(K_point& kp__, Hamiltonian& H__) const
    {
        if (kp__.comm().rank() == 0) {
            printf("checking wave-functions\n");
        }

        if (!ctx_.full_potential()) {

            dmatrix<T> ovlp(ctx_.num_bands(), ctx_.num_bands(), ctx_.blacs_grid(), ctx_.cyclic_block_size(), ctx_.cyclic_block_size());

            const bool nc_mag = (ctx_.num_mag_dims() == 3);
            const int num_sc = nc_mag ? 2 : 1;

            auto& psi = kp__.spinor_wave_functions();
            Wave_functions spsi(kp__.gkvec_partition(), ctx_.num_bands(), ctx_.preferred_memory_t(), num_sc);

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
            kp__.beta_projectors().prepare();
            /* compute residuals */
            for (int ispin_step = 0; ispin_step < ctx_.num_spin_dims(); ispin_step++) {
                if (nc_mag) {
                    /* apply Hamiltonian and S operators to the wave-functions */
                    H__.apply_h_s<T>(&kp__, 2, 0, ctx_.num_bands(), psi, nullptr, &spsi);
                    inner(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), 2, psi, 0, ctx_.num_bands(),
                          spsi, 0, ctx_.num_bands(), ovlp, 0, 0);
                } else {
                    /* apply Hamiltonian and S operators to the wave-functions */
                    H__.apply_h_s<T>(&kp__, ispin_step, 0, ctx_.num_bands(), psi, 0, &spsi);
                    inner(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), 0, psi, 0, ctx_.num_bands(),
                          spsi, 0, ctx_.num_bands(), ovlp, 0, 0);
                }
                double diff = check_identity(ovlp, ctx_.num_bands());

                if (kp__.comm().rank() == 0) {
                    if (diff > 1e-12) {
                        printf("overlap matrix is not identity, maximum error : %f\n", diff);
                    } else {
                        printf("OK! Wave functions are orthonormal.\n");
                    }
                }
            }
            if (is_device_memory(ctx_.preferred_memory_t())) {
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    psi.pw_coeffs(ispn).deallocate(memory_t::device);
                }
            }
            kp__.beta_projectors().dismiss();
        }
    }

    /** Compute \f$ O_{ii'} = \langle \phi_i | \hat O | \phi_{i'} \rangle \f$ operator matrix
     *  for the subspace spanned by the wave-functions \f$ \phi_i \f$. The matrix is always returned
     *  in the CPU pointer because most of the standard math libraries start from the CPU. */
    template <typename T>
    inline void set_subspace_mtrx(int N__,
                                  int n__,
                                  Wave_functions& phi__,
                                  Wave_functions& op_phi__,
                                  dmatrix<T>& mtrx__,
                                  dmatrix<T>* mtrx_old__ = nullptr) const;

  public:
    /// Constructor
    Band(Simulation_context& ctx__)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
        , blacs_grid_(ctx__.blacs_grid())
    {
        if (!ctx_.initialized()) {
            TERMINATE("Simulation_context is not initialized");
        }
    }

    /// Solve \f$ \hat H \psi = E \psi \f$ and find eigen-states of the Hamiltonian.
    inline void solve(K_point_set& kset__, Hamiltonian& hamiltonian__, bool precompute__) const;

    /// Initialize the subspace for the entire k-point set.
    inline void initialize_subspace(K_point_set& kset__, Hamiltonian& hamiltonian__) const;

    /// Initialize the wave-functions subspace.
    template <typename T>
    inline void initialize_subspace(K_point* kp__, Hamiltonian& hamiltonian__, int num_ao__) const;

    static double& evp_work_count()
    {
        static double evp_work_count_{0};
        return evp_work_count_;
    }
};

#include "residuals.hpp"
#include "diag_full_potential.hpp"
#include "diag_pseudo_potential.hpp"
#include "initialize_subspace.hpp"
#include "solve.hpp"
#include "set_subspace_mtrx.hpp"

}

#endif // __BAND_HPP__
