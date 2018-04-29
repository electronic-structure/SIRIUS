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
#include "Hamiltonian/local_operator.hpp"
#include "non_local_operator.h"
#include "hubbard.hpp"
#include "Hamiltonian.h"

namespace sirius {

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
    void check_residuals(K_point* kp__, Hamiltonian& H__) const;

    /** Compute \f$ O_{ii'} = \langle \phi_i | \hat O | \phi_{i'} \rangle \f$ operator matrix
     *  for the subspace spanned by the wave-functions \f$ \phi_i \f$. The matrix is always returned
     *  in the CPU pointer because most of the standard math libraries start from the CPU. */
    template <typename T>
    inline void set_subspace_mtrx(int N__,
                                  int n__,
                                  Wave_functions& phi__,
                                  Wave_functions& op_phi__,
                                  dmatrix<T>& mtrx__,
                                  dmatrix<T>& mtrx_old__) const;

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

#include "Band/residuals.hpp"
#include "Band/diag_full_potential.hpp"
#include "Band/diag_pseudo_potential.hpp"
#include "Band/initialize_subspace.hpp"
#include "Band/solve.hpp"
#include "Band/set_subspace_mtrx.hpp"

}

#endif // __BAND_H__
