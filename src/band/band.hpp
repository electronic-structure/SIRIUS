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
#include "generate_subspace_matrix.hpp"

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

    /// Solve the band eigen-problem for pseudopotential case.
    template <typename T, typename F>
    int solve_pseudo_potential(Hamiltonian_k<T>& Hk__, double itsol_tol__, double empy_tol__) const;

    /// Solve the band eigen-problem for full-potential case.
    template <typename T>
    void solve_full_potential(Hamiltonian_k<T>& Hk__, double itsol_tol__) const;

    /// Solve \f$ \hat H \psi = E \psi \f$ and find eigen-states of the Hamiltonian.
    template <typename T, typename F>
    void solve(K_point_set& kset__, Hamiltonian0<T>& H0__, double itsol_tol__) const;

    /// Initialize the subspace for the entire k-point set.
    //template <typename T>
    //void initialize_subspace(K_point_set& kset__, Hamiltonian0<T>& H0__) const;
};

}

#endif // __BAND_HPP__
