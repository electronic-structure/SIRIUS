// Copyright (c) 2013-2018 Mathieu Taillefumier, Anton Kozhevnikov, Thomas Schulthess
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

/** \file hubbard.hpp
 *
 *  \brief Contains declaration and partial implementation of sirius::Hubbard class.
 */

#ifndef __HUBBARD_HPP__
#define __HUBBARD_HPP__

#include <cstdio>
#include <cstdlib>
#include "simulation_context.hpp"
#include "K_point/k_point.hpp"
#include "K_point/k_point_set.hpp"
#include "wave_functions.hpp"
#include "wf_inner.hpp"
#include "wf_ortho.hpp"
#include "wf_trans.hpp"
#include "Hamiltonian/non_local_operator.hpp"
#include "Beta_projectors/beta_projectors.hpp"
#include "Beta_projectors/beta_projectors_gradient.hpp"
#include "Beta_projectors/beta_projectors_strain_deriv.hpp"
#include "radial_integrals.hpp"

namespace sirius {

/// Apply Hubbard correction in the colinear case
class Hubbard
{
  private:
    Simulation_context& ctx_;

    Unit_cell& unit_cell_;

    int max_number_of_orbitals_per_atom_{-1};

    int number_of_hubbard_orbitals_{0};

    mdarray<double_complex, 4> occupancy_number_;

    double hubbard_energy_{0.0};
    double hubbard_energy_u_{0.0};
    double hubbard_energy_dc_contribution_{0.0};
    double hubbard_energy_noflip_{0.0};
    double hubbard_energy_flip_{0.0};

    mdarray<double_complex, 4> hubbard_potential_;

    /// Type of hubbard correction to be considered.
    /** True if we consider a simple hubbard correction. Not valid if spin orbit coupling is included */
    bool approximation_{false};

    /// Orthogonalize and/or normalize the projectors.
    bool orthogonalize_hubbard_orbitals_{false};

    /// True if localized orbitals have to be normalized.
    bool normalize_orbitals_only_{false};

    /// hubbard correction with next nearest neighbors
    bool hubbard_U_plus_V_{false};

    /// hubbard projection method. By default we use the wave functions
    /// provided by the pseudo potentials.
    int projection_method_{0};

    /// Hubbard with multi channels (not implemented yet)
    bool multi_channels_{false};

    /// file containing the hubbard wave functions
    std::string wave_function_file_;

    void calculate_initial_occupation_numbers();

    void compute_occupancies(K_point&                    kp,
                             dmatrix<double_complex>&    phi_s_psi,
                             dmatrix<double_complex>&    dphi_s_psi,
                             Wave_functions&             dphi,
                             mdarray<double_complex, 5>& dn_,
                             matrix<double_complex>&     dm,
                             const int                   index);

    void symmetrize_occupancy_matrix_noncolinear_case();
    void symmetrize_occupancy_matrix();
    void print_occupancies();

    void calculate_wavefunction_with_U_offset();

    /// Compute the strain gradient of the hubbard wave functions.
    void wavefunctions_strain_deriv(K_point& kp, Wave_functions& dphi, mdarray<double, 2> const& rlm_g,
                                    mdarray<double, 3> const& rlm_dg, const int mu, const int nu);

  public:
    /// Constructor.
    Hubbard(Simulation_context& ctx__);

    std::vector<int> offset_;

    void set_hubbard_U_plus_V()
    {
        hubbard_U_plus_V_ = true;
    }

    void set_hubbard_simple_correction()
    {
        approximation_ = true;
    }

    inline int max_number_of_orbitals_per_atom() const
    {
        return max_number_of_orbitals_per_atom_;
    }

    void set_orthogonalize_hubbard_orbitals(const bool test)
    {
        orthogonalize_hubbard_orbitals_ = test;
    }

    void set_normalize_hubbard_orbitals(const bool test)
    {
        this->normalize_orbitals_only_ = test;
    }

    double_complex U(int m1, int m2, int m3, int m4) const
    {
        return hubbard_potential_(m1, m2, m3, m4);
    }

    double_complex& U(int m1, int m2, int m3, int m4)
    {
        return hubbard_potential_(m1, m2, m3, m4);
    }

    bool orthogonalize_hubbard_orbitals() const
    {
        return orthogonalize_hubbard_orbitals_;
    }

    bool normalize_hubbard_orbitals() const
    {
        return normalize_orbitals_only_;
    }

    /// Apply the hubbard potential on wave functions
    void apply_hubbard_potential(Wave_functions& hub_wf,
                                 const int       ispn,
                                 const int       idx,
                                 const int       n,
                                 Wave_functions& phi,
                                 Wave_functions& ophi);

    /// Generate the atomic orbitals.
    //void generate_atomic_orbitals(K_point& kp, Q_operator& q_op);

    void hubbard_compute_occupation_numbers(K_point_set& kset_);

    void compute_occupancies_derivatives(K_point&                    kp,
                                         Q_operator& q_op,
                                         mdarray<double_complex, 6>& dn);

    /// Compute derivatives of the occupancy matrix w.r.t.atomic displacement.
    /** \param [in]  kp   K-point.
     *  \param [in]  q_op Overlap operator.
     *  \param [out] dn   Derivative of the occupation number compared to displacement of each atom.
     */
    void compute_occupancies_stress_derivatives(K_point&                    kp,
                                                Q_operator& q_op,
                                                mdarray<double_complex, 5>& dn);

    void calculate_hubbard_potential_and_energy_colinear_case();
    void calculate_hubbard_potential_and_energy_non_colinear_case();
    void calculate_hubbard_potential_and_energy()
    {
        this->hubbard_energy_                 = 0.0;
        this->hubbard_energy_u_               = 0.0;
        this->hubbard_energy_dc_contribution_ = 0.0;
        this->hubbard_energy_noflip_          = 0.0;
        this->hubbard_energy_flip_            = 0.0;
        // the hubbard potential has the same structure than the occupation
        // numbers
        this->hubbard_potential_.zero();

        if (ctx_.num_mag_dims() != 3) {
            calculate_hubbard_potential_and_energy_colinear_case();
        } else {
            calculate_hubbard_potential_and_energy_non_colinear_case();
        }
    }

    inline double hubbard_energy() const
    {
        return this->hubbard_energy_;
    }

    inline int number_of_hubbard_orbitals() const
    {
        return number_of_hubbard_orbitals_;
    }

    mdarray<double_complex, 4>& occupation_matrix()
    {
        return occupancy_number_;
    }

    mdarray<double_complex, 4>& potential_matrix()
    {
        return hubbard_potential_;
    }

    void access_hubbard_potential(char const*     what,
                                  double_complex* occ,
                                  int const*      ld);

    void access_hubbard_occupancies(char const*     what,
                                    double_complex* occ,
                                    int const*      ld);

};

} // namespace sirius

#endif // __HUBBARD_HPP__
