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
#include "wave_functions.hpp"
#include "Hamiltonian/non_local_operator.hpp"
#include "Beta_projectors/beta_projectors.hpp"
#include "Beta_projectors/beta_projectors_gradient.hpp"
#include "Beta_projectors/beta_projectors_strain_deriv.hpp"
#include "radial_integrals.hpp"
#include "mixer.hpp"

namespace sirius {

/// Apply Hubbard correction in the colinear case
class Hubbard
{
  private:
    Simulation_context& ctx_;

    Unit_cell& unit_cell_;

    int lmax_{-1};

    int number_of_hubbard_orbitals_{0};

    mdarray<double_complex, 5> occupancy_number_;

    double hubbard_energy_{0.0};
    double hubbard_energy_u_{0.0};
    double hubbard_energy_dc_contribution_{0.0};
    double hubbard_energy_noflip_{0.0};
    double hubbard_energy_flip_{0.0};

    mdarray<double_complex, 5> hubbard_potential_;

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

    inline void symmetrize_occupancy_matrix_noncolinear_case();
    inline void symmetrize_occupancy_matrix();
    inline void print_occupancies();

    inline void calculate_wavefunction_with_U_offset()
    {
        offset.clear();
        offset.resize(ctx_.unit_cell().num_atoms(), -1);

        int counter = 0;

        // we loop over atoms to check which atom has hubbard orbitals
        // and then compute the number of hubbard orbitals associated to
        // it.
        for (auto ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom = unit_cell_.atom(ia);
            if (atom.type().hubbard_correction()) {
                offset[ia] = counter;
                if (ctx_.num_mag_dims() == 3) {
                    for (auto&& orb : atom.type().hubbard_orbital()) {
                        if (atom.type().spin_orbit_coupling()) {
                            counter += (2 * orb.l() + 1);
                        } else {
                            counter += 2 * (2 * orb.l() + 1);
                        }
                    }
                } else {
                    for (auto&& orb : atom.type().hubbard_orbital()) {
                        counter += (2 * orb.l() + 1);
                    }
                }
            }
        }

        this->number_of_hubbard_orbitals_ = counter;
    }

    /// Compute the strain gradient of the hubbard wave functions.
    /// Unfortunately it is dependent of the pp.

    void compute_gradient_strain_wavefunctions(K_point&                  kp,
                                               Wave_functions&           dphi,
                                               const mdarray<double, 2>& rlm_g,
                                               const mdarray<double, 3>& rlm_dg,
                                               const int                 mu,
                                               const int                 nu);

    /// apply the S operator in the us pp case. Otherwise it makes a simple copy
    void apply_S_operator(K_point&        kp,
                          Q_operator&     q_op,
                          Wave_functions& phi,
                          Wave_functions& ophi,
                          const int       idx0,
                          const int       num_phi);

    /// orthogonize (normalize) the hubbard wave functions
    void orthogonalize_atomic_orbitals(K_point& kp, Wave_functions& sphi);

  public:
    std::vector<int> offset;

    void set_hubbard_U_plus_V()
    {
        hubbard_U_plus_V_ = true;
    }

    void set_hubbard_simple_correction()
    {
        approximation_ = true;
    }

    inline int lmax() const
    {
        return lmax_;
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
        return hubbard_potential_(m1, m2, m3, m4, 0);
    }

    double_complex& U(int m1, int m2, int m3, int m4)
    {
        return hubbard_potential_(m1, m2, m3, m4, 0);
    }

    double_complex U(int m1, int m2, int m3, int m4, int channel) const
    {
        return hubbard_potential_(m1, m2, m3, m4, channel);
    }

    double_complex& U(int m1, int m2, int m3, int m4, int channel)
    {
        return hubbard_potential_(m1, m2, m3, m4, channel);
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
    void apply_hubbard_potential(K_point&        kp,
                                 const int       ispn,
                                 const int       idx,
                                 const int       n,
                                 Wave_functions& phi,
                                 Wave_functions& ophi);

    /// Generate the atomic orbitals.
    void generate_atomic_orbitals(K_point& kp, Q_operator& q_op);

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

    Hubbard(Simulation_context& ctx__)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
    {
        if (!ctx_.hubbard_correction()) {
            return;
        }
        orthogonalize_hubbard_orbitals_ = ctx_.Hubbard().orthogonalize_hubbard_orbitals_;
        normalize_orbitals_only_        = ctx_.Hubbard().normalize_hubbard_orbitals_;
        projection_method_              = ctx_.Hubbard().projection_method_;

        // if the projectors are defined externaly then we need the file
        // that contains them. All the other methods do not depend on
        // that parameter
        if (this->projection_method_ == 1) {
            this->wave_function_file_ = ctx_.Hubbard().wave_function_file_;
        }

        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            auto& atom_type = ctx_.unit_cell().atom(ia).type();
            if (ctx__.unit_cell().atom(ia).type().hubbard_correction()) {
                for (int channel = 0; channel < atom_type.number_of_hubbard_channels(); channel++) {
                    this->lmax_ = std::max(this->lmax_, atom_type.hubbard_orbital(channel).l());
                }
            }
        }

        /// if spin orbit coupling or non colinear magnetisms are
        /// activated, then we consider the full spherical hubbard
        /// correction
        if ((ctx_.so_correction()) || (ctx_.num_mag_dims() == 3)) {
            approximation_ = false;
        }

        // prepare things for the multi channel case. The last index
        // indicates which channel we consider. By default we only have
        // one channel per atomic type
        occupancy_number_  = mdarray<double_complex, 5>(2 * lmax_ + 1, 2 * lmax_ + 1, 4, ctx_.unit_cell().num_atoms(), 1);
        hubbard_potential_ = mdarray<double_complex, 5>(2 * lmax_ + 1, 2 * lmax_ + 1, 4, ctx_.unit_cell().num_atoms(), 1);

        calculate_wavefunction_with_U_offset();
        calculate_initial_occupation_numbers();
        calculate_hubbard_potential_and_energy();
    }

    mdarray<double_complex, 5>& occupation_matrix()
    {
        return occupancy_number_;
    }

    mdarray<double_complex, 5>& potential_matrix()
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

#include "hubbard_generate_atomic_orbitals.hpp"
#include "hubbard_potential_energy.hpp"
#include "apply_hubbard_potential.hpp"
#include "hubbard_occupancy.hpp"
#include "hubbard_occupancies_derivatives.hpp"

} // namespace sirius

#endif // __HUBBARD_HPP__
