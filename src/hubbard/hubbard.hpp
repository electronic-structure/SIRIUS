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
#include "context/simulation_context.hpp"
#include "k_point/k_point.hpp"
#include "k_point/k_point_set.hpp"
#include "SDDK/wave_functions.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "beta_projectors/beta_projectors.hpp"
#include "beta_projectors/beta_projectors_gradient.hpp"
#include "beta_projectors/beta_projectors_strain_deriv.hpp"
#include "radial/radial_integrals.hpp"
#include "hubbard_matrix.hpp"

namespace sirius {

void generate_potential(Hubbard_matrix const& om__, Hubbard_matrix& um__);

double energy(Hubbard_matrix const& om__);
double one_electron_energy_hubbard(Hubbard_matrix const& om__, Hubbard_matrix const& pm__);

/// Apply Hubbard correction in the collinear case
class Hubbard
{
  private:
    Simulation_context& ctx_;

    Unit_cell& unit_cell_;

    /// Hubbard correction with next nearest neighbors
    bool hubbard_U_plus_V_{false};

    /// Hubbard with multi channels apply to both LDA+U+V case
    bool multi_channels_{false};

    void compute_occupancies(K_point<double>& kp__, dmatrix<double_complex>& phi_s_psi__,
                             Wave_functions<double>& dphi__, mdarray<double_complex, 4>& dn__, const int index__);

    void calculate_wavefunction_with_U_offset();

    /// Compute the strain gradient of the hubbard wave functions.
    void wavefunctions_strain_deriv(K_point<double>& kp, Wave_functions<double>& dphi, mdarray<double, 2> const& rlm_g,
                                    mdarray<double, 3> const& rlm_dg, const int mu, const int nu);
    // TODO: make a seprate function
    void apply_dS(K_point<double>& kp, Q_operator<double>& q_op, Beta_projectors_gradient<double>& bp_grad,
                  const int atom_id, const int dir, Wave_functions<double>& phi, Wave_functions<double>& dphi);

  public:
    /// Constructor.
    Hubbard(Simulation_context& ctx__);

    /// Compute the occupancy derivatives with respect to atomic displacements.
    /**
     *  Derivative of the inverse square root of the overlap matrix.
     *
     *  Let's label \f$ \frac{\partial}{\partial {\bf r}_{\alpha}} {\bf O}^{-1/2} = {\bf X} \f$.
     *  From taking derivative of the identity
     *  \f[
     *     {\bf O}^{-1/2} {\bf O} {\bf O}^{-1/2}= {\bf I}
     *  \f]
     *  we get
     *  \f[
     *    {\bf X} {\bf O} {\bf O}^{-1/2} + {\bf O}^{-1/2}{\bf O}'{\bf O}^{-1/2} + {\bf O}^{-1/2}{\bf O}{\bf X} = 0
     *  \f]
     *  or simplifying
     *  \f[
     *    {\bf X} {\bf O}^{1/2} + {\bf O}^{1/2}{\bf X} = -{\bf O}^{-1/2}{\bf O}'{\bf O}^{-1/2}
     *  \f]
     *  We have equation of the type
     *  \f[
     *    {\bf X} {\bf A} + {\bf A} {\bf X} = {\bf B}
     *  \f]
     *  which is a symmetric Sylvester equation. To solve it we SVD the \f$ {\bf A} \f$ matrix
     *  (which is \f$ {\bf O}^{1/2} \f$):
     *  \f[
     *    {\bf A} = {\bf U}{\bf \Lambda}^{1/2} {\bf U}^{H}
     *  \f]
     *  Here \f$ \Lambda_i \f$ and \f$ {\bf U} \f$ are the eigen-values and eigen-vectors of the overap matrix
     *  \f$ {\bf O} \f$. Now we put it back to the original equation:
     *  \f[
     *    {\bf X} {\bf U}{\bf \Lambda}^{1/2} {\bf U}^{H} + {\bf U}{\bf \Lambda}^{1/2} {\bf U}^{H}{\bf X} = {\bf B}
     *  \f]
     *  Now we multiply with \f$ {\bf U}^{H} \f$ from the left and with \f$ {\bf U}\f$ from the right and
     *  simplify the right-hand side:
     *  \f[
     *    {\bf U}^{H} {\bf X} {\bf U}{\bf \Lambda}^{1/2} + {\bf \Lambda}^{1/2} {\bf U}^{H} {\bf X} {\bf U} =
     *      {\bf U}^{H} {\bf B} {\bf U} = -{\bf U}^{H} \Big( {\bf U}{\bf \Lambda}^{-1/2} {\bf U}^{H} \Big)
     *    {\bf O}' \Big( {\bf U}{\bf \Lambda}^{-1/2} {\bf U}^{H}\Big) {\bf U} =
     *     -{\bf \Lambda}^{-1/2} {\bf U}^{H} {\bf O}' {\bf U}{\bf \Lambda}^{-1/2}
     *  \f]
     *  or in the new basis
     *  \f[
     *    \tilde{\bf X} {\bf \Lambda}^{1/2} + {\bf \Lambda}^{1/2} \tilde{\bf X} = \tilde {\bf B}
     *  \f]
     *  Because \f$ {\bf \Lambda}^{1/2} \f$ is a diagonal matrix, we can write the last equation for each
     *  element of the matrix \f$ \tilde{\bf X} \f$:
     *  \f[
     *    \tilde X_{ij} \Lambda_{j}^{1/2} + \Lambda_{i}^{1/2} \tilde X_{ij} = \tilde B_{ij}
     *  \f]
     *  And thus
     *  \f[
     *    \tilde X_{ij} =  \frac{\tilde B_{ij}}{\Lambda_{i}^{1/2} + \Lambda_{j}^{1/2}}
     *  \f]
     *
     *  The elements of matrix \f$ \tilde {\bf B} \f$ have the following expression:
     *  \f[
     *    \tilde B_{ij} = -\Lambda_{i}^{-1/2} \tilde O_{ij}' \Lambda_{j}^{-1/2}
     *  \f]
     *  where
     *  \f[
     *    \tilde {\bf O'} = {\bf U}^{H} {\bf O}' {\bf U}
     *  \f]
     *
     *  Putting all together, we can write the final expression for \f$ {\bf X} \f$:
     * \f[
     *  {\bf X} = -{\bf U} \frac{\Lambda_{i}^{-1/2} ({\bf U}^{H}{\bf O}'{\bf U})_{ij} \Lambda_{j}^{-1/2} }
     *    {\Lambda_{i}^{1/2} + \Lambda_{j}^{1/2}} {\bf U}^{H}
     *
     *  \f]
     *
     *  Now we can draft the calculation of \f$ \frac{\partial}{\partial {\bf r}_{\alpha}} {\bf O}^{-1/2} \f$:
     *    - SVD the overlap matrix \f$ {\bf O} \f$
     *    - compute the derivative of \f$ {\bf O} \f$ for each atom displacement
     *    - compute \f$ \tilde {\bf O'} = {\bf U}^{H} {\bf O}' {\bf U} \f$
     *    - compute \f$ \tilde X_{ij} = \frac{\Lambda_{i}^{-1/2} \tilde O_{ij}' \Lambda_{j}^{-1/2}} {\Lambda_{i}^{1/2} + \Lambda_{j}^{1/2}} \f$
     *    - compute \f$ \frac{\partial}{\partial {\bf r}_{\alpha}} {\bf O}^{-1/2} = -{\bf U}\tilde {\bf X}{\bf U}^{H} \f$
     */
    void compute_occupancies_derivatives(K_point<double>& kp, Q_operator<double>& q_op,
                                         mdarray<std::complex<double>, 5>& dn);

    /// Compute derivatives of the occupancy matrix w.r.t.atomic displacement.
    /** \param [in]  kp   K-point.
     *  \param [in]  q_op Overlap operator.
     *  \param [out] dn   Derivative of the occupation number compared to displacement of each atom.
     */
    void compute_occupancies_stress_derivatives(K_point<double>& kp, Q_operator<double>& q_op,
                                                sddk::mdarray<std::complex<double>, 4>& dn);

    void set_hubbard_U_plus_V()
    {
        hubbard_U_plus_V_ = true;
    }

    inline int num_hubbard_wf() const
    {
        return ctx_.unit_cell().num_hubbard_wf().first;
    }
};

} // namespace sirius

#endif // __HUBBARD_HPP__
