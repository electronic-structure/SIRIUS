// Copyright (c) 2013-2018 Anton Kozhevnikov, Mathieu Taillefumier, Thomas Schulthess
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

/** \file hamiltonian.hpp
 *
 *  \brief Contains declaration and definition of sirius::Hamiltonian class.
 */

#ifndef __HAMILTONIAN_HPP__
#define __HAMILTONIAN_HPP__

#include <memory>
#include <complex>
#include "SDDK/memory.hpp"
#include "typedefs.hpp"

namespace sddk {
/* forward declaration */
class Wave_functions;
template <typename T>
class dmatrix;
class spin_range;
}

namespace sirius {
/* forward declaration */
class Atom;
class Simulation_context;
class Potential;
class Unit_cell;
class Local_operator;
class D_operator;
class Q_operator;
class K_point;
template <typename T>
class Gaunt_coefficients;

/// Representation of Kohn-Sham Hamiltonian.
/** In general, Hamiltonian consists of kinetic term, local part of potential and non-local
    part of potential:
    \f[
      H = -\frac{1}{2} \nabla^2 + V_{loc}({\bf r}) + \sum_{\alpha} \sum_{\xi \xi'} |\beta_{\xi}^{\alpha} \rangle
        D_{\xi \xi'}^{\alpha} \langle \beta_{\xi'}^{\alpha}|
    \f]
*/

/* forward declaration */
class Hamiltonian_k;

/// Represent the k-point independent part of Hamiltonian.
class Hamiltonian0
{
  private:
    /// Simulation context.
    Simulation_context& ctx_;

    /// Alias for the potential.
    Potential* potential_{nullptr};

    /// Alias for unit cell.
    Unit_cell& unit_cell_;

    /// Local part of the Hamiltonian operator.
    std::unique_ptr<Local_operator> local_op_;

    /// Non-zero Gaunt coefficients
    std::unique_ptr<Gaunt_coefficients<double_complex>> gaunt_coefs_;

    /// D operator (non-local part of Hamiltonian).
    std::unique_ptr<D_operator> d_op_;

    /// Q operator (non-local part of S-operator).
    std::unique_ptr<Q_operator> q_op_;

    /* copy constructor is forbidden */
    Hamiltonian0(Hamiltonian0 const& src) = delete;
    /* copy assigment operator is forbidden */
    Hamiltonian0& operator=(Hamiltonian0 const& src) = delete;

  public:
    /// Constructor.
    //Hamiltonian0(Simulation_context& ctx__);

    /// Constructor.
    Hamiltonian0(Potential& potential__);

    ~Hamiltonian0();

    /// Default move constructor.
    Hamiltonian0(Hamiltonian0&& src) = default;

    /// Return a Hamiltonian for the given k-point.
    inline Hamiltonian_k operator()(K_point& kp__);

    Simulation_context& ctx() const
    {
        return ctx_;
    }

    Potential& potential() const
    {
        return *potential_;
    }

    Local_operator& local_op() const
    {
        return *local_op_;
    }

    inline Gaunt_coefficients<double_complex> const& gaunt_coefs() const
    {
        return *gaunt_coefs_;
    }

    inline Q_operator& Q() const
    {
        return *q_op_;
    }

    inline D_operator& D() const
    {
        return *d_op_;
    }

    /// Apply the muffin-tin part of the Hamiltonian to the apw basis functions of an atom.
    /** The following matrix is computed:
     *  \f[
     *    b_{L_2 \nu_2}^{\alpha}({\bf G'}) = \sum_{L_1 \nu_1} \sum_{L_3}
     *      a_{L_1\nu_1}^{\alpha}({\bf G'})
     *      \langle u_{\ell_1\nu_1}^{\alpha} | h_{L3}^{\alpha} |  u_{\ell_2\nu_2}^{\alpha}
     *      \rangle  \langle Y_{L_1} | R_{L_3} | Y_{L_2} \rangle
     *  \f]
     */
    template <spin_block_t sblock>
    void apply_hmt_to_apw(Atom const& atom__, int ngv__, sddk::mdarray<double_complex, 2>& alm__,
                          sddk::mdarray<double_complex, 2>& halm__) const;

    /// Add correction to LAPW overlap arising in the infinite-order relativistic approximation (IORA).
    void add_o1mt_to_apw(Atom const& atom__, int num_gkvec__, sddk::mdarray<double_complex, 2>& alm__) const; // TODO: documentation

    /// Apply muffin-tin part of magnetic filed to the wave-functions.
    void apply_bmt(sddk::Wave_functions& psi__, std::vector<sddk::Wave_functions>& bpsi__) const;

    /// Apply SO correction to the first-variational LAPW wave-functions.
    /** Raising and lowering operators:
     *  \f[
     *      L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
     *  \f]
     */
    void apply_so_correction(sddk::Wave_functions& psi__, std::vector<sddk::Wave_functions>& hpsi__) const;
};

class Hamiltonian_k
{
  private:
    Hamiltonian0& H0_;
    K_point& kp_;

    /// Copy constructor is forbidden.
    Hamiltonian_k(Hamiltonian_k const& src__) = delete;

    /// Assignment operator is forbidden.
    Hamiltonian_k& operator=(Hamiltonian_k const& src__) = delete;

  public:
    Hamiltonian_k(Hamiltonian0& H0__, K_point& kp__);

    ~Hamiltonian_k();

    Hamiltonian0 const& H0() const
    {
        return H0_;
    }

    K_point& kp()
    {
        return kp_;
    }

    K_point const& kp() const
    {
        return kp_;
    }

    Hamiltonian_k(Hamiltonian_k&& src__) = default;

    template <typename T, int what>
    std::pair<sddk::mdarray<double, 2>, sddk::mdarray<double, 2>>
    get_h_o_diag_pw() const;

    template <int what>
    std::pair<sddk::mdarray<double, 2>, sddk::mdarray<double, 2>>
    get_h_o_diag_lapw() const;

    /// Apply first-variational LAPW Hamiltonian and overlap matrices.
    /** Check the documentation of Hamiltonain::set_fv_h_o() for the expressions of Hamiltonian and overlap
     *  matrices and \ref basis for the definition of the LAPW+lo basis.
     *
     *  For the set of wave-functions expanded in LAPW+lo basis (k-point index is dropped for simplicity)
     *  \f[
     *      \psi_{i} = \sum_{\mu} \phi_{\mu} C_{\mu i}
     *  \f]
     *  where \f$ \mu = \{ {\bf G}, j \} \f$ is a combined index of LAPW and local orbitals we want to contrusct
     *  a subspace Hamiltonian and overlap matrices:
     *  \f[
     *      H_{i' i} = \langle \psi_{i'} | \hat H | \psi_i \rangle =
     *          \sum_{\mu' \mu} C_{\mu' i'}^{*} \langle \phi_{\mu'} | \hat H | \phi_{\mu} \rangle C_{\mu i} =
     *          \sum_{\mu'} C_{\mu' i'}^{*} h_{\mu' i}(\psi)
     *  \f]
     *  \f[
     *      O_{i' i} = \langle \psi_{i'} | \psi_i \rangle =
     *          \sum_{\mu' \mu} C_{\mu' i'}^{*} \langle \phi_{\mu'} | \phi_{\mu} \rangle C_{\mu i} =
     *          \sum_{\mu'} C_{\mu' i'}^{*} o_{\mu' i}(\psi)
     *  \f]
     *  where
     *  \f[
     *      h_{\mu' i}(\psi) = \sum_{\mu} \langle \phi_{\mu'} | \hat H | \phi_{\mu} \rangle C_{\mu i}
     *  \f]
     *  and
     *  \f[
     *      o_{\mu' i}(\psi) = \sum_{\mu} \langle \phi_{\mu'} | \phi_{\mu} \rangle C_{\mu i}
     *  \f]
     *  For the APW block of \f$  h_{\mu' i}(\psi)  \f$ and \f$  o_{\mu' i}(\psi)  \f$ we have:
     *  \f[
     *       h_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} +
     *          \sum_{j} \langle \phi_{\bf G'} | \hat H | \phi_{j} \rangle C_{j i}
     *  \f]
     *  \f[
     *       o_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \phi_{\bf G} \rangle C_{{\bf G} i} +
     *          \sum_{j} \langle \phi_{\bf G'} | \phi_{j} \rangle C_{j i}
     *  \f]
     *  and for the lo block:
     *  \f[
     *       h_{j' i}(\psi) = \sum_{{\bf G}} \langle \phi_{j'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} +
     *          \sum_{j} \langle \phi_{j'} | \hat H | \phi_{j} \rangle C_{j i}
     *  \f]
     *  \f[
     *       o_{j' i}(\psi) = \sum_{{\bf G}} \langle \phi_{j'} |  \phi_{\bf G} \rangle C_{{\bf G} i} +
     *          \sum_{j} \langle \phi_{j'} | \phi_{j} \rangle C_{j i}
     *  \f]
     *
     *  APW-APW contribution, muffin-tin part:
     *  \f[
     *      h_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} =
     *          \sum_{{\bf G}} \sum_{\alpha} \sum_{\xi'} a_{\xi'}^{\alpha *}({\bf G'}) b_{\xi'}^{\alpha}({\bf G})
     *           C_{{\bf G} i}
     *  \f]
     *  \f[
     *      o_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \phi_{\bf G} \rangle C_{{\bf G} i} =
     *          \sum_{{\bf G}} \sum_{\alpha} \sum_{\xi'} a_{\xi'}^{\alpha *}({\bf G'}) a_{\xi'}^{\alpha}({\bf G})
     *           C_{{\bf G} i}
     *  \f]
     *  APW-APW contribution, interstitial effective potential part:
     *  \f[
     *      h_{{\bf G'} i}(\psi) = \int \Theta({\bf r}) e^{-i{\bf G'}{\bf r}} V({\bf r}) \psi_{i}({\bf r}) d{\bf r}
     *  \f]
     *  This is done by transforming \f$ \psi_i({\bf G}) \f$ to the real space, multiplying by effectvive potential
     *  and step function and transforming the result back to the \f$ {\bf G} \f$ domain.
     *
     *  APW-APW contribution, interstitial kinetic energy part:
     *  \f[
     *      h_{{\bf G'} i}(\psi) = \int \Theta({\bf r}) e^{-i{\bf G'}{\bf r}} \Big( -\frac{1}{2} \nabla \Big)
     *          \Big( \nabla \psi_{i}({\bf r}) \Big) d{\bf r}
     *  \f]
     *  and the gradient of the wave-function is computed with FFT as:
     *  \f[
     *      \Big( \nabla \psi_{i}({\bf r}) \Big) = \sum_{\bf G} i{\bf G}e^{i{\bf G}{\bf r}}\psi_i({\bf G})
     *  \f]
     *
     *  APW-APW contribution, interstitial overlap:
     *  \f[
     *      o_{{\bf G'} i}(\psi) = \int \Theta({\bf r}) e^{-i{\bf G'}{\bf r}} \psi_{i}({\bf r}) d{\bf r}
     *  \f]
     *
     *  APW-lo contribution:
     *  \f[
     *      h_{{\bf G'} i}(\psi) =  \sum_{j} \langle \phi_{\bf G'} | \hat H | \phi_{j} \rangle C_{j i} =
     *      \sum_{j} C_{j i}   \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'})
     *          \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}|\hat h^{\alpha_j} | \phi_{\ell_j}^{\zeta_j \alpha_j}
     * Y_{\ell_j m_j} \rangle =
     *      \sum_{j} C_{j i} \sum_{\xi'} a_{\xi'}^{\alpha_j *}({\bf G'}) h_{\xi' \xi_j}^{\alpha_j}
     *  \f]
     *  \f[
     *      o_{{\bf G'} i}(\psi) =  \sum_{j} \langle \phi_{\bf G'} | \phi_{j} \rangle C_{j i} =
     *      \sum_{j} C_{j i}   \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'})
     *          \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}| \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j} \rangle
     * =
     *      \sum_{j} C_{j i} \sum_{\nu'} a_{\ell_j m_j \nu'}^{\alpha_j *}({\bf G'}) o_{\nu' \zeta_j \ell_j}^{\alpha_j}
     *  \f]
     *  lo-APW contribution:
     *  \f[
     *     h_{j' i}(\psi) = \sum_{\bf G} \langle \phi_{j'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} =
     *      \sum_{\bf G} C_{{\bf G} i} \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'}
     * m_{j'}}
     *          |\hat h^{\alpha_{j'}} | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G}) =
     *      \sum_{\bf G} C_{{\bf G} i} \sum_{\xi} h_{\xi_{j'} \xi}^{\alpha_{j'}} a_{\xi}^{\alpha_{j'}}({\bf G})
     *  \f]
     *  \f[
     *     o_{j' i}(\psi) = \sum_{\bf G} \langle \phi_{j'} |  \phi_{\bf G} \rangle C_{{\bf G} i} =
     *      \sum_{\bf G} C_{{\bf G} i} \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'}
     * m_{j'}}
     *          | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G}) =
     *      \sum_{\bf G} C_{{\bf G} i} \sum_{\nu} o_{\zeta_{j'} \nu \ell_{j'}}^{\alpha_{j'}} a_{\ell_{j'} m_{j'}
     * \nu}^{\alpha_{j'}}({\bf G})
     *  \f]
     *  lo-lo contribution:
     *  \f[
     *      h_{j' i}(\psi) = \sum_{j} \langle \phi_{j'} | \hat H | \phi_{j} \rangle C_{j i} = \sum_{j} C_{j i}
     * h_{\xi_{j'} \xi_j}^{\alpha_j}
     *          \delta_{\alpha_j \alpha_{j'}}
     *  \f]
     *  \f[
     *      o_{j' i}(\psi) = \sum_{j} \langle \phi_{j'} | \phi_{j} \rangle C_{j i} = \sum_{j} C_{j i}
     *          o_{\zeta_{j'} \zeta_{j} \ell_j}^{\alpha_j}
     *            \delta_{\alpha_j \alpha_{j'}} \delta_{\ell_j \ell_{j'}} \delta_{m_j m_{j'}}
     *  \f]
     *
     *  \param [in]  apw_only   True if only APW-APW block of H and O are applied.
     *  \param [in]  phi_is_lo  True if input wave-functions are pure local orbitals.
     *  \param [in]  N          Starting index of wave-functions.
     *  \param [in]  n          Number of wave-functions to which H and S are applied.
     *  \param [in]  phi        Input wave-functions.
     *  \param [out] hphi       Result of Hamiltonian, applied to wave-functions.
     *  \param [out] ophi       Result of overlap operator, applied to wave-functions.
     */
    void apply_fv_h_o(bool apw_only__, bool phi_is_lo__, int N__, int n__, sddk::Wave_functions& phi__,
                      sddk::Wave_functions* hphi__, sddk::Wave_functions* ophi__);

    /// Setup the Hamiltonian and overlap matrices in APW+lo basis
    /** The Hamiltonian matrix has the following expression:
     *  \f[
     *      H_{\mu' \mu}=\langle \varphi_{\mu' } | \hat H | \varphi_{\mu } \rangle  =
     *      \left( \begin{array}{cc}
     *         H_{\bf G'G} & H_{{\bf G'}j} \\
     *         H_{j'{\bf G}} & H_{j'j}
     *      \end{array} \right)
     *  \f]
     *  APW-APW block:
     *  \f{eqnarray*}{
     *      H_{{\bf G'} {\bf G}}^{\bf k} &=& \sum_{\alpha} \sum_{L'\nu', L\nu} a_{L'\nu'}^{\alpha *}({\bf G'+k})
     *      \langle  u_{\ell' \nu'}^{\alpha}Y_{\ell' m'}|\hat h^{\alpha} | u_{\ell \nu}^{\alpha}Y_{\ell m}  \rangle
     *       a_{L\nu}^{\alpha}({\bf G+k}) + \frac{1}{2}{\bf G'} {\bf G} \cdot \Theta({\bf G - G'}) + \tilde V_{eff}({\bf
     * G - G'}) \\
     *          &=& \sum_{\alpha} \sum_{\xi' } a_{\xi'}^{\alpha *}({\bf G'+k})
     *              b_{\xi'}^{\alpha}({\bf G+k}) + \frac{1}{2}{\bf G'} {\bf G} \cdot \Theta({\bf G - G'}) + \tilde
     * V_{eff}({\bf G - G'})
     *  \f}
     *  APW-lo block:
     *  \f[
     *      H_{{\bf G'} j}^{\bf k} = \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'+k})
     *      \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}|\hat h^{\alpha_j} |  \phi_{\ell_j}^{\zeta_j \alpha_j}
     * Y_{\ell_j m_j}  \rangle
     *  \f]
     *  lo-APW block:
     *  \f[
     *      H_{j' {\bf G}}^{\bf k} = \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}}
     *          |\hat h^{\alpha_{j'}} | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G+k})
     *  \f]
     *  lo-lo block:
     *  \f[
     *      H_{j' j}^{\bf k} = \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}}
     *          |\hat h^{\alpha_{j}} |  \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j}  \rangle  \delta_{\alpha_j
     * \alpha_{j'}}
     *  \f]
     *
     *  The overlap matrix has the following expression:
     *  \f[
     *      O_{\mu' \mu} = \langle \varphi_{\mu'} | \varphi_{\mu} \rangle
     *  \f]
     *  APW-APW block:
     *  \f[
     *      O_{{\bf G'} {\bf G}}^{\bf k} = \sum_{\alpha} \sum_{L\nu} a_{L\nu}^{\alpha *}({\bf G'+k})
     *      a_{L\nu}^{\alpha}({\bf G+k}) + \Theta({\bf G-G'})
     *  \f]
     *
     *  APW-lo block:
     *  \f[
     *      O_{{\bf G'} j}^{\bf k} = \sum_{\nu'} a_{\ell_j m_j \nu'}^{\alpha_j *}({\bf G'+k})
     *      \langle u_{\ell_j \nu'}^{\alpha_j} | \phi_{\ell_j}^{\zeta_j \alpha_j} \rangle
     *  \f]
     *
     *  lo-APW block:
     *  \f[
     *      O_{j' {\bf G}}^{\bf k} =
     *      \sum_{\nu'} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} | u_{\ell_{j'} \nu'}^{\alpha_{j'}} \rangle
     *      a_{\ell_{j'} m_{j'} \nu'}^{\alpha_{j'}}({\bf G+k})
     *  \f]
     *
     *  lo-lo block:
     *  \f[
     *      O_{j' j}^{\bf k} = \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} |
     *      \phi_{\ell_{j}}^{\zeta_{j} \alpha_{j}} \rangle \delta_{\alpha_{j'} \alpha_j}
     *      \delta_{\ell_{j'} \ell_j} \delta_{m_{j'} m_j}
     *  \f]
     */
    void set_fv_h_o(sddk::dmatrix<double_complex>& h__, sddk::dmatrix<double_complex>& o__) const;

    /// Add interstitial contribution to apw-apw block of Hamiltonian and overlap.
    void set_fv_h_o_it(sddk::dmatrix<double_complex>& h__, sddk::dmatrix<double_complex>& o__) const;

    /// Setup lo-lo block of Hamiltonian and overlap matrices.
    void set_fv_h_o_lo_lo(sddk::dmatrix<double_complex>& h__, sddk::dmatrix<double_complex>& o__) const;

    /// Setup apw-lo and lo-apw blocks of LAPW Hamiltonian and overlap matrices.
    void set_fv_h_o_apw_lo(Atom const& atom, int ia, sddk::mdarray<double_complex, 2>& alm_row,
                           sddk::mdarray<double_complex, 2>& alm_col, sddk::mdarray<double_complex, 2>& h,
                           sddk::mdarray<double_complex, 2>& o) const;

    /// Apply pseudopotential H and S operators to the wavefunctions.
    /** \param [in]  spins Spin index range
     *  \param [in]  N     Starting index of wave-functions.
     *  \param [in]  n     Number of wave-functions to which H and S are applied.
     *  \param [in]  phi   Input wave-functions [storage: CPU && GPU].
     *  \param [out] hphi  Result of Hamiltonian, applied to wave-functions [storage: CPU || GPU].
     *  \param [out] sphi  Result of S-operator, applied to wave-functions [storage: CPU || GPU].
     *
     *  In non-collinear case (spins in [0,1]) the Hamiltonian and S operator are applied to both components of spinor
     *  wave-functions. Otherwise they are applied to a single component.
     */
    template <typename T>
    void apply_h_s(sddk::spin_range spins__, int N__, int n__, sddk::Wave_functions& phi__, sddk::Wave_functions* hphi__,
                   sddk::Wave_functions* sphi__);

    /// Apply magnetic field to first-variational LAPW wave-functions.
    void apply_b(sddk::Wave_functions& psi__, std::vector<sddk::Wave_functions>& bpsi__);
};

Hamiltonian_k Hamiltonian0::operator()(K_point& kp__)
{
    return Hamiltonian_k(*this, kp__);
}

}

#endif
