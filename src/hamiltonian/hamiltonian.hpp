/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file hamiltonian.hpp
 *
 *  \brief Contains declaration and definition of sirius::Hamiltonian class.
 */

#ifndef __HAMILTONIAN_HPP__
#define __HAMILTONIAN_HPP__

#include <memory>
#include <complex>
#include "core/wf/wave_functions.hpp"
#include "core/typedefs.hpp"
#include "core/fft/fft.hpp"
#include "core/la/dmatrix.hpp"
#include "local_operator.hpp"
#include "non_local_operator.hpp"

namespace sirius {
/* forward declaration */
class Atom;
class Simulation_context;
class Potential;
class Unit_cell;
template <typename T>
class Local_operator;
template <typename T>
class D_operator;
template <typename T>
class Q_operator;
template <typename T>
class U_operator;
template <typename T>
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
template <typename T>
class Hamiltonian_k;

/// Represent the k-point independent part of Hamiltonian.
/** \tparam T   Precision of the wave-functions (float or double).
 */
template <typename T> // type is real type precision
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
    std::unique_ptr<Local_operator<T>> local_op_;

    /// D operator (non-local part of Hamiltonian).
    std::unique_ptr<D_operator<T>> d_op_;

    /// Q operator (non-local part of S-operator).
    std::unique_ptr<Q_operator<T>> q_op_;

    std::vector<mdarray<std::complex<T>, 2>> hmt_;

    /* copy constructor is forbidden */
    Hamiltonian0(Hamiltonian0<T> const& src) = delete;
    /* copy assignment operator is forbidden */
    Hamiltonian0<T>&
    operator=(Hamiltonian0<T> const& src) = delete;

  public:
    /// Constructor.
    Hamiltonian0(Potential& potential__, bool precompute_lapw__, bool update_lapw_rf__ = true);

    ~Hamiltonian0();

    /// Default move constructor.
    Hamiltonian0(Hamiltonian0<T>&& src) = default;

    /// Return a Hamiltonian for the given k-point.
    inline Hamiltonian_k<T>
    operator()(K_point<T>& kp__) const;

    Simulation_context&
    ctx() const
    {
        return ctx_;
    }

    Potential&
    potential() const
    {
        return *potential_;
    }

    Local_operator<T>&
    local_op() const
    {
        return *local_op_;
    }

    inline Q_operator<T>&
    Q() const
    {
        return *q_op_;
    }

    inline D_operator<T>&
    D() const
    {
        return *d_op_;
    }

    auto const&
    hmt(int ia__) const
    {
        return hmt_[ia__];
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
    void
    apply_hmt_to_apw(Atom const& atom__, spin_block_t sblock__, int ngv__, mdarray<std::complex<T>, 2>& alm__,
                     mdarray<std::complex<T>, 2>& halm__) const;

    /// Add correction to LAPW overlap arising in the infinite-order relativistic approximation (IORA).
    void
    add_o1mt_to_apw(Atom const& atom__, int num_gkvec__,
                    mdarray<std::complex<T>, 2>& alm__) const; // TODO: documentation

    /// Apply muffin-tin part of magnetic filed to the wave-functions.
    void
    apply_bmt(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& bpsi__) const;

    /// Apply SO correction to the first-variational LAPW wave-functions.
    /** Raising and lowering operators:
     *  \f[
     *      L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
     *  \f]
     */
    void
    apply_so_correction(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& hpsi__) const;
};

template <typename T>
class Hamiltonian_k
{
  private:
    /// K-point independent part of Hamiltonian.
    Hamiltonian0<T> const& H0_;
    K_point<T>& kp_;
    /// Hubbard correction.
    /** In general case it is a k-dependent matrix */
    std::shared_ptr<U_operator<T>> u_op_;

    /// Copy constructor is forbidden.
    Hamiltonian_k(Hamiltonian_k<T> const& src__) = delete;

    /// Assignment operator is forbidden.
    Hamiltonian_k<T>&
    operator=(Hamiltonian_k<T> const& src__) = delete;

  public:
    Hamiltonian_k(Hamiltonian0<T> const& H0__, K_point<T>& kp__);

    Hamiltonian_k(Hamiltonian_k<T>&& src__);

    ~Hamiltonian_k();

    Hamiltonian0<T> const&
    H0() const
    {
        return H0_;
    }

    template <typename F, int what>
    std::pair<mdarray<T, 2>, mdarray<T, 2>>
    get_h_o_diag_pw() const;

    template <int what>
    std::pair<mdarray<T, 2>, mdarray<T, 2>>
    get_h_o_diag_lapw() const;

    auto
    U() const -> U_operator<T> const&
    {
        return *u_op_;
    }

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
     *  \param [in]  b          Range of bands.
     *  \param [in]  phi        Input wave-functions.
     *  \param [out] hphi       Result of Hamiltonian, applied to wave-functions.
     *  \param [out] ophi       Result of overlap operator, applied to wave-functions.
     */
    void
    apply_fv_h_o(bool apw_only__, bool phi_is_lo__, wf::band_range b__, wf::Wave_functions<T>& phi__,
                 wf::Wave_functions<T>* hphi__, wf::Wave_functions<T>* ophi__) const;

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
     *
     *  For IORA there is additional term that arises after linearisation of 1/M over energy:
     *  \f[
     *    \frac{1}{M} = \frac{1}{M_0} - \frac{\alpha^2}{2} \frac{1}{M_0^2} E
     *  \f]
     *
     *  If we now insert it into kinetic energy \f$ \hat {\bf P} \frac{1}{2M} \hat {\bf P} \f$ we obtain
     *  correction linear to energy which has to be added to the overlap matrix:
     *  \f[
     *   \hat {\bf P} \frac{\alpha^2}{2} \frac{1}{2M_0^2} \hat {\bf P}
     *  \f]
     */
    void
    set_fv_h_o(la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__) const;

    /// Add interstitial contribution to apw-apw block of Hamiltonian and overlap.
    void
    set_fv_h_o_it(la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__) const;

    /// Setup lo-lo block of Hamiltonian and overlap matrices.
    void
    set_fv_h_o_lo_lo(la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__) const;

    /// Setup apw-lo and lo-apw blocks of LAPW Hamiltonian and overlap matrices.
    void
    set_fv_h_o_apw_lo(Atom const& atom, int ia, mdarray<std::complex<T>, 2>& alm_row,
                      mdarray<std::complex<T>, 2>& alm_col, mdarray<std::complex<T>, 2>& h,
                      mdarray<std::complex<T>, 2>& o) const;

    /// Apply pseudopotential H and S operators to the wavefunctions.
    /** \tparam F    Type of the subspace matrix.
     *  \param [in]  spins Range of spins.
     *  \param [in]  br    Range of bands.
     *  \param [in]  phi   Input wave-functions [storage: CPU && GPU].
     *  \param [out] hphi  Result of Hamiltonian, applied to wave-functions [storage: CPU || GPU].
     *  \param [out] sphi  Result of S-operator, applied to wave-functions [storage: CPU || GPU].
     *
     *  In non-collinear case (spins in [0,1]) the Hamiltonian and S operator are applied to both components of spinor
     *  wave-functions. Otherwise they are applied to a single component.
     */
    template <typename F>
    std::enable_if_t<std::is_same<T, real_type<F>>::value, void>
    apply_h_s(wf::spin_range spins__, wf::band_range br__, wf::Wave_functions<T> const& phi__,
              wf::Wave_functions<T>* hphi__, wf::Wave_functions<T>* sphi__) const
    {
        PROFILE("sirius::Hamiltonian_k::apply_h_s");

        auto pcs = env::print_checksum();

        if (hphi__ != nullptr) {
            /* apply local part of Hamiltonian */
            H0().local_op().apply_h(reinterpret_cast<fft::spfft_transform_type<T>&>(kp_.spfft_transform()),
                                    kp_.gkvec_fft_sptr(), spins__, phi__, *hphi__, br__);
        }

        auto mem = H0().ctx().processing_unit_memory_t();

        if (pcs) {
            auto cs = phi__.checksum(mem, br__);
            print_checksum("phi", cs, RTE_OUT(H0().ctx().out()));
            if (hphi__) {
                auto cs = hphi__->checksum(mem, br__);
                print_checksum("hloc_phi", cs, RTE_OUT(H0().ctx().out()));
            }
        }

        /* set initial sphi */
        if (sphi__ != nullptr) {
            for (auto s = spins__.begin(); s != spins__.end(); s++) {
                auto sp = phi__.actual_spin_index(s);
                wf::copy(mem, phi__, sp, br__, *sphi__, sp, br__);
            }
        }

        /* return if there are no beta-projectors */
        if (H0().ctx().unit_cell().max_mt_basis_size()) {
            auto bp_generator = kp_.beta_projectors().make_generator();
            auto beta_coeffs  = bp_generator.prepare();
            apply_non_local_D_Q<T, F>(mem, spins__, br__, bp_generator, beta_coeffs, phi__, &H0().D(), hphi__,
                                      &H0().Q(), sphi__);
        }

        /* apply the hubbard potential if relevant */
        if (H0().ctx().hubbard_correction() && !H0().ctx().gamma_point() && hphi__) {
            /* apply the hubbard potential */
            apply_U_operator(H0().ctx(), spins__, br__, kp_.hubbard_wave_functions_S(), phi__, this->U(), *hphi__);
        }

        if (pcs) {
            if (hphi__) {
                auto cs = hphi__->checksum(mem, br__);
                print_checksum("hphi", cs, RTE_OUT(H0().ctx().out()));
            }
            if (sphi__) {
                auto cs = sphi__->checksum(mem, br__);
                print_checksum("hsphi", cs, RTE_OUT(H0().ctx().out()));
            }
        }
    }

    template <typename F>
    std::enable_if_t<!std::is_same<T, real_type<F>>::value, void>
    apply_h_s(wf::spin_range spins__, wf::band_range br__, wf::Wave_functions<T> const& phi__,
              wf::Wave_functions<T>* hphi__, wf::Wave_functions<T>* sphi__) const
    {
        RTE_THROW("implement this");
    }

    /// apply S operator
    /** \tparam F    Type of the subspace matrix.
     *  \param [in]  spins Range of spins.
     *  \param [in]  br    Range of bands.
     *  \param [in]  phi   Input wave-functions [storage: CPU && GPU].
     *  \param [out] sphi  Result of S-operator, applied to wave-functions [storage: CPU || GPU].
     *
     *  In non-collinear case (spins in [0,1]) the S operator is applied to both components of spinor
     *  wave-functions. Otherwise they are applied to a single component.
     */
    template <typename F>
    std::enable_if_t<std::is_same<T, real_type<F>>::value, void>
    apply_s(wf::spin_range spin__, wf::band_range br__, wf::Wave_functions<T> const& phi__,
            wf::Wave_functions<T>& sphi__) const
    {
        auto mem       = H0().ctx().processing_unit_memory_t();
        auto bp_gen    = kp_.beta_projectors().make_generator();
        auto bp_coeffs = bp_gen.prepare();
        apply_S_operator<T, F>(mem, spin__, br__, bp_gen, bp_coeffs, phi__, &H0().Q(), sphi__);
    }

    /// Apply magnetic field to first-variational LAPW wave-functions.
    void
    apply_b(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& bpsi__) const;
};

template <typename T>
Hamiltonian_k<T>
Hamiltonian0<T>::operator()(K_point<T>& kp__) const
{
    return Hamiltonian_k<T>(*this, kp__);
}

} // namespace sirius

#endif
