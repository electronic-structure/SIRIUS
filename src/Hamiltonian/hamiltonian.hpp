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

#include <typeinfo>
#include "simulation_context.hpp"
#include "Hubbard/hubbard.hpp"
#include "Potential/potential.hpp"
#include "K_point/k_point.hpp"
#include "local_operator.hpp"
#include "non_local_operator.hpp"

namespace sirius {

/// Representation of Kohn-Sham Hamiltonian.
/** In general, Hamiltonian consists of kinetic term, local part of potential and non-local
    part of potential:
    \f[
      H = -\frac{1}{2} \nabla^2 + V_{loc}({\bf r}) + \sum_{\alpha} \sum_{\xi \xi'} |\beta_{\xi}^{\alpha} \rangle
        D_{\xi \xi'}^{\alpha} \langle \beta_{\xi'}^{\alpha}|
    \f]
*/
class Hamiltonian
{
  private:
    /// Simulation context.
    Simulation_context& ctx_;

    /// Alias for the potential.
    Potential& potential_;

    /// Alias for unit cell.
    Unit_cell& unit_cell_;

    /// Alias for the hubbard potential (note it is a pointer)
    std::unique_ptr<Hubbard> U_;

    /// Local part of the Hamiltonian operator.
    std::unique_ptr<Local_operator> local_op_;

    /// Non-zero Gaunt coefficients
    std::unique_ptr<Gaunt_coefficients<double_complex>> gaunt_coefs_;

    /// D operator (non-local part of Hamiltonian).
    std::unique_ptr<D_operator> d_op_;

    /// Q operator (non-local part of S-operator).
    std::unique_ptr<Q_operator> q_op_;

  public:
    /// Constructor.
    Hamiltonian(Simulation_context& ctx__, Potential& potential__)
        : ctx_(ctx__)
        , potential_(potential__)
        , unit_cell_(ctx__.unit_cell())
    {
        if (ctx_.full_potential()) {
            using gc_z = Gaunt_coefficients<double_complex>;
            gaunt_coefs_ = std::unique_ptr<gc_z>(new gc_z(ctx_.lmax_apw(), ctx_.lmax_pot(), ctx_.lmax_apw(), SHT::gaunt_hybrid));
        }

        local_op_ = std::unique_ptr<Local_operator>(new Local_operator(ctx_, ctx_.fft_coarse(), ctx_.gvec_coarse_partition()));

        if (ctx_.hubbard_correction()) {
            U_ = std::unique_ptr<Hubbard>(new Hubbard(ctx_));
        }
    }

    Hubbard& U() const
    {
        return *U_;
    }

    Potential& potential() const
    {
        return potential_;
    }

    Simulation_context& ctx() const
    {
        return ctx_;
    }

    Local_operator& local_op() const
    {
        return *local_op_;
    }

    /// Prepare k-point independent operators.
    inline void prepare()
    {
        if (!ctx_.full_potential()) {
            d_op_ = std::unique_ptr<D_operator>(new D_operator(ctx_));
            q_op_ = std::unique_ptr<Q_operator>(new Q_operator(ctx_));
        }
        local_op().prepare(potential_);
    }

    inline void dismiss()
    {
        d_op_ = nullptr;
        q_op_ = nullptr;
        local_op().dismiss();
    }

    inline Q_operator& Q() const
    {
        return *q_op_;
    }

    inline D_operator& D() const
    {
        return *d_op_;
    }

    /// Apply pseudopotential H and S operators to the wavefunctions.
    template <typename T>
    inline void apply_h_s(K_point*        kp__,
                          int             ispn__,
                          int             N__,
                          int             n__,
                          Wave_functions& phi__,
                          Wave_functions* hphi__,
                          Wave_functions* sphi__) const;

    /// Apply magnetic field to first-variational LAPW wave-functions.
    inline void apply_magnetic_field(K_point*                     kp__,
                                     Wave_functions&              fv_states__,
                                     std::vector<Wave_functions>& hpsi__) const;

    /// Apply SO correction to the first-variational LAPW wave-functions.
    /** Raising and lowering operators:
     *  \f[
     *      L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
     *  \f]
     */
    void apply_so_correction(K_point* kp__, Wave_functions& fv_states__, std::vector<Wave_functions>& hpsi__) const;

    /// Apply UJ correction to scalar wave functions
    template <spin_block_t sblock>
    void apply_uj_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi);

    /// Add interstitial contribution to apw-apw block of Hamiltonian and overlap
    inline void set_fv_h_o_it(K_point* kp__, matrix<double_complex>& h__, matrix<double_complex>& o__) const;

    inline void set_o_it(K_point* kp, mdarray<double_complex, 2>& o) const;

    template <spin_block_t sblock>
    inline void set_h_it(K_point* kp,
                         Periodic_function<double>* effective_potential,
                         Periodic_function<double>* effective_magnetic_field[3],
                         matrix<double_complex>& h) const;

    /// Setup lo-lo block of Hamiltonian and overlap matrices
    inline void set_fv_h_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o) const;

    template <spin_block_t sblock>
    inline void set_h_lo_lo(K_point* kp, mdarray<double_complex, 2>& h) const;

    inline void set_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& o) const;

    inline void set_o(K_point* kp, mdarray<double_complex, 2>& o);

    template <spin_block_t sblock>
    inline void set_h(K_point* kp,
                      Periodic_function<double>* effective_potential,
                      Periodic_function<double>* effective_magnetic_field[3],
                      mdarray<double_complex, 2>& h);

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
    void apply_hmt_to_apw(Atom const& atom__,
                          int num_gkvec__,
                          mdarray<double_complex, 2>& alm__,
                          mdarray<double_complex, 2>& halm__) const
    {
        auto& type = atom__.type();

        // TODO: this is k-independent and can in principle be precomputed together with radial integrals if memory is
        // available
        // TODO: for spin-collinear case hmt is Hermitian; compute upper triangular part and use zhemm
        mdarray<double_complex, 2> hmt(type.mt_aw_basis_size(), type.mt_aw_basis_size());
        /* compute the muffin-tin Hamiltonian */
        for (int j2 = 0; j2 < type.mt_aw_basis_size(); j2++) {
            int lm2    = type.indexb(j2).lm;
            int idxrf2 = type.indexb(j2).idxrf;
            for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
                int lm1    = type.indexb(j1).lm;
                int idxrf1 = type.indexb(j1).idxrf;
                hmt(j1, j2) =
                    atom__.radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));
            }
        }
        linalg<device_t::CPU>::gemm(0, 1, num_gkvec__, type.mt_aw_basis_size(), type.mt_aw_basis_size(), alm__, hmt, halm__);
    }

    void apply_o1mt_to_apw(Atom const& atom__,
                           int num_gkvec__,
                           mdarray<double_complex, 2>& alm__,
                           mdarray<double_complex, 2>& oalm__) const
    {
        auto& type = atom__.type();

        for (int j = 0; j < type.mt_aw_basis_size(); j++) {
            int l     = type.indexb(j).l;
            int lm    = type.indexb(j).lm;
            int idxrf = type.indexb(j).idxrf;
            for (int order = 0; order < type.aw_order(l); order++) {
                int j1     = type.indexb().index_by_lm_order(lm, order);
                int idxrf1 = type.indexr().index_by_l_order(l, order);
                for (int ig = 0; ig < num_gkvec__; ig++) {
                    oalm__(ig, j) += atom__.symmetry_class().o1_radial_integral(idxrf, idxrf1) * alm__(ig, j1);
                }
            }
        }
    }

    /// Setup apw-lo and lo-apw blocs of Hamiltonian and overlap matrices
    void set_fv_h_o_apw_lo(K_point* kp,
                           Atom_type const& type,
                           Atom const& atom,
                           int ia,
                           mdarray<double_complex, 2>& alm_row,
                           mdarray<double_complex, 2>& alm_col,
                           mdarray<double_complex, 2>& h,
                           mdarray<double_complex, 2>& o) const;

    template <spin_block_t sblock>
    void set_h_apw_lo(K_point* kp,
                      Atom_type* type,
                      Atom* atom,
                      int ia,
                      mdarray<double_complex, 2>& alm,
                      mdarray<double_complex, 2>& h);

    /// Set APW-lo and lo-APW blocks of the overlap matrix.
    void set_o_apw_lo(K_point* kp,
                      Atom_type* type,
                      Atom* atom,
                      int ia,
                      mdarray<double_complex, 2>& alm,
                      mdarray<double_complex, 2>& o);

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
    template <device_t pu, electronic_structure_method_t basis>
    inline void set_fv_h_o(K_point* kp, dmatrix<double_complex>& h, dmatrix<double_complex>& o) const;

    /// Apply LAPW Hamiltonain and overlap to the trial wave-functions.
    /** Check the documentation of Band::set_fv_h_o() for the expressions of Hamiltonian and overlap
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
     */
    inline void apply_fv_h_o(K_point*        kp__,
                             bool            apw_only__,
                             bool            phi_is_lo__,
                             int             N__,
                             int             n__,
                             Wave_functions& phi__,
                             Wave_functions* hphi__,
                             Wave_functions* ophi__) const;

    /// Get diagonal elements of LAPW Hamiltonian.
    inline mdarray<double, 2> get_h_diag(K_point* kp__, double v0__, double theta0__) const;

    /// Get diagonal elements of pseudopotential Hamiltonian.
    template <typename T>
    inline mdarray<double, 2> get_h_diag(K_point* kp__) const;

    /// Get diagonal elements of LAPW overlap.
    inline mdarray<double, 1> get_o_diag(K_point* kp__, double theta0__) const;

    /// Get diagonal elements of pseudopotential overlap matrix.
    template <typename T>
    inline mdarray<double, 1> get_o_diag(K_point* kp__) const;
};

/* forward declaration */
class Hamiltonian_k;

class Hamiltonian0
{
  private:
    /// Simulation context.
    Simulation_context& ctx_;

    /// Alias for the potential.
    Potential& potential_;

    /// Alias for unit cell.
    Unit_cell& unit_cell_;

    /// Alias for the hubbard potential (note it is a pointer)
    std::unique_ptr<Hubbard> U_;

    /// Local part of the Hamiltonian operator.
    std::unique_ptr<Local_operator> local_op_;

    /// Non-zero Gaunt coefficients
    std::unique_ptr<Gaunt_coefficients<double_complex>> gaunt_coefs_;

    /// D operator (non-local part of Hamiltonian).
    std::unique_ptr<D_operator> d_op_;

    /// Q operator (non-local part of S-operator).
    std::unique_ptr<Q_operator> q_op_;

    Hamiltonian0(Hamiltonian0 const& src) = delete;
    Hamiltonian0& operator=(Hamiltonian0 const& src) = delete;

  public:
    /// Constructor.
    Hamiltonian0(Simulation_context& ctx__, Potential& potential__)
        : ctx_(ctx__)
        , potential_(potential__)
        , unit_cell_(ctx__.unit_cell())
    {
        PROFILE("sirius::Hamiltonian0");

        if (ctx_.full_potential()) {
            using gc_z = Gaunt_coefficients<double_complex>;
            gaunt_coefs_ = std::unique_ptr<gc_z>(new gc_z(ctx_.lmax_apw(), ctx_.lmax_pot(), ctx_.lmax_apw(), SHT::gaunt_hybrid));
        }

        local_op_ = std::unique_ptr<Local_operator>(new Local_operator(ctx_, ctx_.fft_coarse(), ctx_.gvec_coarse_partition()));

        if (ctx_.hubbard_correction()) {
            U_ = std::unique_ptr<Hubbard>(new Hubbard(ctx_));
        }

        if (!ctx_.full_potential()) {
            d_op_ = std::unique_ptr<D_operator>(new D_operator(ctx_));
            q_op_ = std::unique_ptr<Q_operator>(new Q_operator(ctx_));
        }
        local_op_->prepare(potential_);
    }

    ~Hamiltonian0()
    {
        local_op_->dismiss();
    }

    Hamiltonian0(Hamiltonian0&& src) = default;

    inline Hamiltonian_k operator()(K_point& kp__);

    //Potential& potential() const
    //{
    //    return potential_;
    //}

    Simulation_context& ctx() const
    {
        return ctx_;
    }

    Local_operator& local_op() const
    {
        return *local_op_;
    }

    inline Gaunt_coefficients<double_complex> const& gaunt_coefs() const
    {
        return *gaunt_coefs_;
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
    void apply_hmt_to_apw(Atom const& atom__, int ngv__, mdarray<double_complex, 2>& alm__,
                          mdarray<double_complex, 2>& halm__) const
    {
        auto& type = atom__.type();

        // TODO: this is k-independent and can in principle be precomputed together with radial integrals if memory is
        // available
        // TODO: for spin-collinear case hmt is Hermitian; compute upper triangular part and use zhemm
        mdarray<double_complex, 2> hmt(type.mt_aw_basis_size(), type.mt_aw_basis_size());
        /* compute the muffin-tin Hamiltonian */
        for (int j2 = 0; j2 < type.mt_aw_basis_size(); j2++) {
            int lm2    = type.indexb(j2).lm;
            int idxrf2 = type.indexb(j2).idxrf;
            for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
                int lm1    = type.indexb(j1).lm;
                int idxrf1 = type.indexb(j1).idxrf;
                hmt(j1, j2) =
                    atom__.radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));
            }
        }
        linalg<::sddk::device_t::CPU>::gemm(0, 1, ngv__, type.mt_aw_basis_size(), type.mt_aw_basis_size(), alm__, hmt, halm__);
    }

    inline Q_operator& Q() const
    {
        return *q_op_;
    }

    inline D_operator& D() const
    {
        return *d_op_;
    }
};

class Hamiltonian_k
{
  private:
    Hamiltonian0& H0_;
    K_point& kp_;

    inline mdarray<double, 2> get_h_diag_lapw() const;

    //template <typename T>
    //inline mdarray<double, 2> get_h_diag_pw() const;

    inline mdarray<double, 2> get_o_diag_lapw() const;

    //template <typename T>
    //inline mdarray<double, 2> get_o_diag_pw() const;

    /// Copy constructor is forbidden.
    Hamiltonian_k(Hamiltonian_k const& src__) = delete;

    /// Assignment operator is forbidden.
    Hamiltonian_k& operator=(Hamiltonian_k const& src__) = delete;

  public:
    Hamiltonian_k(Hamiltonian0& H0__, K_point& kp__)
        : H0_(H0__)
        , kp_(kp__)
    {
        PROFILE("sirius::Hamiltonian_k");
        H0_.local_op().prepare(kp_.gkvec_partition());
        H0_.ctx().fft_coarse().prepare(kp_.gkvec_partition());
        kp_.beta_projectors().prepare();
    }

    ~Hamiltonian_k()
    {
        kp_.beta_projectors().dismiss();
        H0_.ctx().fft_coarse().dismiss();
    }

    Hamiltonian0 const& H0() const
    {
        return H0_;
    }

    K_point& kp()
    {
        return kp_;
    }

    Hamiltonian_k(Hamiltonian_k&& src__) = default;

    template <typename T, int what>
    inline std::pair<mdarray<double, 2>, mdarray<double, 2>> get_h_o_diag_pw() const;

    /// Apply pseudopotential H and S operators to the wavefunctions.
    template <typename T>
    inline void apply_h_s(int ispn__, int N__, int n__, Wave_functions& phi__, Wave_functions* hphi__,
                          Wave_functions* sphi__);

    //template <typename T>
    //inline mdarray<double, 2> get_h_diag() const
    //{
    //    if (H0_.ctx().full_potential()) {
    //        return get_h_diag_lapw();
    //    } else {
    //        return get_h_diag_pw<T>();
    //    }
    //}

    //template <typename T>
    //inline mdarray<double, 2> get_o_diag() const
    //{
    //    if (H0_.ctx().full_potential()) {
    //        return get_o_diag_lapw();
    //    } else {
    //        return get_o_diag_pw<T>();
    //    }
    //}
};

inline Hamiltonian_k Hamiltonian0::operator()(K_point& kp__)
{
    return Hamiltonian_k(*this, kp__);
}

#include "apply.hpp"
#include "set_lapw_h_o.hpp"
#include "get_h_o_diag.hpp"

}

#endif
