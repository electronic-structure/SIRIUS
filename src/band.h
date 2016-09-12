// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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
#include "k_point.h"
#include "non_local_operator.h"
#include "hloc_operator.h"
#include "potential.h"
#include "k_set.h"

namespace sirius
{

// TODO: Band problem is a mess and needs more formal organizaiton. We have different basis functions. 
//       We can do first- and second-variation or a full variation. We can do iterative or exact diagonalization.
//       This has to be organized. 

/// Setup and solve eigen value problem.
class Band
{
    private:

        /// Simulation context.
        Simulation_context& ctx_;

        /// Alias for the unit cell.
        Unit_cell& unit_cell_;

        /// BLACS grid for distributed linear algebra operations.
        BLACS_grid const& blacs_grid_;

        /// Non-zero Gaunt coefficients
        Gaunt_coefficients<double_complex>* gaunt_coefs_;
        
        /// Interface to a standard eigen-value solver.
        Eigenproblem* std_evp_solver_; 

        /// Interface to a generalized eigen-value solver.
        Eigenproblem* gen_evp_solver_;

        /// Apply effective magentic field to the first-variational state.
        /** Must be called first because hpsi is overwritten with B|fv_j>. */
        void apply_magnetic_field(wave_functions& fv_states__,
                                  Gvec const& gkvec__,
                                  Periodic_function<double>* effective_magnetic_field__[3],
                                  std::vector<wave_functions*>& hpsi__) const;

        /// Apply SO correction to the first-variational states.
        /** Raising and lowering operators:
         *  \f[
         *      L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
         *  \f]
         */
        void apply_so_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi);
        
        /// Apply UJ correction to scalar wave functions
        template <spin_block_t sblock>
        void apply_uj_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi);

        /// Add interstitial contribution to apw-apw block of Hamiltonian and overlap
        inline void set_fv_h_o_it(K_point* kp,
                                  Periodic_function<double>* effective_potential, 
                                  matrix<double_complex>& h,
                                  matrix<double_complex>& o) const;

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
       
        /// Diagonalize a full-potential Hamiltonian.
        void diag_fv_full_potential(K_point* kp__,
                                    Periodic_function<double>* effective_potential__) const;

        void diag_fv_full_potential_exact(K_point* kp__,
                                          Periodic_function<double>* effective_potential__) const;

        void diag_fv_full_potential_davidson(K_point* kp__,
                                             Periodic_function<double>* effective_potential__) const;

        /// Diagonalize a pseudo-potential Hamiltonian.
        template <typename T>
        void diag_pseudo_potential(K_point* kp__, 
                                   Periodic_function<double>* effective_potential__,
                                   Periodic_function<double>* effective_magnetic_field__[3]) const
        {
            PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential");

            Hloc_operator hloc(ctx_.fft_coarse(), kp__->gkvec_vloc(), ctx_.mpi_grid_fft_vloc().communicator(1 << 1),
                               ctx_.num_mag_dims(), ctx_.gvec_coarse(), effective_potential__, effective_magnetic_field__);
            
            ctx_.fft_coarse().prepare(kp__->gkvec().partition());

            D_operator<T> d_op(ctx_, kp__->beta_projectors());
            Q_operator<T> q_op(ctx_, kp__->beta_projectors());

            auto& itso = ctx_.iterative_solver_input_section();
            if (itso.type_ == "exact") {
                if (ctx_.num_mag_dims() != 3) {
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        diag_pseudo_potential_exact(kp__, ispn, hloc, d_op, q_op);
                    }
                } else {
                    STOP();
                }
            } else if (itso.type_ == "davidson") {
                if (ctx_.num_mag_dims() != 3) {
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        diag_pseudo_potential_davidson(kp__, ispn, hloc, d_op, q_op);
                    }
                } else {
                    STOP();
                }
            } else if (itso.type_ == "rmm-diis") {
                if (ctx_.num_mag_dims() != 3) {
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        diag_pseudo_potential_rmm_diis(kp__, ispn, hloc, d_op, q_op);
                    }
                } else {
                    STOP();
                }
            } else if (itso.type_ == "chebyshev") {
                P_operator<T> p_op(ctx_, kp__->beta_projectors(), kp__->p_mtrx());
                if (ctx_.num_mag_dims() != 3) {
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        diag_pseudo_potential_chebyshev(kp__, ispn, hloc, d_op, q_op, p_op);

                    }
                } else {
                    STOP();
                }
            } else {
                TERMINATE("unknown iterative solver type");
            }

            ctx_.fft_coarse().dismiss();
        }

        /// Exact (not iterative) diagonalization of the Hamiltonian.
        template <typename T>
        void diag_pseudo_potential_exact(K_point* kp__,
                                         int ispn__,
                                         Hloc_operator& h_op__,
                                         D_operator<T>& d_op__,
                                         Q_operator<T>& q_op__) const;

        /// Iterative Davidson diagonalization.
        template <typename T>
        void diag_pseudo_potential_davidson(K_point* kp__,
                                            int ispn__,
                                            Hloc_operator& h_op__,
                                            D_operator<T>& d_op__,
                                            Q_operator<T>& q_op__) const;

        /// RMM-DIIS diagonalization.
        template <typename T>
        void diag_pseudo_potential_rmm_diis(K_point* kp__,
                                            int ispn__,
                                            Hloc_operator& h_op__,
                                            D_operator<T>& d_op__,
                                            Q_operator<T>& q_op__) const;

        template <typename T>
        void diag_pseudo_potential_chebyshev(K_point* kp__,
                                             int ispn__,
                                             Hloc_operator& h_op__,
                                             D_operator<T>& d_op__,
                                             Q_operator<T>& q_op__,
                                             P_operator<T>& p_op__) const;

        template <typename T>
        inline void diag_h_o(K_point* kp__,
                             int N__,
                             int num_bands__,
                             matrix<T>& hmlt__,
                             matrix<T>& ovlp__,
                             matrix<T>& evec__,
                             dmatrix<T>& hmlt_dist__,
                             dmatrix<T>& ovlp_dist__,
                             dmatrix<T>& evec_dist__,
                             std::vector<double>& eval__) const;

        template <typename T>
        inline void apply_h(K_point* kp__,
                            int ispn__, 
                            int N__,
                            int n__,
                            wave_functions& phi__,
                            wave_functions& hphi__,
                            Hloc_operator &h_op,
                            D_operator<T>& d_op) const;

        template <typename T>
        void apply_h_o(K_point* kp__,
                       int ispn__, 
                       int N__,
                       int n__,
                       wave_functions& phi__,
                       wave_functions& hphi__,
                       wave_functions& ophi__,
                       Hloc_operator &h_op,
                       D_operator<T>& d_op,
                       Q_operator<T>& q_op) const;
        
        template <typename T>
        inline void set_h_o(K_point* kp__,
                            int N__,
                            int n__,
                            wave_functions& phi__,
                            wave_functions& hphi__,
                            wave_functions& ophi__,
                            matrix<T>& h__,
                            matrix<T>& o__,
                            matrix<T>& h_old__,
                            matrix<T>& o_old__) const;
        
        template <typename T>
        inline int residuals(K_point* kp__,
                             int ispn__,
                             int N__,
                             int num_bands__,
                             std::vector<double>& eval__,
                             std::vector<double>& eval_old__,
                             matrix<T>& evec__,
                             wave_functions& hphi__,
                             wave_functions& ophi__,
                             wave_functions& hpsi__,
                             wave_functions& opsi__,
                             wave_functions& res__,
                             std::vector<double>& h_diag__,
                             std::vector<double>& o_diag__) const;

        inline int residuals(K_point* kp__,
                             int N__,
                             int num_bands__,
                             std::vector<double>& eval__,
                             std::vector<double>& eval_old__,
                             matrix<double_complex>& evec__,
                             wave_functions& hphi__,
                             wave_functions& ophi__,
                             wave_functions& hpsi__,
                             wave_functions& opsi__,
                             wave_functions& res__,
                             std::vector<double>& h_diag__,
                             std::vector<double>& o_diag__) const;

        //inline void orthogonalize(K_point* kp__,
        //                          int N__,
        //                          int n__,
        //                          lapwlo_wave_functions& phi__,
        //                          lapwlo_wave_functions& hphi__,
        //                          lapwlo_wave_functions& ophi__,
        //                          matrix<double_complex>& o__) const;

        template <typename T>
        inline void orthogonalize(K_point* kp__,
                                  int N__,
                                  int n__,
                                  wave_functions& phi__,
                                  wave_functions& hphi__,
                                  wave_functions& ophi__,
                                  matrix<T>& o__) const;

        void residuals_aux(K_point* kp__,
                           int num_bands__,
                           std::vector<double>& eval__,
                           wave_functions& hpsi__,
                           wave_functions& opsi__,
                           wave_functions& res__,
                           std::vector<double>& h_diag__,
                           std::vector<double>& o_diag__,
                           std::vector<double>& res_norm__) const;

        void add_nl_h_o_rs(K_point* kp__,
                           int n__,
                           matrix<double_complex>& phi__,
                           matrix<double_complex>& hphi__,
                           matrix<double_complex>& ophi__,
                           mdarray<int, 1>& packed_mtrx_offset__,
                           mdarray<double_complex, 1>& d_mtrx_packed__,
                           mdarray<double_complex, 1>& q_mtrx_packed__,
                           mdarray<double_complex, 1>& kappa__);

    public:
        
        /// Constructor
        Band(Simulation_context& ctx__)
            : ctx_(ctx__),
              unit_cell_(ctx__.unit_cell()),
              blacs_grid_(ctx__.blacs_grid())
        {
            PROFILE();

            gaunt_coefs_ = new Gaunt_coefficients<double_complex>(ctx_.lmax_apw(), 
                                                                  ctx_.lmax_pot(), 
                                                                  ctx_.lmax_apw(),
                                                                  SHT::gaunt_hybrid);

            /* create standard eigen-value solver */
            switch (ctx_.std_evp_solver_type())
            {
                case ev_lapack:
                {
                    std_evp_solver_ = new Eigenproblem_lapack(2 * linalg_base::dlamch('S'));
                    break;
                }
                case ev_scalapack:
                {
                    std_evp_solver_ = new Eigenproblem_scalapack(blacs_grid_, ctx_.cyclic_block_size(), ctx_.cyclic_block_size(), 1e-12);
                    break;
                }
                case ev_plasma:
                {
                    std_evp_solver_ = new Eigenproblem_plasma();
                    break;
                }
                case ev_magma:
                {
                    std_evp_solver_ = new Eigenproblem_magma();
                    break;
                }
                case ev_elpa1:
                {
                    std_evp_solver_ = new Eigenproblem_elpa1(blacs_grid_, ctx_.cyclic_block_size());
                    break;
                }
                case ev_elpa2:
                {
                    std_evp_solver_ = new Eigenproblem_elpa2(blacs_grid_, ctx_.cyclic_block_size());
                    break;
                }
                default:
                {
                    TERMINATE("wrong standard eigen-value solver");
                }
            }
            
            /* create generalized eign-value solver */
            switch (ctx_.gen_evp_solver_type())
            {
                case ev_lapack:
                {
                    gen_evp_solver_ = new Eigenproblem_lapack(2 * linalg_base::dlamch('S'));
                    break;
                }
                case ev_scalapack:
                {
                    gen_evp_solver_ = new Eigenproblem_scalapack(blacs_grid_, ctx_.cyclic_block_size(), ctx_.cyclic_block_size(), 1e-12);
                    break;
                }
                case ev_elpa1:
                {
                    gen_evp_solver_ = new Eigenproblem_elpa1(blacs_grid_, ctx_.cyclic_block_size());
                    break;
                }
                case ev_elpa2:
                {
                    gen_evp_solver_ = new Eigenproblem_elpa2(blacs_grid_, ctx_.cyclic_block_size());
                    break;
                }
                case ev_magma:
                {
                    gen_evp_solver_ = new Eigenproblem_magma();
                    break;
                }
                case ev_rs_gpu:
                {
                    gen_evp_solver_ = new Eigenproblem_RS_GPU(blacs_grid_, ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
                    break;
                }
                case ev_rs_cpu:
                {
                    gen_evp_solver_ = new Eigenproblem_RS_CPU(blacs_grid_, ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
                    break;
                }
                default:
                {
                    TERMINATE("wrong generalized eigen-value solver");
                }
            }

            if (std_evp_solver_->parallel() != gen_evp_solver_->parallel())
                TERMINATE("both eigen-value solvers must be serial or parallel");
        }

        ~Band()
        {
            PROFILE();

            delete gaunt_coefs_;
            delete std_evp_solver_;
            delete gen_evp_solver_;
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
        void apply_hmt_to_apw(Atom const&                 atom__,
                              int                         num_gkvec__,
                              mdarray<double_complex, 2>& alm__,
                              mdarray<double_complex, 2>& halm__) const
        {
            auto& type = atom__.type();

            // TODO: this is k-independent and can in principle be precomputed together with radial integrals if memory is available
            // TODO: check that hmt is indeed Hermitian; compute  upper triangular part and use zhemm
            mdarray<double_complex, 2> hmt(type.mt_aw_basis_size(), type.mt_aw_basis_size());
            /* compute the muffin-tin Hamiltonian */
            for (int j2 = 0; j2 < type.mt_aw_basis_size(); j2++) {
                int lm2 = type.indexb(j2).lm;
                int idxrf2 = type.indexb(j2).idxrf;
                for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
                    int lm1 = type.indexb(j1).lm;
                    int idxrf1 = type.indexb(j1).idxrf;
                    hmt(j1, j2) = atom__.radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));
                }
            }
            linalg<CPU>::gemm(0, 1, num_gkvec__, type.mt_aw_basis_size(), type.mt_aw_basis_size(), alm__, hmt, halm__);
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
        void set_h_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
                          mdarray<double_complex, 2>& h);
        
        /// Set APW-lo and lo-APW blocks of the overlap matrix.
        void set_o_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
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
         *       a_{L\nu}^{\alpha}({\bf G+k}) + \frac{1}{2}{\bf G'} {\bf G} \cdot \Theta({\bf G - G'}) + \tilde V_{eff}({\bf G - G'}) \\
         *          &=& \sum_{\alpha} \sum_{\xi' } a_{\xi'}^{\alpha *}({\bf G'+k}) 
         *              b_{\xi'}^{\alpha}({\bf G+k}) + \frac{1}{2}{\bf G'} {\bf G} \cdot \Theta({\bf G - G'}) + \tilde V_{eff}({\bf G - G'})  
         *  \f}
         *  APW-lo block:
         *  \f[
         *      H_{{\bf G'} j}^{\bf k} = \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'+k}) 
         *      \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}|\hat h^{\alpha_j} |  \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j}  \rangle 
         *  \f]
         *  lo-APW block:
         *  \f[
         *      H_{j' {\bf G}}^{\bf k} = \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}} 
         *          |\hat h^{\alpha_{j'}} | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G+k}) 
         *  \f]
         *  lo-lo block:
         *  \f[
         *      H_{j' j}^{\bf k} = \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}} 
         *          |\hat h^{\alpha_{j}} |  \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j}  \rangle  \delta_{\alpha_j \alpha_{j'}}
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
        inline void set_fv_h_o(K_point* kp,
                               Periodic_function<double>* effective_potential,
                               dmatrix<double_complex>& h,
                               dmatrix<double_complex>& o) const;
        
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
         *          \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}|\hat h^{\alpha_j} | \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j} \rangle = 
         *      \sum_{j} C_{j i} \sum_{\xi'} a_{\xi'}^{\alpha_j *}({\bf G'}) h_{\xi' \xi_j}^{\alpha_j}  
         *  \f]
         *  \f[
         *      o_{{\bf G'} i}(\psi) =  \sum_{j} \langle \phi_{\bf G'} | \phi_{j} \rangle C_{j i} = 
         *      \sum_{j} C_{j i}   \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'}) 
         *          \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}| \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j} \rangle = 
         *      \sum_{j} C_{j i} \sum_{\nu'} a_{\ell_j m_j \nu'}^{\alpha_j *}({\bf G'}) o_{\nu' \zeta_j \ell_j}^{\alpha_j}  
         *  \f]
         *  lo-APW contribution:
         *  \f[
         *     h_{j' i}(\psi) = \sum_{\bf G} \langle \phi_{j'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} = 
         *      \sum_{\bf G} C_{{\bf G} i} \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}} 
         *          |\hat h^{\alpha_{j'}} | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G}) = 
         *      \sum_{\bf G} C_{{\bf G} i} \sum_{\xi} h_{\xi_{j'} \xi}^{\alpha_{j'}} a_{\xi}^{\alpha_{j'}}({\bf G})
         *  \f]
         *  \f[
         *     o_{j' i}(\psi) = \sum_{\bf G} \langle \phi_{j'} |  \phi_{\bf G} \rangle C_{{\bf G} i} = 
         *      \sum_{\bf G} C_{{\bf G} i} \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}} 
         *          | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G}) = 
         *      \sum_{\bf G} C_{{\bf G} i} \sum_{\nu} o_{\zeta_{j'} \nu \ell_{j'}}^{\alpha_{j'}} a_{\ell_{j'} m_{j'} \nu}^{\alpha_{j'}}({\bf G})
         *  \f]
         *  lo-lo contribution:
         *  \f[
         *      h_{j' i}(\psi) = \sum_{j} \langle \phi_{j'} | \hat H | \phi_{j} \rangle C_{j i} = \sum_{j} C_{j i} h_{\xi_{j'} \xi_j}^{\alpha_j}
         *          \delta_{\alpha_j \alpha_{j'}}
         *  \f]
         *  \f[
         *      o_{j' i}(\psi) = \sum_{j} \langle \phi_{j'} | \phi_{j} \rangle C_{j i} = \sum_{j} C_{j i} 
         *          o_{\zeta_{j'} \zeta_{j} \ell_j}^{\alpha_j}
         *            \delta_{\alpha_j \alpha_{j'}} \delta_{\ell_j \ell_{j'}} \delta_{m_j m_{j'}}
         *  \f]
         */
        inline void apply_fv_h_o(K_point* kp__,
                                 Periodic_function<double>* effective_potential__,
                                 int N,
                                 int n,
                                 wave_functions& phi__,
                                 wave_functions& hphi__,
                                 wave_functions& ophi__) const;

        /// Solve first-variational (non-magnetic) problem
        inline void solve_fv(K_point* kp__, Periodic_function<double>* effective_potential__) const
        {

            if (kp__->gklo_basis_size() < ctx_.num_fv_states()) {
                TERMINATE("basis size is too small");
            }

            switch (ctx_.esm_type()) {
                case electronic_structure_method_t::full_potential_pwlo:
                case electronic_structure_method_t::full_potential_lapwlo: {
                    diag_fv_full_potential(kp__, effective_potential__);
                    break;
                }
                default: {
                    TERMINATE_NOT_IMPLEMENTED
                }
            }
        }

        /// Solve second-variational problem
        void solve_sv(K_point* kp, Periodic_function<double>* effective_magnetic_field[3]) const;

        void solve_fd(K_point* kp,
                      Periodic_function<double>* effective_potential, 
                      Periodic_function<double>* effective_magnetic_field[3]) const;

        void solve_for_kset(K_set& kset, Potential& potential, bool precompute) const;

        inline Eigenproblem* std_evp_solver() const
        {
            return std_evp_solver_;
        }

        inline Eigenproblem const* gen_evp_solver() const
        {
            return gen_evp_solver_;
        }

        /// Get diagonal elements of LAPW Hamiltonian.
        inline std::vector<double> get_h_diag(K_point* kp__,
                                              double v0__,
                                              double theta0__) const;

        /// Get diagonal elements of LAPW overlap.
        inline std::vector<double> get_o_diag(K_point* kp__,
                                              double theta0__) const;

        template <typename T>
        inline std::vector<double> get_h_diag(K_point* kp__,
                                              int ispn__,
                                              double v0__,
                                              D_operator<T>& d_op__) const;

        /// Get diagonal elements of overlap matrix.
        template <typename T>
        inline std::vector<double> get_o_diag(K_point* kp__,
                                              Q_operator<T>& q_op__) const;

        template <typename T>
        void initialize_subspace(K_point* kp__,
                                 Periodic_function<double>* effective_potential__,
                                 Periodic_function<double>* effective_magnetic_field[3],
                                 int num_ao__,
                                 int lmax__,
                                 std::vector< std::vector< Spline<double> > >& rad_int__) const;
};

#include "Band/get_h_o_diag.hpp"
#include "Band/apply_h.hpp"
#include "Band/set_fv_h_o.hpp"
#include "Band/residuals.hpp"

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [in] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [in] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
template <>
inline void Band::set_h_o<double_complex>(K_point* kp__,
                                          int N__,
                                          int n__,
                                          wave_functions& phi__,
                                          wave_functions& hphi__,
                                          wave_functions& ophi__,
                                          matrix<double_complex>& h__,
                                          matrix<double_complex>& o__,
                                          matrix<double_complex>& h_old__,
                                          matrix<double_complex>& o_old__) const
{
    PROFILE_WITH_TIMER("sirius::Band::set_h_o");
    
    assert(n__ != 0);

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < N__; i++) {
        std::memcpy(&h__(0, i), &h_old__(0, i), N__ * sizeof(double_complex));
        std::memcpy(&o__(0, i), &o_old__(0, i), N__ * sizeof(double_complex));
    }

    /* <{phi,res}|H|res> */
    //phi__.inner<double_complex>(0, N__ + n__, hphi__, N__, n__, h__, 0, N__, kp__->comm());
    inner(phi__, 0, N__ + n__, hphi__, N__, n__, h__, 0, N__, kp__->comm(), ctx_.processing_unit());
    /* <{phi,res}|O|res> */
    //phi__.inner<double_complex>(0, N__ + n__, ophi__, N__, n__, o__, 0, N__, kp__->comm());
    inner(phi__, 0, N__ + n__, ophi__, N__, n__, o__, 0, N__, kp__->comm(), ctx_.processing_unit());

    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex cs1(0, 0);
    double_complex cs2(0, 0);
    for (int i = 0; i < N__ + n__; i++)
    {
        for (int j = 0; j <= i; j++) 
        {
            cs1 += h__(j, i);
            cs2 += o__(j, i);
        }
    }
    DUMP("checksum(h): %18.10f %18.10f", cs1.real(), cs1.imag());
    DUMP("checksum(o): %18.10f %18.10f", cs2.real(), cs2.imag());
    #endif

    for (int i = 0; i < N__ + n__; i++)
    {
        //= if (h__(i, i).imag() > 1e-12)
        //= {
        //=     std::stringstream s;
        //=     s << "wrong diagonal of H: " << h__(i, i);
        //=     TERMINATE(s);
        //= }
        //= if (o__(i, i).imag() > 1e-12)
        //= {
        //=     std::stringstream s;
        //=     s << "wrong diagonal of O: " << o__(i, i);
        //=     TERMINATE(s);
        //= }
        h__(i, i) = h__(i, i).real();
        o__(i, i) = o__(i, i).real();
    }

    #if (__VERIFICATION > 0)
    /* check n__ * n__ block */
    for (int i = N__; i < N__ + n__; i++) {
        for (int j = N__; j < N__ + n__; j++) {
            if (std::abs(h__(i, j) - std::conj(h__(j, i))) > 1e-10 ||
                std::abs(o__(i, j) - std::conj(o__(j, i))) > 1e-10) {
                double_complex z1, z2;
                z1 = h__(i, j);
                z2 = h__(j, i);

                std::cout << "h(" << i << "," << j << ")=" << z1 << " "
                          << "h(" << j << "," << i << ")=" << z2 << ", diff=" << std::abs(z1 - std::conj(z2)) << std::endl;
                
                z1 = o__(i, j);
                z2 = o__(j, i);

                std::cout << "o(" << i << "," << j << ")=" << z1 << " "
                          << "o(" << j << "," << i << ")=" << z2 << ", diff=" << std::abs(z1 - std::conj(z2)) << std::endl;
                
            }
        }
    }
    #endif
    
    int i0 = N__;
    if (gen_evp_solver_->type() == ev_magma || gen_evp_solver_->type() == ev_elpa1 || gen_evp_solver_->type() == ev_elpa2) {
        /* restore the lower part */
        #pragma omp parallel for
        for (int i = 0; i < N__; i++) {
            for (int j = N__; j < N__ + n__; j++) {
                h__(j, i) = std::conj(h__(i, j));
                o__(j, i) = std::conj(o__(i, j));
            }
        }
        i0 = 0;
    }

    /* save Hamiltonian and overlap */
    #pragma omp parallel for
    for (int i = i0; i < N__ + n__; i++) {
        if (h_old__.size()) {
            std::memcpy(&h_old__(0, i), &h__(0, i), (N__ + n__) * sizeof(double_complex));
        }
        if (o_old__.size()) {
            std::memcpy(&o_old__(0, i), &o__(0, i), (N__ + n__) * sizeof(double_complex));
        }
    }
}

template <>
inline void Band::set_h_o<double>(K_point* kp__,
                                  int N__,
                                  int n__,
                                  wave_functions& phi__,
                                  wave_functions& hphi__,
                                  wave_functions& ophi__,
                                  matrix<double>& h__,
                                  matrix<double>& o__,
                                  matrix<double>& h_old__,
                                  matrix<double>& o_old__) const
{
    PROFILE_WITH_TIMER("sirius::Band::set_h_o");
    
    assert(n__ != 0);

    /* copy old Hamiltonian */
    for (int i = 0; i < N__; i++) {
        std::memcpy(&h__(0, i), &h_old__(0, i), N__ * sizeof(double));
    }

    /* <{phi,res}|H|res> */
    //phi__.inner<double>(0, N__ + n__, hphi__, N__, n__, h__, 0, N__, kp__->comm());
    inner(phi__, 0, N__ + n__, hphi__, N__, n__, h__, 0, N__, kp__->comm(), ctx_.processing_unit());

    int i0 = N__;
    if (gen_evp_solver_->type() == ev_magma || gen_evp_solver_->type() == ev_elpa1 || gen_evp_solver_->type() == ev_elpa2) {
        /* restore the lower part */
        #pragma omp parallel for
        for (int i = 0; i < N__; i++) {
            for (int j = N__; j < N__ + n__; j++) {
                h__(j, i) = h__(i, j);
            }
        }
        i0 = 0;
    }

    /* save Hamiltonian */
    if (h_old__.size()) {
        #pragma omp parallel for
        for (int i = i0; i < N__ + n__; i++) {
            std::memcpy(&h_old__(0, i), &h__(0, i), (N__ + n__) * sizeof(double));
        }
    }
}

template <>
inline void Band::diag_h_o<double_complex>(K_point* kp__,
                                           int N__,
                                           int num_bands__,
                                           matrix<double_complex>& hmlt__,
                                           matrix<double_complex>& ovlp__,
                                           matrix<double_complex>& evec__,
                                           dmatrix<double_complex>& hmlt_dist__,
                                           dmatrix<double_complex>& ovlp_dist__,
                                           dmatrix<double_complex>& evec_dist__,
                                           std::vector<double>& eval__) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_h_o");

    runtime::Timer t1("sirius::Band::diag_h_o|load");
    if (kp__->comm().size() > 1 && gen_evp_solver()->parallel())
    {
        for (int jloc = 0; jloc < hmlt_dist__.num_cols_local(); jloc++)
        {
            int j = hmlt_dist__.icol(jloc);
            for (int iloc = 0; iloc < hmlt_dist__.num_rows_local(); iloc++)
            {
                int i = hmlt_dist__.irow(iloc);
                hmlt_dist__(iloc, jloc) = (i > j) ? std::conj(hmlt__(j, i)) : hmlt__(i, j);
                ovlp_dist__(iloc, jloc) = (i > j) ? std::conj(ovlp__(j, i)) : ovlp__(i, j);
            }
        }
    }
    t1.stop();

    runtime::Timer t2("sirius::Band::diag_h_o|diag");
    int result;
    if (gen_evp_solver()->parallel())
    {
        result = gen_evp_solver()->solve(N__,  num_bands__, hmlt_dist__.at<CPU>(), hmlt_dist__.ld(),
                                         ovlp_dist__.at<CPU>(), ovlp_dist__.ld(), &eval__[0], evec_dist__.at<CPU>(),
                                         evec_dist__.ld(), hmlt_dist__.num_rows_local(), hmlt_dist__.num_cols_local());
    }
    else
    {
        result = gen_evp_solver()->solve(N__, num_bands__, hmlt__.at<CPU>(), hmlt__.ld(), ovlp__.at<CPU>(), ovlp__.ld(),
                                         &eval__[0], evec__.at<CPU>(), evec__.ld());
    }
    if (result)
    {
        std::stringstream s;
        s << "error in diagonalziation for k-point (" << kp__->vk()[0] << " " << kp__->vk()[1] << " " << kp__->vk()[2] << ")";
        TERMINATE(s);
    }
    t2.stop();

    runtime::Timer t3("sirius::Band::diag_h_o|gather");
    if (kp__->comm().size() > 1 && gen_evp_solver()->parallel())
    {
        evec__.zero();
        for (int i = 0; i < evec_dist__.num_cols_local(); i++)
        {
            for (int j = 0; j < evec_dist__.num_rows_local(); j++)
            {
                evec__(evec_dist__.irow(j), evec_dist__.icol(i)) = evec_dist__(j, i);
            }
        }
        kp__->comm().allreduce(evec__.at<CPU>(), evec__.ld() * num_bands__);
    }
    t3.stop();
    
    //// --== DEBUG ==--
    //printf("checking evec\n");
    //for (int i = 0; i < num_bands__; i++)
    //{
    //    for (int j = 0; j < N__; j++)
    //    {
    //        if (std::abs(evec__(j, i).imag()) > 1e-12)
    //        {
    //            printf("evec(%i, %i) = %20.16f %20.16f\n", i, j, evec__(j, i).real(), evec__(j, i).real());
    //        }
    //    }
    //}
    //printf("done.\n");

    /* copy eigen-vectors to GPU */
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
        acc::copyin(evec__.at<GPU>(), evec__.ld(), evec__.at<CPU>(), evec__.ld(), N__, num_bands__);
    #endif
}

template <>
inline void Band::diag_h_o<double>(K_point* kp__,
                                   int N__,
                                   int num_bands__,
                                   matrix<double>& hmlt__,
                                   matrix<double>& ovlp__,
                                   matrix<double>& evec__,
                                   dmatrix<double>& hmlt_dist__,
                                   dmatrix<double>& ovlp_dist__,
                                   dmatrix<double>& evec_dist__,
                                   std::vector<double>& eval__) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_h_o");

    runtime::Timer t1("sirius::Band::diag_h_o|load");
    if (kp__->comm().size() > 1 && std_evp_solver()->parallel())
    {
        for (int jloc = 0; jloc < hmlt_dist__.num_cols_local(); jloc++)
        {
            int j = hmlt_dist__.icol(jloc);
            for (int iloc = 0; iloc < hmlt_dist__.num_rows_local(); iloc++)
            {
                int i = hmlt_dist__.irow(iloc);
                hmlt_dist__(iloc, jloc) = (i > j) ? hmlt__(j, i) : hmlt__(i, j);
            }
        }
    }
    t1.stop();

    runtime::Timer t2("sirius::Band::diag_h_o|diag");
    int result;
    if (std_evp_solver()->parallel())
    {
        result = std_evp_solver()->solve(N__,  num_bands__, hmlt_dist__.at<CPU>(), hmlt_dist__.ld(),
                                         &eval__[0], evec_dist__.at<CPU>(), evec_dist__.ld(),
                                         hmlt_dist__.num_rows_local(), hmlt_dist__.num_cols_local());
    }
    else
    {
        result = std_evp_solver()->solve(N__, num_bands__, hmlt__.at<CPU>(), hmlt__.ld(), &eval__[0], evec__.at<CPU>(), evec__.ld());
    }
    if (result) TERMINATE("error in diagonalziation");
    t2.stop();

    runtime::Timer t3("sirius::Band::diag_h_o|gather");
    if (kp__->comm().size() > 1 && std_evp_solver()->parallel())
    {
        evec__.zero();
        for (int i = 0; i < evec_dist__.num_cols_local(); i++)
        {
            for (int j = 0; j < evec_dist__.num_rows_local(); j++)
            {
                evec__(evec_dist__.irow(j), evec_dist__.icol(i)) = evec_dist__(j, i);
            }
        }
        kp__->comm().allreduce(evec__.at<CPU>(), evec__.ld() * num_bands__);
    }
    t3.stop();

    /* copy eigen-vectors to GPU */
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
        acc::copyin(evec__.at<GPU>(), evec__.ld(), evec__.at<CPU>(), evec__.ld(), N__, num_bands__);
    #endif
}

template <>
inline void Band::orthogonalize<double_complex>(K_point* kp__,
                                                int N__,
                                                int n__,
                                                wave_functions& phi__,
                                                wave_functions& hphi__,
                                                wave_functions& ophi__,
                                                matrix<double_complex>& o__) const
{
    auto wfs = {&phi__, &hphi__, &ophi__};

    if (phi__.params().full_potential()) STOP();

    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new> */
    if (N__ > 0) {
        inner(phi__, 0, N__, ophi__, N__, n__, o__, 0, 0, kp__->comm(), ctx_.processing_unit());

        if (ctx_.processing_unit() == CPU) {
            for (auto& e: wfs) {
                linalg<CPU>::gemm(0, 0, e->pw_coeffs().num_rows_loc(), n__, N__,
                                  double_complex(-1, 0), 
                                  e->pw_coeffs().prime().at<CPU>(0, 0), e->pw_coeffs().prime().ld(),
                                  o__.at<CPU>(0, 0), o__.ld(),
                                  double_complex(1, 0),
                                  e->pw_coeffs().prime().at<CPU>(0, N__), e->pw_coeffs().prime().ld());
            }
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            STOP();
            //==/* copy overlap matrix <phi_old|phi_new> to GPU */
            //==acc::copyin(o__.at<GPU>(0, 0), o__.ld(), o__.at<CPU>(0, 0), o__.ld(), N__, n__);

            //==double alpha = 1;
            //==double m_alpha = -1;

            //==for (int i = 0; i < 3; i++)
            //=={
            //==    linalg<GPU>::gemm(0, 0, 2 * kp__->num_gkvec_loc(), n__, N__,
            //==                      &m_alpha, 
            //==                      (double*)wfs[i]->coeffs().at<GPU>(0, 0), 2 * kp__->num_gkvec_loc(),
            //==                      o__.at<GPU>(0, 0), o__.ld(),
            //==                      &alpha,
            //==                      (double*)wfs[i]->coeffs().at<GPU>(0, N__), 2 * kp__->num_gkvec_loc());
            //==}

            //==acc::sync_stream(-1);
        }
        #endif
    }

    /* orthogonalize new n__ x n__ block */
    inner(phi__, N__, n__, ophi__, N__, n__, o__, 0, 0, kp__->comm(), ctx_.processing_unit());
    
    if (ctx_.processing_unit() == CPU) {
        int info;
        if ((info = linalg<CPU>::potrf(n__, &o__(0, 0), o__.ld()))) {
            std::stringstream s;
            s << "error in factorization, info = " << info;
            TERMINATE(s);
        }

        if (linalg<CPU>::trtri(n__, &o__(0, 0), o__.ld())) {
            TERMINATE("error in inversion");
        }

        for (auto& e: wfs) {
            linalg<CPU>::trmm('R', 'U', 'N', e->pw_coeffs().num_rows_loc(), n__, double_complex(1, 0),
                              o__.at<CPU>(0, 0), o__.ld(),
                              e->pw_coeffs().prime().at<CPU>(0, N__), e->pw_coeffs().prime().ld());
        }
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
    {
        STOP();
        //== acc::copyin(o__.at<GPU>(0, 0), o__.ld(), o__.at<CPU>(0, 0), o__.ld(), n__, n__);

        //== int info;
        //== if ((info = linalg<GPU>::potrf(n__, o__.at<GPU>(0, 0), o__.ld())))
        //== {
        //==     std::stringstream s;
        //==     s << "error in factorization, info = " << info;
        //==     TERMINATE(s);
        //== }

        //== if (linalg<GPU>::trtri(n__, o__.at<GPU>(0, 0), o__.ld()))
        //==     TERMINATE("error in inversion");

        //== double alpha = 1;

        //== for (int i = 0; i < 3; i++)
        //== {
        //==     linalg<GPU>::trmm('R', 'U', 'N', 2 * kp__->num_gkvec_loc(), n__, &alpha,
        //==                       o__.at<GPU>(0, 0), o__.ld(),
        //==                       (double*)wfs[i]->coeffs().at<GPU>(0, N__), 2 * kp__->num_gkvec_loc());
        //== }
        //== acc::sync_stream(-1);
    }
    #endif
}

template <>
inline void Band::orthogonalize<double>(K_point* kp__,
                                        int N__,
                                        int n__,
                                        wave_functions& phi__,
                                        wave_functions& hphi__,
                                        wave_functions& ophi__,
                                        matrix<double>& o__) const
{
    PROFILE_WITH_TIMER("sirius::Band::orthogonalize");

    auto wfs = {&phi__, &hphi__, &ophi__};

    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new> */
    if (N__ > 0) {
        inner(phi__, 0, N__, ophi__, N__, n__, o__, 0, 0, kp__->comm(), ctx_.processing_unit());

        if (ctx_.processing_unit() == CPU) {
            for (auto& e: wfs) {
                linalg<CPU>::gemm(0, 0, 2 * e->pw_coeffs().num_rows_loc(), n__, N__,
                                  -1.0, 
                                  (double*)e->pw_coeffs().prime().at<CPU>(0, 0), 2 * e->pw_coeffs().prime().ld(),
                                  o__.at<CPU>(0, 0), o__.ld(),
                                  1.0,
                                  (double*)e->pw_coeffs().prime().at<CPU>(0, N__), 2 * e->pw_coeffs().prime().ld());
            }
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            /* copy overlap matrix <phi_old|phi_new> to GPU */
            acc::copyin(o__.at<GPU>(0, 0), o__.ld(), o__.at<CPU>(0, 0), o__.ld(), N__, n__);

            double alpha = 1;
            double m_alpha = -1;

            for (int i = 0; i < 3; i++) {
                linalg<GPU>::gemm(0, 0, 2 * kp__->num_gkvec_loc(), n__, N__,
                                  &m_alpha, 
                                  (double*)wfs[i]->coeffs().at<GPU>(0, 0), 2 * kp__->num_gkvec_loc(),
                                  o__.at<GPU>(0, 0), o__.ld(),
                                  &alpha,
                                  (double*)wfs[i]->coeffs().at<GPU>(0, N__), 2 * kp__->num_gkvec_loc());
            }

            acc::sync_stream(-1);
        }
        #endif
    }

    /* orthogonalize new n__ x n__ block */
    inner(phi__, N__, n__, ophi__, N__, n__, o__, 0, 0, kp__->comm(), ctx_.processing_unit());

    if (ctx_.processing_unit() == CPU) {
        int info;
        if ((info = linalg<CPU>::potrf(n__, &o__(0, 0), o__.ld()))) {
            std::stringstream s;
            s << "error in factorization, info = " << info;
            TERMINATE(s);
        }

        if (linalg<CPU>::trtri(n__, &o__(0, 0), o__.ld())) {
            TERMINATE("error in inversion");
        }

        for (auto& e: wfs) {
            linalg<CPU>::trmm('R', 'U', 'N', 2 * e->pw_coeffs().num_rows_loc(), n__, 1.0,
                              o__.at<CPU>(0, 0), o__.ld(),
                              (double*)e->pw_coeffs().prime().at<CPU>(0, N__), 2 * e->pw_coeffs().prime().ld());
        }
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        acc::copyin(o__.at<GPU>(0, 0), o__.ld(), o__.at<CPU>(0, 0), o__.ld(), n__, n__);

        int info;
        if ((info = linalg<GPU>::potrf(n__, o__.at<GPU>(0, 0), o__.ld()))) {
            std::stringstream s;
            s << "error in factorization, info = " << info;
            TERMINATE(s);
        }

        if (linalg<GPU>::trtri(n__, o__.at<GPU>(0, 0), o__.ld())) {
            TERMINATE("error in inversion");
        }

        double alpha = 1;

        for (int i = 0; i < 3; i++) {
            linalg<GPU>::trmm('R', 'U', 'N', 2 * kp__->num_gkvec_loc(), n__, &alpha,
                              o__.at<GPU>(0, 0), o__.ld(),
                              (double*)wfs[i]->coeffs().at<GPU>(0, N__), 2 * kp__->num_gkvec_loc());
        }
        acc::sync_stream(-1);
    }
    #endif

    //// --== DEBUG ==--
    //phi__.inner<double>(0, N__ + n__, ophi__, 0, N__ + n__, o__, 0, 0);
    //for (int i = 0; i < N__ + n__; i++)
    //{
    //    for (int j = 0; j < N__ + n__; j++)
    //    {
    //        double a = o__(j, i);
    //        if (i == j) a -= 1;

    //        if (std::abs(a) > 1e-10)
    //        {
    //            printf("wrong overlap");
    //            std::stringstream s;
    //            s << "wrong overlap, diff=" << a;
    //            TERMINATE(s);
    //        }
    //    }
    //}
}

inline void Band::apply_fv_h_o(K_point* kp__,
                               Periodic_function<double>* effective_potential__,
                               int N__,
                               int n__,
                               wave_functions& phi__,
                               wave_functions& hphi__,
                               wave_functions& ophi__) const
{
    PROFILE_WITH_TIMER("sirius::Band::apply_fv_h_o");

    ctx_.fft().prepare(kp__->gkvec().partition());

    STOP();

//==     mdarray<double_complex, 1> buf_rg(ctx_.fft().local_size());
//==     mdarray<double_complex, 1> buf_pw(kp__->gkvec().partition().gvec_count_fft());
//== 
//==     hphi__.copy_from(phi__, N__, n__);
//==     ophi__.copy_from(phi__, N__, n__);
//== 
//==      phi__.apw_coeffs().remap_forward(N__, n__, kp__->gkvec().partition(), ctx_.mpi_grid_fft().communicator(1 << 1));
//==     hphi__.apw_coeffs().remap_forward(N__, n__, kp__->gkvec().partition(), ctx_.mpi_grid_fft().communicator(1 << 1));
//==     ophi__.apw_coeffs().remap_forward(N__, n__, kp__->gkvec().partition(), ctx_.mpi_grid_fft().communicator(1 << 1));
//==     
//==     STOP();
//==     for (int j = N__; j < N__ + n__; j++) {
//==         /* phi(G) -> phi(r) */
//==         ctx_.fft().transform<1>(kp__->gkvec().partition(), phi__.apw_coeffs().spare().at<CPU>(0, j));
//==         #pragma omp parallel for
//==         for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
//==             ctx_.fft().buffer(ir) *= ctx_.step_function().theta_r(ir);
//==             /* save phi(r) * Theta(r) */
//==             buf_rg[ir] = ctx_.fft().buffer(ir);
//==         }
//==         /* phi(r) * Theta(r) -> ophi(G) */
//==         ctx_.fft().transform<-1>(kp__->gkvec().partition(), ophi__.apw_coeffs().spare().at<CPU>(0, j));
//==         #pragma omp parallel for
//==         for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
//==             ctx_.fft().buffer(ir) = buf_rg[ir] * effective_potential__->f_rg(ir);
//==         }
//==         /* phi(r) * Theta(r) * V(r) -> ophi(G) */
//==         ctx_.fft().transform<-1>(kp__->gkvec().partition(), hphi__.apw_coeffs().spare().at<CPU>(0, j));
//==         
//==         for (int x: {0, 1, 2}) {
//==             for (int igloc = 0; igloc < kp__->gkvec().partition().gvec_count_fft(); igloc++) {
//==                 int ig = kp__->gkvec().partition().gvec_offset_fft() + igloc;
//==                 buf_pw[igloc] = phi__.apw_coeffs().spare()(igloc, j) * kp__->gkvec().gkvec_cart(ig)[x];
//==             }
//==             ctx_.fft().transform<1>(kp__->gkvec().partition(), &buf_pw[0]);
//==             #pragma omp parallel for
//==             for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
//==                 ctx_.fft().buffer(ir) *= ctx_.step_function().theta_r(ir);
//==             }
//==             ctx_.fft().transform<-1>(kp__->gkvec().partition(), &buf_pw[0]);
//==             for (int igloc = 0; igloc < kp__->gkvec().partition().gvec_count_fft(); igloc++) {
//==                 int ig = kp__->gkvec().partition().gvec_offset_fft() + igloc;
//==                 hphi__.apw_coeffs().spare()(igloc, j) += 0.5 * buf_pw[igloc] * kp__->gkvec().gkvec_cart(ig)[x];
//==             }
//==         }
//==     }
//== 
//==     hphi__.apw_coeffs().remap_backward(N__, n__, kp__->gkvec().partition(), ctx_.mpi_grid_fft().communicator(1 << 1));
//==     ophi__.apw_coeffs().remap_backward(N__, n__, kp__->gkvec().partition(), ctx_.mpi_grid_fft().communicator(1 << 1));
//== 
//==     ctx_.fft().dismiss();
//== 
//==     matrix<double_complex> alm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
//==     matrix<double_complex> halm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
//==     matrix<double_complex> tmp(unit_cell_.max_mt_aw_basis_size(), n__);
//==     matrix<double_complex> htmp(unit_cell_.max_mt_aw_basis_size(), n__);
//== 
//==     double_complex zone(1, 0);
//== 
//==     //mdarray<double_complex, 2> ophi_lo(ophi__.num_lo_loc(), n__);
//==     //mdarray<double_complex, 2> hphi_lo(hphi__.num_lo_loc(), n__);
//== 
//==     //ophi_lo.zero();
//==     //hphi_lo.zero();
//==     
//==     assert(ophi__.num_lo_loc() == hphi__.num_lo_loc());
//==     for (int ibnd = N__; ibnd < N__ + n__; ibnd++) {
//==         for (int ilo = 0; ilo < hphi__.num_lo_loc(); ilo++) {
//==             ophi__.lo_coeffs()(ilo, ibnd) = 0;
//==             hphi__.lo_coeffs()(ilo, ibnd) = 0;
//==         }
//==     }
//==     
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
//==         auto& atom = unit_cell_.atom(ia);
//==         auto& type = atom.type(); 
//==         int nmt = atom.mt_aw_basis_size();
//==         kp__->alm_coeffs_loc().generate(ia, alm);
//==         apply_hmt_to_apw<spin_block_t::nm>(atom, kp__->num_gkvec(), alm, halm);
//==         
//==         /* tmp(lm, i) = A(G, lm)^{T} * C(G, i) */
//==         linalg<CPU>::gemm(1, 0, nmt, n__, kp__->num_gkvec_loc(),
//==                           alm.at<CPU>(), alm.ld(),
//==                           phi__.apw_coeffs().prime().at<CPU>(0, N__), phi__.apw_coeffs().prime().ld(),
//==                           tmp.at<CPU>(), tmp.ld());
//==         kp__->comm().allreduce(tmp.at<CPU>(), static_cast<int>(tmp.size()));
//== 
//==         /* htmp(lm, i) = H_{mt}A(G, lm)^{T} * C(G, i) */
//==         linalg<CPU>::gemm(1, 0, nmt, n__, kp__->num_gkvec_loc(),
//==                           halm.at<CPU>(), halm.ld(),
//==                           phi__.apw_coeffs().prime().at<CPU>(0, N__), phi__.apw_coeffs().prime().ld(),
//==                           htmp.at<CPU>(), htmp.ld());
//==         kp__->comm().allreduce(htmp.at<CPU>(), static_cast<int>(tmp.size()));
//== 
//==         for (int xi = 0; xi < nmt; xi++) {
//==             for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
//==                 alm(ig, xi) = std::conj(alm(ig, xi));
//==             }
//==         }
//==         /* APW-APW contribution to overlap */
//==         linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nmt, zone,
//==                           alm.at<CPU>(), alm.ld(),
//==                           tmp.at<CPU>(), tmp.ld(),
//==                           zone,
//==                           ophi__.apw_coeffs().prime().at<CPU>(0, N__),
//==                           ophi__.apw_coeffs().prime().ld());
//==         /* APW-APW contribution to Hamiltonian */
//==         linalg<CPU>::gemm(0, 0, kp__->num_gkvec_loc(), n__, nmt, zone,
//==                           alm.at<CPU>(), alm.ld(),
//==                           htmp.at<CPU>(), htmp.ld(),
//==                           zone,
//==                           hphi__.apw_coeffs().prime().at<CPU>(0, N__),
//==                           hphi__.apw_coeffs().prime().ld());
//==         
//==         int offs = type.mt_aw_basis_size();
//==         
//==         /* local orbital coefficients of atom ia for all states */
//==         matrix<double_complex> phi_lo_ia(atom.mt_lo_basis_size(), n__);
//==         auto ia_location = phi__.spl_num_atoms().location(ia);
//==         if (ia_location.second == kp__->comm().rank()) {
//==             for (int i = 0; i < n__; i++) {
//==                 std::memcpy(&phi_lo_ia(0, i),
//==                             phi__.lo_coeffs().at<CPU>(phi__.offset_lo(ia_location.first), N__ + i),
//==                             atom.mt_lo_basis_size() * sizeof(double_complex));
//==             }
//==         }
//==         kp__->comm().bcast(phi_lo_ia.at<CPU>(), static_cast<int>(phi_lo_ia.size()), ia_location.second);
//== 
//==         /* sum over local obritals (this are the APW-lo and lo-lo contributions) */
//==         for (int ibnd = N__; ibnd < N__ + n__; ibnd++) {
//==             for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
//==                 //int j = ilo + kp__->num_gkvec() + atom.offset_lo();
//==                 int xi_lo = offs + ilo;
//==                 /* local orbital indices */
//==                 int l_lo     = type.indexb(xi_lo).l;
//==                 int lm_lo    = type.indexb(xi_lo).lm;
//==                 int order_lo = type.indexb(xi_lo).order;
//==                 int idxrf_lo = type.indexb(xi_lo).idxrf;
//== 
//==                 for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
//==                     /* APW-lo contribution */
//==                     auto z = phi_lo_ia(ilo, ibnd) * atom.symmetry_class().o_radial_integral(l_lo, order_aw, order_lo);
//==                     for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
//==                         ophi__.apw_coeffs()(ig, ibnd) += alm(ig, type.indexb_by_lm_order(lm_lo, order_aw)) * z;
//==                     }
//==                 }
//== 
//==                 for (int xi = 0; xi < nmt; xi++) {
//==                     int lm_aw    = type.indexb(xi).lm;
//==                     int idxrf_aw = type.indexb(xi).idxrf;
//==                     auto& gc = gaunt_coefs_->gaunt_vector(lm_aw, lm_lo);
//== 
//==                     auto z = phi_lo_ia(ilo, ibnd) * atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_aw, idxrf_lo, gc);
//==                     /* APW-lo contribution */
//==                     for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
//==                         hphi__.apw_coeffs()(ig, ibnd) += alm(ig, xi) * z; 
//==                     }
//==                 }
//==                 
//==                 if (ia_location.second == kp__->comm().rank()) {
//==                     /* lo-lo contribution */
//==                     for (int jlo = 0; jlo < type.mt_lo_basis_size(); jlo++) {
//==                         //int j1     = jlo + kp__->num_gkvec() + atom.offset_lo();
//==                         int lm1    = type.indexb(offs + jlo).lm;
//==                         int order1 = type.indexb(offs + jlo).order;
//==                         int idxrf1 = type.indexb(offs + jlo).idxrf;
//==                         auto& gc = gaunt_coefs_->gaunt_vector(lm_lo, lm1);
//==                         if (lm_lo == lm1) {
//==                             //ophi__(j, ibnd) += phi__(j1, ibnd) * atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1);
//==                             ophi__.lo_coeffs()(ophi__.offset_lo(ia_location.first) + ilo, ibnd) += phi_lo_ia(jlo, ibnd) *
//==                                 atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1);
//==                         }
//==                         //hphi__(j, ibnd) +=  phi__(j1, ibnd) * atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf1, gc);
//==                         hphi__.lo_coeffs()(hphi__.offset_lo(ia_location.first) + ilo, ibnd) += phi_lo_ia(jlo, ibnd) *
//==                             atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf1, gc);
//==                     }
//== 
//==                     for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
//==                         /* lo-APW contribution */
//==                         ophi__.lo_coeffs()(ophi__.offset_lo(ia_location.first) + ilo, ibnd) += atom.symmetry_class().o_radial_integral(l_lo, order_lo, order_aw) *
//==                                                                                                tmp(type.indexb_by_lm_order(lm_lo, order_aw), ibnd - N__);
//==                     }
//== 
//==                     double_complex z(0, 0);
//==                     for (int xi = 0; xi < nmt; xi++) {
//==                         int lm_aw    = type.indexb(xi).lm;
//==                         int idxrf_aw = type.indexb(xi).idxrf;
//==                         auto& gc = gaunt_coefs_->gaunt_vector(lm_lo, lm_aw);
//==                         z += atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf_aw, gc) * tmp(xi, ibnd - N__);
//==                     }
//==                     /* lo-APW contribution */
//==                     hphi__.lo_coeffs()(hphi__.offset_lo(ia_location.first) + ilo, ibnd) += z;
//==                 }
//==             }
//==         }
//==         
//==         //== /* lo-APW contribution */
//==         //== for (int ibnd = N__; ibnd < N__ + n__; ibnd++) {
//==         //==     for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
//==         //==         int j = ilo + kp__->num_gkvec() + atom.offset_lo();
//==         //==         int xi_lo = offs + ilo;
//==         //==         /* local orbital indices */
//==         //==         int l_lo     = type.indexb(xi_lo).l;
//==         //==         int lm_lo    = type.indexb(xi_lo).lm;
//==         //==         int order_lo = type.indexb(xi_lo).order;
//==         //==         int idxrf_lo = type.indexb(xi_lo).idxrf;
//== 
//==         //==         for (int order_aw = 0; order_aw < (int)type.aw_descriptor(l_lo).size(); order_aw++) {
//==         //==             /* lo-APW contribution */
//==         //==             ophi__(j, ibnd) += atom.symmetry_class().o_radial_integral(l_lo, order_lo, order_aw) *
//==         //==                                tmp(type.indexb_by_lm_order(lm_lo, order_aw), ibnd - N__);
//==         //==             ///* APW-lo contribution */
//==         //==             //auto z = phi__(j, ibnd) * atom.symmetry_class().o_radial_integral(l_lo, order_aw, order_lo);
//==         //==             //for (int ig = 0; ig < kp__->num_gkvec(); ig++) {
//==         //==             //    ophi__(ig, ibnd) += alm(ig, type.indexb_by_lm_order(lm_lo, order_aw)) * z;
//==         //==             //}
//==         //==         }
//== 
//==         //==         double_complex z(0, 0);
//==         //==         for (int xi = 0; xi < nmt; xi++) {
//==         //==             int lm_aw    = type.indexb(xi).lm;
//==         //==             int idxrf_aw = type.indexb(xi).idxrf;
//==         //==             auto& gc = gaunt_coefs_->gaunt_vector(lm_lo, lm_aw);
//==         //==             z += atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf_aw, gc) * tmp(xi, ibnd - N__);
//==         //==         }
//==         //==         /* lo-APW contribution */
//==         //==         hphi__(j, ibnd) += z;
//==         //==         
//==         //==         //==for (int xi = 0; xi < nmt; xi++) {
//==         //==         //==    int lm_aw    = type.indexb(xi).lm;
//==         //==         //==    int idxrf_aw = type.indexb(xi).idxrf;
//== 
//==         //==         //==    auto z = phi__(j, ibnd) * atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_aw, idxrf_lo, gaunt_coefs_->gaunt_vector(lm_aw, lm_lo));
//==         //==         //==    /* APW-lo contribution */
//==         //==         //==    for (int ig = 0; ig < kp__->num_gkvec(); ig++) {
//==         //==         //==        hphi__(ig, ibnd) += alm(ig, xi) * z; 
//==         //==         //==    }
//==         //==         //==}
//== 
//==         //==         //== /* lo-lo contribution */
//==         //==         //== for (int jlo = 0; jlo < type.mt_lo_basis_size(); jlo++) {
//==         //==         //==     int j1     = jlo + kp__->num_gkvec() + atom.offset_lo();
//==         //==         //==     int lm1    = type.indexb(offs + jlo).lm;
//==         //==         //==     int order1 = type.indexb(offs + jlo).order;
//==         //==         //==     int idxrf1 = type.indexb(offs + jlo).idxrf;
//==         //==         //==     if (lm_lo == lm1) {
//==         //==         //==         ophi__(j, ibnd) += phi__(j1, ibnd) * atom.symmetry_class().o_radial_integral(l_lo, order_lo, order1);
//==         //==         //==     }
//==         //==         //==     hphi__(j, ibnd) +=  phi__(j1, ibnd) * atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf1, gaunt_coefs_->gaunt_vector(lm_lo, lm1));
//==         //==         //== }
//==         //==     }
//==         //== }
//==     }
//==     //for (int i = 0; i < n__; i++) {
//==     //    std::memcpy(hphi__.lo_coeffs().at<CPU>(0, N__ + i),
//==     //                &hphi_lo(0, i),
//==     //                hphi__.num_lo_loc() * sizeof(double_complex));
//==     //    
//==     //    std::memcpy(ophi__.lo_coeffs().at<CPU>(0, N__ + i),
//==     //                &ophi_lo(0, i),
//==     //                ophi__.num_lo_loc() * sizeof(double_complex));
//==     //}
}

inline void Band::diag_fv_full_potential(K_point* kp, Periodic_function<double>* effective_potential) const
{
    auto& itso = ctx_.iterative_solver_input_section();
    if (itso.type_ == "exact") {
        Band::diag_fv_full_potential_exact(kp, effective_potential);
    } else if (itso.type_ == "davidson") {
        Band::diag_fv_full_potential_davidson(kp, effective_potential);
    }
}

inline void Band::diag_fv_full_potential_exact(K_point* kp, Periodic_function<double>* effective_potential) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_fv_full_potential_exact");

    if (kp->num_ranks() > 1 && !gen_evp_solver()->parallel()) {
        TERMINATE("eigen-value solver is not parallel");
    }

    int ngklo = kp->gklo_basis_size();
    int bs = ctx_.cyclic_block_size();
    dmatrix<double_complex> h(ngklo, ngklo, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> o(ngklo, ngklo, ctx_.blacs_grid(), bs, bs);
    
    /* setup Hamiltonian and overlap */
    switch (ctx_.processing_unit()) {
        case CPU: {
            set_fv_h_o<CPU, electronic_structure_method_t::full_potential_lapwlo>(kp, effective_potential, h, o);
            break;
        }
        #ifdef __GPU
        case GPU: {
            set_fv_h_o<GPU, electronic_structure_method_t::full_potential_lapwlo>(kp, effective_potential, h, o);
            break;
        }
        #endif
        default: {
            TERMINATE("wrong processing unit");
        }
    }

    // TODO: move debug code to a separate function
    #if (__VERIFICATION > 0)
    if (!gen_evp_solver()->parallel())
    {
        Utils::check_hermitian("h", h.panel());
        Utils::check_hermitian("o", o.panel());
    }
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    auto z1 = h.checksum();
    auto z2 = o.checksum();
    DUMP("checksum(h): %18.10f %18.10f", std::real(z1), std::imag(z1));
    DUMP("checksum(o): %18.10f %18.10f", std::real(z2), std::imag(z2));
    #endif

    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(h): %16llX", h.panel().hash());
    DUMP("hash(o): %16llX", o.panel().hash());
    #endif

    assert(kp->gklo_basis_size() > ctx_.num_fv_states());
    
    std::vector<double> eval(ctx_.num_fv_states());
    
    runtime::Timer t("sirius::Band::diag_fv_full_potential|genevp");
    
    if (gen_evp_solver()->solve(kp->gklo_basis_size(), ctx_.num_fv_states(), h.at<CPU>(), h.ld(), o.at<CPU>(), o.ld(), 
                                &eval[0], kp->fv_eigen_vectors().prime().at<CPU>(), kp->fv_eigen_vectors().prime().ld(),
                                kp->gklo_basis_size_row(), kp->gklo_basis_size_col()))

    {
        TERMINATE("error in generalized eigen-value problem");
    }
    kp->set_fv_eigen_values(&eval[0]);
    
    //== Wave_functions<true> hphi(kp->gklo_basis_size(), ctx_.num_fv_states(), ctx_.cyclic_block_size(), ctx_.blacs_grid(), ctx_.blacs_grid_slice());
    //== Wave_functions<true> ophi(kp->gklo_basis_size(), ctx_.num_fv_states(), ctx_.cyclic_block_size(), ctx_.blacs_grid(), ctx_.blacs_grid_slice());

    //== apply_fv_h_o(kp, effective_potential, kp->fv_eigen_vectors(), hphi, ophi);

    //== matrix<double_complex> ovlp(ctx_.num_fv_states(), ctx_.num_fv_states());
    //== matrix<double_complex> hmlt(ctx_.num_fv_states(), ctx_.num_fv_states());
    //== linalg<CPU>::gemm(2, 0, ctx_.num_fv_states(), ctx_.num_fv_states(), kp->gklo_basis_size(), kp->fv_eigen_vectors().coeffs(), ophi.coeffs(), ovlp);
    //== linalg<CPU>::gemm(2, 0, ctx_.num_fv_states(), ctx_.num_fv_states(), kp->gklo_basis_size(), kp->fv_eigen_vectors().coeffs(), hphi.coeffs(), hmlt);

    //== for (int i = 0; i < ctx_.num_fv_states(); i++) {
    //==     for (int j = 0; j < ctx_.num_fv_states(); j++) {
    //==         double_complex z = (i == j) ? ovlp(i, j) - 1.0 : ovlp(i, j);
    //==         double_complex z1 = (i == j) ? hmlt(i, j) - eval[i] : hmlt(i, j);
    //==         if (std::abs(z) > 1e-10) {
    //==             printf("ovlp(%i, %i) = %f %f\n", i, j, z.real(), z.imag());
    //==         }
    //==         if (std::abs(z1) > 1e-10) {
    //==             printf("hmlt(%i, %i) = %f %f\n", i, j, z1.real(), z1.imag());
    //==         }
    //==     }
    //== }
    //== STOP();
}

inline int Band::residuals(K_point* kp__,
                           int N__,
                           int num_bands__,
                           std::vector<double>& eval__,
                           std::vector<double>& eval_old__,
                           matrix<double_complex>& evec__,
                           wave_functions& hphi__,
                           wave_functions& ophi__,
                           wave_functions& hpsi__,
                           wave_functions& opsi__,
                           wave_functions& res__,
                           std::vector<double>& h_diag__,
                           std::vector<double>& o_diag__) const
{
    PROFILE_WITH_TIMER("sirius::Band::residuals");

    auto& itso = ctx_.iterative_solver_input_section();
    bool converge_by_energy = (itso.converge_by_energy_ == 1);

    /* norm of residuals */
    std::vector<double> res_norm(num_bands__);

    int n = 0;
    if (converge_by_energy) {
        STOP();

        //std::vector<double> eval_tmp(num_bands__);

        ///* main trick here: first estimate energy difference, and only then compute unconverged residuals */
        //for (int i = 0; i < num_bands__; i++)
        //{
        //    bool take_res = true;
        //    if (itso.converge_occupied_ && kp__->band_occupancy(i + ispn__ * ctx_.num_fv_states()) < 1e-10) take_res = false;

        //    if (take_res && std::abs(eval__[i] - eval_old__[i]) > itso.energy_tolerance_)
        //    {
        //        std::memcpy(&evec__(0, num_bands__ + n), &evec__(0, i), N__ * sizeof(T));
        //        eval_tmp[n++] = eval__[i];
        //    }
        //}
        //// TODO: do this on GPU

        ///* create alias for eigen-vectors corresponding to unconverged residuals */
        //matrix<T> evec_tmp;
        //if (ctx_.processing_unit() == CPU)
        //{
        //    evec_tmp = matrix<T>(&evec__(0, num_bands__), evec__.ld(), n);
        //}
        //#ifdef __GPU
        //if (ctx_.processing_unit() == GPU)
        //{
        //    evec_tmp = matrix<T>(evec__.template at<CPU>(0, num_bands__), evec__.template at<GPU>(0, num_bands__), evec__.ld(), n);
        //    /* copy matrix of eigen-vectors to GPU */
        //    acc::copyin(evec_tmp.template at<GPU>(), evec_tmp.ld(), evec_tmp.template at<CPU>(), evec_tmp.ld(), N__, n);
        //}
        //#endif

        ///* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
        //hpsi__.transform_from<T>(hphi__, N__, evec_tmp, n);
        ///* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        //opsi__.transform_from<T>(ophi__, N__, evec_tmp, n);

        //residuals_aux(kp__, n, eval_tmp, hpsi__, opsi__, res__, h_diag__, o_diag__, res_norm);

        //int nmax = n;
        //n = 0;
        //for (int i = 0; i < nmax; i++)
        //{
        //    /* take the residual if it's norm is above the threshold */
        //    if (res_norm[i] > itso.residual_tolerance_)
        //    {
        //        /* shift unconverged residuals to the beginning of array */
        //        if (n != i)
        //        {
        //            switch (ctx_.processing_unit())
        //            {
        //                case CPU:
        //                {
        //                    std::memcpy(&res__(0, n), &res__(0, i), res__.num_gvec_loc() * sizeof(double_complex));
        //                    break;
        //                }
        //                case GPU:
        //                {
        //                    #ifdef __GPU
        //                    acc::copy(res__.coeffs().at<GPU>(0, n), res__.coeffs().at<GPU>(0, i), res__.num_gvec_loc());
        //                    #else
        //                    TERMINATE_NO_GPU
        //                    #endif
        //                    break;
        //                }
        //            }
        //        }
        //        n++;
        //    }
        //}
        //#if (__VERBOSITY > 2)
        //if (kp__->comm().rank() == 0)
        //{
        //    DUMP("initial and final number of residuals : %i %i", nmax, n);
        //}
        //#endif
    } else {
        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
        hpsi__.transform_from<double_complex>(hphi__, N__, evec__, num_bands__);
        /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        opsi__.transform_from<double_complex>(ophi__, N__, evec__, num_bands__);

        residuals_aux(kp__, num_bands__, eval__, hpsi__, opsi__, res__, h_diag__, o_diag__, res_norm);

        for (int i = 0; i < num_bands__; i++) {
            /* take the residual if it's norm is above the threshold */
            if (res_norm[i] > itso.residual_tolerance_) {
                /* shift unconverged residuals to the beginning of array */
                if (n != i) {
                    res__.copy_from(res__, i, 1, n);
                }
                n++;
            }
        }
    }

    return n;
}

inline void Band::diag_fv_full_potential_davidson(K_point* kp, Periodic_function<double>* effective_potential) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_fv_full_potential_davidson");

    STOP();

    //auto h_diag = get_h_diag(kp, effective_potential->f_pw(0).real(), ctx_.step_function().theta_pw(0).real());
    //auto o_diag = get_o_diag(kp, ctx_.step_function().theta_pw(0).real());

    ///* short notation for number of target wave-functions */
    //int num_bands = ctx_.num_fv_states();

    //auto& itso = ctx_.iterative_solver_input_section();

    ///* short notation for target wave-functions */
    //Wave_functions<false> psi(kp->gklo_basis_size(), num_bands, CPU);
    //kp->fv_eigen_vectors().prime() >> psi.prime();

    //lapwlo_wave_functions psi1(kp, ctx_, num_bands);
    //psi1.apw_coeffs().prime().zero();
    //psi1.lo_coeffs().zero();
    //for (int i = 0; i < num_bands; i++) {
    //    for (int ig = 0; ig < kp->gkvec().gvec_count(kp->comm().rank()); ig++) {
    //        if (ig + kp->gkvec().gvec_offset(kp->comm().rank()) == i) {
    //            psi1.apw_coeffs()(ig, i) = 1.0;
    //        }
    //        if (ig + kp->gkvec().gvec_offset(kp->comm().rank()) == i + 1) {
    //            psi1.apw_coeffs()(ig, i) = 0.5;
    //        }
    //        if (ig + kp->gkvec().gvec_offset(kp->comm().rank()) == i + 2) {
    //            psi1.apw_coeffs()(ig, i) = 0.25;
    //        }
    //    }
    //}

    //bool converge_by_energy = (itso.converge_by_energy_ == 1);
    //
    //assert(num_bands * 2 < kp->num_gkvec()); // iterative subspace size can't be smaller than this

    ///* number of auxiliary basis functions */
    //int num_phi = std::min(itso.subspace_size_ * num_bands, kp->num_gkvec());

    ///* allocate wave-functions */
    //Wave_functions<false>  phi(kp->gklo_basis_size(), num_phi, CPU);
    //Wave_functions<false> hphi(kp->gklo_basis_size(), num_phi, CPU);
    //Wave_functions<false> ophi(kp->gklo_basis_size(), num_phi, CPU);
    //Wave_functions<false> hpsi(kp->gklo_basis_size(), num_bands, CPU);
    //Wave_functions<false> opsi(kp->gklo_basis_size(), num_bands, CPU);

    //lapwlo_wave_functions  phi1(kp, ctx_, num_phi);
    //lapwlo_wave_functions hphi1(kp, ctx_, num_phi);
    //lapwlo_wave_functions ophi1(kp, ctx_, num_phi);

    ///* residuals */
    //Wave_functions<false> res(kp->gklo_basis_size(), num_bands, CPU);

    //auto mem_type = (gen_evp_solver_->type() == ev_magma) ? memory_t::host_pinned : memory_t::host;

    ///* allocate Hamiltonian and overlap */
    //matrix<double_complex> hmlt(num_phi, num_phi, mem_type);
    //matrix<double_complex> ovlp(num_phi, num_phi, mem_type);
    //matrix<double_complex> hmlt_old(num_phi, num_phi, mem_type);
    //matrix<double_complex> ovlp_old(num_phi, num_phi, mem_type);

    //matrix<double_complex> evec(num_phi, num_phi);

    //int bs = ctx_.cyclic_block_size();

    //dmatrix<double_complex> hmlt_dist;
    //dmatrix<double_complex> ovlp_dist;
    //dmatrix<double_complex> evec_dist;
    //if (kp->comm().size() == 1) {
    //    hmlt_dist = dmatrix<double_complex>(&hmlt(0, 0), num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    //    ovlp_dist = dmatrix<double_complex>(&ovlp(0, 0), num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    //    evec_dist = dmatrix<double_complex>(&evec(0, 0), num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    //} else {
    //    hmlt_dist = dmatrix<double_complex>(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    //    ovlp_dist = dmatrix<double_complex>(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    //    evec_dist = dmatrix<double_complex>(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    //}

    //std::vector<double> eval(num_bands);
    //for (int i = 0; i < num_bands; i++) {
    //    eval[i] = kp->band_energy(i);
    //}
    //std::vector<double> eval_old(num_bands);
    //
    ///* trial basis functions */
    //phi.copy_from(psi, 0, num_bands);
    //
    //phi1.copy_from(psi1, 0, num_bands);

    ///* current subspace size */
    //int N = 0;

    ///* number of newly added basis functions */
    //int n = num_bands;

    //#if (__VERBOSITY > 2)
    //if (kp->comm().rank() == 0) {
    //    DUMP("iterative solver tolerance: %18.12f", ctx_.iterative_solver_tolerance());
    //}
    //#endif

    //#ifdef __PRINT_MEMORY_USAGE
    //MEMORY_USAGE_INFO();
    //#ifdef __GPU
    //gpu_mem = cuda_get_free_mem() >> 20;
    //printf("[rank%04i at line %i of file %s] CUDA free memory: %i Mb\n", mpi_comm_world().rank(), __LINE__, __FILE__, gpu_mem);
    //#endif
    //#endif
    //
    ///* start iterative diagonalization */
    //for (int k = 0; k < itso.num_steps_; k++) {
    //    /* apply Hamiltonian and overlap operators to the new basis functions */
    //    apply_fv_h_o(kp, effective_potential, N, n, phi1, hphi1, ophi1);
    //    
    //    orthogonalize(kp, N, n, phi1, hphi1, ophi1, ovlp);

    //    /* setup eigen-value problem
    //     * N is the number of previous basis functions
    //     * n is the number of new basis functions */
    //    set_h_o<double_complex>(kp, N, n, phi, hphi, ophi, hmlt, ovlp, hmlt_old, ovlp_old);

    //    /* increase size of the variation space */
    //    N += n;

    //    eval_old = eval;

    //    /* solve generalized eigen-value problem with the size N */
    //    diag_h_o<double_complex>(kp, N, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);
    //    
    //    #if (__VERBOSITY > 2)
    //    if (kp->comm().rank() == 0) {
    //        DUMP("step: %i, current subspace size: %i, maximum subspace size: %i", k, N, num_phi);
    //        for (int i = 0; i < num_bands; i++) DUMP("eval[%i]=%20.16f, diff=%20.16f", i, eval[i], std::abs(eval[i] - eval_old[i]));
    //    }
    //    #endif

    //    /* don't compute residuals on last iteration */
    //    if (k != itso.num_steps_ - 1) {
    //        /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
    //        n = residuals(kp, N, num_bands, eval, eval_old, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag);
    //    }

    //    /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
    //    if (N + n > num_phi || n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {   
    //        runtime::Timer t1("sirius::Band::diag_pseudo_potential_davidson|update_phi");
    //        /* recompute wave-functions */
    //        /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
    //        psi.transform_from<double_complex>(phi, N, evec, num_bands);

    //        /* exit the loop if the eigen-vectors are converged or this is a last iteration */
    //        if (n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {
    //            break;
    //        }
    //        else { /* otherwise, set Psi as a new trial basis */
    //            #if (__VERBOSITY > 2)
    //            if (kp->comm().rank() == 0) {
    //                DUMP("subspace size limit reached");
    //            }
    //            #endif
    //            hmlt_old.zero();
    //            ovlp_old.zero();
    //            for (int i = 0; i < num_bands; i++) {
    //                hmlt_old(i, i) = eval[i];
    //                ovlp_old(i, i) = 1.0;
    //            }

    //            /* need to compute all hpsi and opsi states (not only unconverged) */
    //            if (converge_by_energy) {
    //                hpsi.transform_from<double_complex>(hphi, N, evec, num_bands);
    //                opsi.transform_from<double_complex>(ophi, N, evec, num_bands);
    //            }
 
    //            /* update basis functions */
    //            phi.copy_from(psi, 0, num_bands);
    //            /* update hphi and ophi */
    //            hphi.copy_from(hpsi, 0, num_bands);
    //            ophi.copy_from(opsi, 0, num_bands);
    //            /* number of basis functions that we already have */
    //            N = num_bands;
    //        }
    //    }
    //    /* expand variational subspace with new basis vectors obtatined from residuals */
    //    phi.copy_from(res, 0, n, N);
    //}

    //kp->set_fv_eigen_values(&eval[0]);
    //psi.prime() >> kp->fv_eigen_vectors().prime();

    //kp->comm().barrier();
}

#include "band.hpp"

}

#endif // __BAND_H__
