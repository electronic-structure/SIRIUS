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

#include "global.h"
#include "periodic_function.h"
#include "k_point.h"

namespace sirius
{

// TODO: Band problem is a mess and needs more formal organizaiton. We have different basis functions. 
//       We can do first- and second-variation or a full variation. We can do iterative or exact diagonalization.
//       This has to be organized. 

/// Setup and solve eigen value problem.
class Band
{
    private:

        /// global set of parameters
        Global& parameters_;

        BLACS_grid const& blacs_grid_;

        /// alias for FFT driver
        FFT3D<cpu>* fft_;
        
        /// Non-zero Gaunt coefficients
        Gaunt_coefficients<double_complex>* gaunt_coefs_;

        standard_evp* std_evp_solver_; 

        generalized_evp* gen_evp_solver_;

        /// Apply effective magentic field to the first-variational state.
        /** Must be called first because hpsi is overwritten with B|fv_j>. */
        void apply_magnetic_field(mdarray<double_complex, 2>& fv_states, int num_gkvec, int* fft_index, 
                                  Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 3>& hpsi);

        /// Apply SO correction to the first-variational states.
        /** Raising and lowering operators:
            \f[
                L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
            \f]
        */
        void apply_so_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi);
        
        /// Apply UJ correction to scalar wave functions
        template <spin_block_t sblock>
        void apply_uj_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi);

        /// Add interstitial contribution to apw-apw block of Hamiltonian and overlap
        void set_fv_h_o_it(K_point* kp, Periodic_function<double>* effective_potential, 
                           mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o);

        void set_o_it(K_point* kp, mdarray<double_complex, 2>& o);

        template <spin_block_t sblock>
        void set_h_it(K_point* kp, Periodic_function<double>* effective_potential, 
                      Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 2>& h);
        
        /// Setup lo-lo block of Hamiltonian and overlap matrices
        void set_fv_h_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o);

        template <spin_block_t sblock>
        void set_h_lo_lo(K_point* kp, mdarray<double_complex, 2>& h);
        
        void set_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& o);
       
        void set_o(K_point* kp, mdarray<double_complex, 2>& o);
    
        template <spin_block_t sblock> 
        void set_h(K_point* kp, Periodic_function<double>* effective_potential, 
                   Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 2>& h);
       
        void diag_fv_full_potential(K_point* kp, Periodic_function<double>* effective_potential);

        void apply_h_local_slice(K_point* kp__,
                                 std::vector<double>& effective_potential__,
                                 std::vector<double>& pw_ekin__,
                                 int num_phi__,
                                 mdarray<double_complex, 2>& phi__,
                                 mdarray<double_complex, 2>& hphi__);

        void apply_h_local(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, 
                           int n, double_complex* phi__, double_complex* hphi__);

        void diag_fv_uspp_cpu(K_point* kp__,
                              Periodic_function<double>* effective_potential__);
        
        void diag_fv_uspp_cpu_serial(K_point* kp__,
                                     double v0__,
                                     std::vector<double>& veff_it_coarse__);

        void diag_fv_uspp_cpu_serial_v0(K_point* kp__,
                                        std::vector<double>& veff_it_coarse__);

        void diag_fv_uspp_cpu_serial_v1(K_point* kp__,
                                        double v0__,
                                        std::vector<double>& veff_it_coarse__);

        void diag_fv_uspp_cpu_serial_v2(K_point* kp__,
                                        double v0__,
                                        std::vector<double>& veff_it_coarse__);

        void diag_fv_uspp_cpu_serial_v3(K_point* kp__,
                                        double v0__,
                                        std::vector<double>& veff_it_coarse__);

        void diag_fv_uspp_cpu_serial_v4(K_point* kp__,
                                        double v0__,
                                        std::vector<double>& veff_it_coarse__);

        void diag_fv_uspp_cpu_parallel(K_point* kp__,
                                     double v0__,
                                     std::vector<double>& veff_it_coarse__);
 
        void get_h_o_diag(K_point* kp__,
                          double v0__,
                          std::vector<double>& pw_ekin__,
                          std::vector<double_complex>& h_diag__,
                          std::vector<double_complex>& o_diag__);

        void apply_h_o_uspp_cpu(K_point* kp__, 
                                std::vector<double>& effective_potential__,
                                std::vector<double>& pw_ekin__,
                                int n__,
                                double_complex* phi__, 
                                double_complex* hphi__, 
                                double_complex* ophi__);

        void apply_h_local_parallel(K_point* kp__,
                                    std::vector<double>& effective_potential__,
                                    std::vector<double>& pw_ekin__,
                                    int N__,
                                    int n__,
                                    dmatrix<double_complex>& phi__,
                                    dmatrix<double_complex>& hphi__);

        void set_fv_h_o_uspp_cpu_parallel_simple(int N__,
                                                 int n__,
                                                 K_point* kp__,
                                                 std::vector<double>& veff_it_coarse__,
                                                 std::vector<double>& pw_ekin__,
                                                 dmatrix<double_complex>& phi__,
                                                 dmatrix<double_complex>& hphi__,
                                                 dmatrix<double_complex>& ophi__,
                                                 dmatrix<double_complex>& h__,
                                                 dmatrix<double_complex>& o__,
                                                 dmatrix<double_complex>& h_old__,
                                                 dmatrix<double_complex>& o_old__);

        //= void set_fv_h_o_uspp_cpu_parallel_v2(int N__,
        //=                                   int n__,
        //=                                   K_point* kp__,
        //=                                   std::vector<double>& veff_it_coarse__,
        //=                                   std::vector<double>& pw_ekin__,
        //=                                   dmatrix<double_complex>& phi__,
        //=                                   dmatrix<double_complex>& hphi__,
        //=                                   dmatrix<double_complex>& ophi__,
        //=                                   dmatrix<double_complex>& h__,
        //=                                   dmatrix<double_complex>& o__,
        //=                                   dmatrix<double_complex>& h_old__,
        //=                                   dmatrix<double_complex>& o_old__);

        void set_fv_h_o_uspp_cpu_parallel_v3(int N__,
                                          int n__,
                                          K_point* kp__,
                                          std::vector<double>& veff_it_coarse__,
                                          std::vector<double>& pw_ekin__,
                                          dmatrix<double_complex>& phi__,
                                          dmatrix<double_complex>& hphi__,
                                          dmatrix<double_complex>& ophi__,
                                          dmatrix<double_complex>& h__,
                                          dmatrix<double_complex>& o__,
                                          dmatrix<double_complex>& h_old__,
                                          dmatrix<double_complex>& o_old__,
                                          int num_atoms_in_block__,
                                          mdarray<double_complex, 2>& beta_pw__);

        void uspp_residuals_cpu_parallel_simple(int N__,
                                                int num_bands__,
                                                K_point* kp__,
                                                std::vector<double>& eval__,
                                                dmatrix<double_complex>& evec__,
                                                dmatrix<double_complex>& hphi__,
                                                dmatrix<double_complex>& ophi__,
                                                dmatrix<double_complex>& hpsi__,
                                                dmatrix<double_complex>& opsi__,
                                                dmatrix<double_complex>& res__,
                                                std::vector<double_complex>& h_diag__,
                                                std::vector<double_complex>& o_diag__,
                                                std::vector<double>& res_norm__);

        void uspp_cpu_residuals_parallel_v2(int N__,
                                            int num_bands__,
                                            K_point* kp__,
                                            std::vector<double>& eval__,
                                            dmatrix<double_complex>& evec__,
                                            dmatrix<double_complex>& hphi__,
                                            dmatrix<double_complex>& ophi__,
                                            dmatrix<double_complex>& hpsi__,
                                            dmatrix<double_complex>& opsi__,
                                            dmatrix<double_complex>& res__,
                                            std::vector<double_complex>& h_diag__,
                                            std::vector<double_complex>& o_diag__,
                                            std::vector<double>& res_norm__);

        void apply_h_o_uspp_cpu_parallel_simple(K_point* kp__,
                                                std::vector<double>& effective_potential__,
                                                std::vector<double>& pw_ekin__,
                                                int N__,
                                                int n__,
                                                dmatrix<double_complex>& phi__,
                                                dmatrix<double_complex>& hphi__,
                                                dmatrix<double_complex>& ophi__);

        void apply_h_o_uspp_cpu_parallel_v2(K_point* kp__,
                                         std::vector<double>& effective_potential__,
                                         std::vector<double>& pw_ekin__,
                                         int N__,
                                         int n__,
                                         dmatrix<double_complex>& phi__,
                                         dmatrix<double_complex>& hphi__,
                                         dmatrix<double_complex>& ophi__,
                                         int num_atoms_in_block__,
                                         mdarray<double_complex, 2>& beta_pw__);

        void uspp_residuals_cpu_parallel_v3(int N__,
                                            int num_bands__,
                                            K_point* kp__,
                                            std::vector<double>& eval__,
                                            dmatrix<double_complex>& evec__,
                                            dmatrix<double_complex>& hphi__,
                                            dmatrix<double_complex>& ophi__,
                                            dmatrix<double_complex>& hpsi__,
                                            dmatrix<double_complex>& opsi__,
                                            dmatrix<double_complex>& res__,
                                            std::vector<double_complex>& h_diag__,
                                            std::vector<double_complex>& o_diag__,
                                            std::vector<double>& res_norm__,
                                            mdarray<double_complex, 2>& kappa__);
        
        #ifdef _GPU_
        void uspp_residuals_gpu_parallel(int N__,
                                         int num_bands__,
                                         K_point* kp__,
                                         std::vector<double>& eval__,
                                         dmatrix<double_complex>& evec__,
                                         dmatrix<double_complex>& hphi__,
                                         dmatrix<double_complex>& ophi__,
                                         dmatrix<double_complex>& hpsi__,
                                         dmatrix<double_complex>& opsi__,
                                         dmatrix<double_complex>& res__,
                                         std::vector<double_complex>& h_diag__,
                                         std::vector<double_complex>& o_diag__,
                                         std::vector<double>& res_norm__,
                                         mdarray<double_complex, 2>& kappa__);

        void set_fv_h_o_uspp_gpu_parallel_v3(int N__,
                                             int n__,
                                             K_point* kp__,
                                             std::vector<double>& veff_it_coarse__,
                                             std::vector<double>& pw_ekin__,
                                             dmatrix<double_complex>& phi__,
                                             dmatrix<double_complex>& hphi__,
                                             dmatrix<double_complex>& ophi__,
                                             dmatrix<double_complex>& h__,
                                             dmatrix<double_complex>& o__,
                                             dmatrix<double_complex>& h_old__,
                                             dmatrix<double_complex>& o_old__,
                                             int num_atoms_in_block__,
                                             mdarray<double_complex, 2>& kappa__);

        void diag_fv_uspp_gpu_parallel(K_point* kp__,
                                       double v0__,
                                       std::vector<double>& veff_it_coarse__);

        void apply_h_local_gpu(K_point* kp__,
                               std::vector<double>& effective_potential__,
                               std::vector<double>& pw_ekin__, 
                               int num_phi__,
                               mdarray<double_complex, 2>& gamma__,
                               mdarray<double_complex, 2>& kappa__,
                               double_complex* phi__, 
                               double_complex* hphi__);

        void apply_h_o_uspp_gpu(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, int n,
                                mdarray<double_complex, 2>& gamma, mdarray<double_complex, 2>& kappa, double_complex* phi__, 
                                double_complex* hphi__, double_complex* ophi__);

        void diag_fv_uspp_gpu(K_point* kp, Periodic_function<double>* effective_potential);

        void apply_h_o_uspp_gpu_parallel_v2(K_point* kp__,
                                         std::vector<double>& effective_potential__,
                                         std::vector<double>& pw_ekin__,
                                         int N__,
                                         int n__,
                                         dmatrix<double_complex>& phi__,
                                         dmatrix<double_complex>& hphi__,
                                         dmatrix<double_complex>& ophi__,
                                         int num_atoms_in_block__,
                                         mdarray<double_complex, 2>& kappa__);
        
        #endif

    public:
        
        /// Constructor
        Band(Global& parameters__, BLACS_grid const& blacs_grid__) 
            : parameters_(parameters__),
              blacs_grid_(blacs_grid__)
        {
            fft_ = parameters_.reciprocal_lattice()->fft();

            gaunt_coefs_ = new Gaunt_coefficients<double_complex>(parameters_.lmax_apw(), 
                                                                  parameters_.lmax_pot(), 
                                                                  parameters_.lmax_apw());

            /* create standard eigen-value solver */
            switch (parameters_.std_evp_solver_type())
            {
                case ev_lapack:
                {
                    std_evp_solver_ = new standard_evp_lapack();
                    break;
                }
                case ev_scalapack:
                {
                    std_evp_solver_ = new standard_evp_scalapack(blacs_grid_);
                    break;
                }
                case ev_plasma:
                {
                    std_evp_solver_ = new standard_evp_plasma();
                    break;
                }
                default:
                {
                    TERMINATE("wrong standard eigen-value solver");
                }
            }
            
            /* create generalized eign-value solver */
            switch (parameters_.gen_evp_solver_type())
            {
                case ev_lapack:
                {
                    gen_evp_solver_ = new generalized_evp_lapack(1e-15);
                    break;
                }
                case ev_scalapack:
                {
                    gen_evp_solver_ = new generalized_evp_scalapack(blacs_grid_, 1e-15);
                    break;
                }
                case ev_elpa1:
                {
                    gen_evp_solver_ = new generalized_evp_elpa1(blacs_grid_);
                    break;
                }
                case ev_elpa2:
                {
                    gen_evp_solver_ = new generalized_evp_elpa2(blacs_grid_);
                    break;
                }
                case ev_magma:
                {
                    gen_evp_solver_ = new generalized_evp_magma();
                    break;
                }
                case ev_rs_gpu:
                {
                    gen_evp_solver_ = new generalized_evp_rs_gpu(blacs_grid_);
                    break;
                }
                case ev_rs_cpu:
                {
                    gen_evp_solver_ = new generalized_evp_rs_cpu(blacs_grid_);
                    break;
                }
                default:
                {
                    TERMINATE("wrong generalized eigen-value solver");
                }
            }

            if (std_evp_solver_->parallel() != gen_evp_solver_->parallel())
                error_global(__FILE__, __LINE__, "both eigen-value solvers must be serial or parallel");
        }

        ~Band()
        {
            delete gaunt_coefs_;
            delete std_evp_solver_;
            delete gen_evp_solver_;
        }

        /// Apply the muffin-tin part of the first-variational Hamiltonian to the apw basis function
        /** The following vector is computed:
         *  \f[
         *    b_{L_2 \nu_2}^{\alpha}({\bf G'}) = \sum_{L_1 \nu_1} \sum_{L_3} 
         *      a_{L_1\nu_1}^{\alpha*}({\bf G'}) 
         *      \langle u_{\ell_1\nu_1}^{\alpha} | h_{L3}^{\alpha} |  u_{\ell_2\nu_2}^{\alpha}  
         *      \rangle  \langle Y_{L_1} | R_{L_3} | Y_{L_2} \rangle +  
         *      \frac{1}{2} \sum_{\nu_1} a_{L_2\nu_1}^{\alpha *}({\bf G'})
         *      u_{\ell_2\nu_1}^{\alpha}(R_{\alpha})
         *      u_{\ell_2\nu_2}^{'\alpha}(R_{\alpha})R_{\alpha}^{2}
         *  \f] 
         */
        template <spin_block_t sblock>
        void apply_hmt_to_apw(int num_gkvec, int ia, mdarray<double_complex, 2>& alm, mdarray<double_complex, 2>& halm);
 
        template <spin_block_t sblock>
        void apply_hmt_to_apw(mdarray<double_complex, 2>& alm, mdarray<double_complex, 2>& halm);

        /// Setup apw-lo and lo-apw blocs of Hamiltonian and overlap matrices
        void set_fv_h_o_apw_lo(K_point* kp,
                               Atom_type* type,
                               Atom* atom,
                               int ia,
                               mdarray<double_complex, 2>& alm_row,
                               mdarray<double_complex, 2>& alm_col,
                               mdarray<double_complex, 2>& h,
                               mdarray<double_complex, 2>& o);
        
        template <spin_block_t sblock>
        void set_h_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
                          mdarray<double_complex, 2>& h);

        void set_o_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
                          mdarray<double_complex, 2>& o);

        /// Setup the Hamiltonian and overlap matrices in APW+lo basis
        /** The Hamiltonian matrix has the following expression:
         *  \f[
         *      H_{\mu' \mu} = \langle \varphi_{\mu'} | \hat H | \varphi_{\mu} \rangle
         *  \f]
         *
         *  \f[
         *      H_{\mu' \mu}=\langle \varphi_{\mu' } | \hat H | \varphi_{\mu } \rangle  = 
         *      \left( \begin{array}{cc} 
         *         H_{\bf G'G} & H_{{\bf G'}j} \\
         *         H_{j'{\bf G}} & H_{j'j}
         *      \end{array} \right)
         *  \f]
         *  
         *  The overlap matrix has the following expression:
         *  \f[
         *      O_{\mu' \mu} = \langle \varphi_{\mu'} | \varphi_{\mu} \rangle
         *  \f]
         *  APW-APW block:
         *  \f[
         *      O_{{\bf G'} {\bf G}}^{\bf k} = \sum_{\alpha} \sum_{L\nu} a_{L\nu}^{\alpha *}({\bf G'+k}) 
         *      a_{L\nu}^{\alpha}({\bf G+k})
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
         */
        template <processing_unit_t pu, electronic_structure_method_t basis>
        void set_fv_h_o(K_point* kp, Periodic_function<double>* effective_potential, dmatrix<double_complex>& h, 
                        dmatrix<double_complex>& o);

        /// Solve first-variational problem
        void solve_fv(K_point* kp__, Periodic_function<double>* effective_potential__);

        /// Solve second-variational problem
        void solve_sv(K_point* kp, Periodic_function<double>* effective_magnetic_field[3]);

        void solve_fd(K_point* kp, Periodic_function<double>* effective_potential, 
                      Periodic_function<double>* effective_magnetic_field[3]);

        inline standard_evp* std_evp_solver()
        {
            return std_evp_solver_;
        }

        inline generalized_evp* gen_evp_solver()
        {
            return gen_evp_solver_;
        }
};

#include "band.hpp"

}

#endif // __BAND_H__
