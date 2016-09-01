// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file density.h
 *   
 *  \brief Contains definition and partial implementation of sirius::Density class.
 */

#ifndef __DENSITY_H__
#define __DENSITY_H__

#include "periodic_function.h"
#include "k_set.h"
#include "simulation_context.h"

#ifdef __GPU
extern "C" void generate_dm_pw_gpu(int num_atoms__,
                                   int num_gvec_loc__,
                                   int num_beta__,
                                   double const* atom_pos__,
                                   int const* gvec__,
                                   double* phase_factors__,
                                   double const* dm__,
                                   double* dm_pw__,
                                   int stream_id__);

extern "C" void sum_q_pw_dm_pw_gpu(int num_gvec_loc__,
                                   int nbf__,
                                   double const* q_pw__,
                                   double const* dm_pw__,
                                   double const* sym_weight__,
                                   double_complex* rho_pw__,
                                   int stream_id__);

extern "C" void update_density_rg_1_gpu(int size__, 
                                        cuDoubleComplex const* psi_rg__, 
                                        double wt__, 
                                        double* density_rg__);
#endif

namespace sirius
{

/// Generate charge density and magnetization from occupied spinor wave-functions.
/** Let's start from the definition of the complex density matrix:
 *  \f[
 *      \rho_{\sigma' \sigma}({\bf r}) =
 *       \sum_{j{\bf k}} n_{j{\bf k}} \Psi_{j{\bf k}}^{\sigma*}({\bf r}) \Psi_{j{\bf k}}^{\sigma'}({\bf r}) = 
 *       \frac{1}{2} \left( \begin{array}{cc} \rho({\bf r})+m_z({\bf r}) & 
 *              m_x({\bf r})-im_y({\bf r}) \\ m_x({\bf r})+im_y({\bf r}) & \rho({\bf r})-m_z({\bf r}) \end{array} \right)
 *  \f]
 *  We notice that the diagonal components of the density matrix are actually real and the off-diagonal components are
 *  expressed trough two independent functions \f$ m_x({\bf r}) \f$ and \f$ m_y({\bf r}) \f$. Having this in mind we 
 *  will work with a slightly different object, namely a real density matrix, defined as a 1-, 2- or 4-dimensional 
 *  (depending on the number of magnetic components) vector with the following elements: 
 *      - \f$ [ \rho({\bf r}) ] \f$ in case of non-magnetic configuration
 *      - \f$ [ \rho_{\uparrow \uparrow}({\bf r}), \rho_{\downarrow \downarrow}({\bf r}) ]  = 
 *            [ \frac{\rho({\bf r})+m_z({\bf r})}{2}, \frac{\rho({\bf r})-m_z({\bf r})}{2} ] \f$ in case of collinear 
 *         magnetic configuration
 *      - \f$ [ \rho_{\uparrow \uparrow}({\bf r}), \rho_{\downarrow \downarrow}({\bf r}), 
 *              2 \Re \rho_{\uparrow \downarrow}({\bf r}), -2 \Im \rho_{\uparrow \downarrow}({\bf r}) ] = 
 *            [ \frac{\rho({\bf r})+m_z({\bf r})}{2}, \frac{\rho({\bf r})-m_z({\bf r})}{2}, 
 *              m_x({\bf r}),  m_y({\bf r}) ] \f$ in the general case of non-collinear magnetic configuration
 *  
 *  At this point it is straightforward to compute the density and magnetization in the interstitial (see add_k_point_contribution_rg()).
 *  The muffin-tin part of the density and magnetization is obtained in a slighlty more complicated way. Recall the
 *  expansion of spinor wave-functions inside the muffin-tin \f$ \alpha \f$
 *  \f[
 *      \Psi_{j{\bf k}}^{\sigma}({\bf r}) = \sum_{\xi}^{N_{\xi}^{\alpha}} {S_{\xi}^{\sigma j {\bf k},\alpha}} 
 *      f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)Y_{\ell_{\xi}m_{\xi}}(\hat {\bf r})
 *  \f]
 *  which we insert into expression for the complex density matrix: 
 *  \f[
 *      \rho_{\sigma' \sigma}({\bf r}) = \sum_{j{\bf k}} n_{j{\bf k}} \sum_{\xi}^{N_{\xi}^{\alpha}} 
 *          S_{\xi}^{\sigma j {\bf k},\alpha*} f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)
 *          Y_{\ell_{\xi}m_{\xi}}^{*}(\hat {\bf r}) \sum_{\xi'}^{N_{\xi'}^{\alpha}} S_{\xi'}^{\sigma' j{\bf k},\alpha}
 *          f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r)Y_{\ell_{\xi'}m_{\xi'}}(\hat {\bf r})
 *  \f]
 *  First, we eliminate a sum over bands and k-points by forming an auxiliary density tensor:
 *  \f[
 *      D_{\xi \sigma, \xi' \sigma'}^{\alpha} = \sum_{j{\bf k}} n_{j{\bf k}} S_{\xi}^{\sigma j {\bf k},\alpha*} 
 *          S_{\xi'}^{\sigma' j {\bf k},\alpha}
 *  \f]
 *  The expression for complex density matrix simplifies to:
 *  \f[
 *      \rho_{\sigma' \sigma}({\bf r}) =  \sum_{\xi \xi'} D_{\xi \sigma, \xi' \sigma'}^{\alpha} 
 *          f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)Y_{\ell_{\xi}m_{\xi}}^{*}(\hat {\bf r}) 
 *          f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r)Y_{\ell_{\xi'}m_{\xi'}}(\hat {\bf r})
 *  \f]
 *  Now we can switch to the real density matrix and write its' expansion in real spherical harmonics. Let's take
 *  non-magnetic case as an example:
 *  \f[
 *      \rho({\bf r}) = \sum_{\xi \xi'} D_{\xi \xi'}^{\alpha} 
 *          f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)Y_{\ell_{\xi}m_{\xi}}^{*}(\hat {\bf r}) 
 *          f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r)Y_{\ell_{\xi'}m_{\xi'}}(\hat {\bf r}) = 
 *          \sum_{\ell_3 m_3} \rho_{\ell_3 m_3}^{\alpha}(r) R_{\ell_3 m_3}(\hat {\bf r}) 
 *  \f]
 *  where
 *  \f[
 *      \rho_{\ell_3 m_3}^{\alpha}(r) = \sum_{\xi \xi'} D_{\xi \xi'}^{\alpha} f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r) 
 *          f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r) \langle Y_{\ell_{\xi}m_{\xi}} | R_{\ell_3 m_3} | Y_{\ell_{\xi'}m_{\xi'}} \rangle
 *  \f]
 *  We are almost done. Now it is time to switch to the full index notation  \f$ \xi \rightarrow \{ \ell \lambda m \} \f$
 *  and sum over \a m and \a m' indices:
 *  \f[
 *       \rho_{\ell_3 m_3}^{\alpha}(r) = \sum_{\ell \lambda, \ell' \lambda'} f_{\ell \lambda}^{\alpha}(r)  
 *          f_{\ell' \lambda'}^{\alpha}(r) d_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha} 
 *  \f]
 *  where
 *  \f[
 *      d_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha} = 
 *          \sum_{mm'} D_{\ell \lambda m, \ell' \lambda' m'}^{\alpha} 
 *          \langle Y_{\ell m} | R_{\ell_3 m_3} | Y_{\ell' m'} \rangle
 *  \f]
 *  This is our final answer: radial components of density and magnetization are expressed as a linear combination of
 *  quadratic forms in radial functions. 
 *
 *  \note density and potential are allocated as global function because it's easier to load and save them. */
class Density
{
    private:

        Simulation_context& ctx_;
        
        Unit_cell& unit_cell_;

        /// Density matrix of the system.
        /** In case of full-potential, matrix is stored for local fraction of atoms.
         *  In case of pseudo-potential, full matrix for all atoms is stored. */
        mdarray<double_complex, 4> density_matrix_;

        /// ae and ps local densities used for PAW
        std::vector< mdarray<double, 2> > paw_ae_local_density_; //vector iterates atoms
        std::vector< mdarray<double, 2> > paw_ps_local_density_;

        std::vector< mdarray<double, 3> > paw_ae_local_magnetization_; //vector iterates atoms
        std::vector< mdarray<double, 3> > paw_ps_local_magnetization_;

        /// Pointer to charge density.
        /** In the case of full-potential calculation this is the full (valence + core) electron charge density.
         *  In the case of pseudopotential this is the valence charge density. */ 
        Periodic_function<double>* rho_;

        /// Pointer to pseudo core charge density
        /** In the case of pseudopotential we need to know the non-linear core correction to the 
         *  exchange-correlation energy which is introduced trough the pseudo core density: 
         *  \f$ E_{xc}[\rho_{val} + \rho_{core}] \f$. The 'pseudo' reflects the fact that 
         *  this density integrated does not reproduce the total number of core elctrons. */
        Periodic_function<double>* rho_pseudo_core_;
        
        Periodic_function<double>* magnetization_[3];
        
        /// Non-zero Gaunt coefficients.
        std::unique_ptr< Gaunt_coefficients<double_complex> > gaunt_coefs_;
        
        /// fast mapping between composite lm index and corresponding orbital quantum number
        std::vector<int> l_by_lm_;

        Mixer<double_complex>* high_freq_mixer_;
        Mixer<double_complex>* low_freq_mixer_;
        Mixer<double>* mixer_;

        std::vector<int> lf_gvec_;
        std::vector<int> hf_gvec_;

        /// Symmetrize density matrix.
        /** Initially, density matrix is obtained with summation over irreducible BZ:
         *  \f[
         *      \tilde n_{\ell \lambda m \sigma, \ell' \lambda' m' \sigma'}^{\alpha}  = 
         *          \sum_{j} \sum_{{\bf k}}^{IBZ} \langle Y_{\ell m} u_{\ell \lambda}^{\alpha}| \Psi_{j{\bf k}}^{\sigma} \rangle w_{\bf k} n_{j{\bf k}}
         *          \langle \Psi_{j{\bf k}}^{\sigma'} | u_{\ell' \lambda'}^{\alpha} Y_{\ell' m'} \rangle 
         *  \f]
         *  In order to symmetrize it, the following operation is performed:
         *  \f[
         *      n_{\ell \lambda m \sigma, \ell' \lambda' m' \sigma'}^{\alpha} = \sum_{{\bf P}} 
         *          \sum_{j} \sum_{\bf k}^{IBZ} \langle Y_{\ell m} u_{\ell \lambda}^{\alpha}| \Psi_{j{\bf P}{\bf k}}^{\sigma} \rangle w_{\bf k} n_{j{\bf k}}
         *          \langle \Psi_{j{\bf P}{\bf k}}^{\sigma'} | u_{\ell' \lambda'}^{\alpha} Y_{\ell' m'} \rangle 
         *  \f]
         *  where \f$ {\bf P} \f$ is the space-group symmetry operation. The inner product between wave-function and
         *  local orbital is transformed as:
         *  \f[
         *      \langle \Psi_{j{\bf P}{\bf k}}^{\sigma} | u_{\ell \lambda}^{\alpha} Y_{\ell m} \rangle =
         *          \int \Psi_{j{\bf P}{\bf k}}^{\sigma *}({\bf r}) u_{\ell \lambda}^{\alpha}(r) Y_{\ell m}(\hat {\bf r}) dr =
         *          \int \Psi_{j{\bf k}}^{\sigma *}({\bf P}^{-1}{\bf r}) u_{\ell \lambda}^{\alpha}(r) Y_{\ell m}(\hat {\bf r}) dr =
         *          \int \Psi_{j{\bf k}}^{\sigma *}({\bf r}) u_{\ell \lambda}^{{\bf P}\alpha}(r) Y_{\ell m}({\bf P} \hat{\bf r}) dr
         *  \f]
         *  Under rotation the spherical harmonic is transformed as:
         *  \f[
         *        Y_{\ell m}({\bf P} \hat{\bf r}) = {\bf P}^{-1}Y_{\ell m}(\hat {\bf r}) = \sum_{m'} D_{m'm}^{\ell}({\bf P}^{-1}) Y_{\ell m'}(\hat {\bf r}) = 
         *          \sum_{m'} D_{mm'}^{\ell}({\bf P}) Y_{\ell m'}(\hat {\bf r})
         *  \f]
         *  The inner-product integral is then rewritten as:
         *  \f[
         *      \langle \Psi_{j{\bf P}{\bf k}}^{\sigma} | u_{\ell \lambda}^{\alpha} Y_{\ell m} \rangle  = 
         *          \sum_{m'} D_{mm'}^{\ell}({\bf P}) \langle \Psi_{j{\bf k}}^{\sigma} | u_{\ell \lambda}^{{\bf P}\alpha} Y_{\ell m} \rangle 
         *  \f]
         *  and the final expression for density matrix gets the following form:
         *  \f[
         *      n_{\ell \lambda m \sigma, \ell' \lambda' m' \sigma'}^{\alpha} = \sum_{{\bf P}}
         *          \sum_{j} \sum_{\bf k}^{IBZ} \sum_{m_1 m_2} D_{mm_1}^{\ell *}({\bf P}) D_{m'm_2}^{\ell'}({\bf P})  
         *          \langle Y_{\ell m_1} u_{\ell \lambda}^{{\bf P} \alpha}| 
         *          \Psi_{j{\bf k}}^{\sigma} \rangle w_{\bf k} n_{j{\bf k}} \langle \Psi_{j{\bf k}}^{\sigma'} | 
         *          u_{\ell' \lambda'}^{{\bf P}\alpha} Y_{\ell' m_2} \rangle = \sum_{{\bf P}}
         *          \sum_{m_1 m_2} D_{mm_1}^{\ell *}({\bf P}) D_{m'm_2}^{\ell'}({\bf P}) 
         *          \tilde n_{\ell \lambda m_1 \sigma, \ell' \lambda' m_2 \sigma'}^{{\bf P}\alpha} 
         *  \f]
         */
        void symmetrize_density_matrix();

        /// Reduce complex density matrix over magnetic quantum numbers
        /** The following operation is performed:
         *  \f[
         *      d_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha} = 
         *          \sum_{mm'} D_{\ell \lambda m, \ell' \lambda' m'}^{\alpha} 
         *          \langle Y_{\ell m} | R_{\ell_3 m_3} | Y_{\ell' m'} \rangle
         *  \f] 
         */
        template <int num_mag_dims, typename T>
        void reduce_density_matrix(Atom_type const& atom_type__,
                                   int ia__,
                                   mdarray<double_complex, 4> const& zdens__,
                                   Gaunt_coefficients<T> const& gaunt_coeffs__,
                                   mdarray<double, 3>& mt_density_matrix__)
        {
            mt_density_matrix__.zero();
            
            #pragma omp parallel for default(shared)
            for (int idxrf2 = 0; idxrf2 < atom_type__.mt_radial_basis_size(); idxrf2++) {
                int l2 = atom_type__.indexr(idxrf2).l;
                for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++) {
                    int offs = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                    int l1 = atom_type__.indexr(idxrf1).l;

                    int xi2 = atom_type__.indexb().index_by_idxrf(idxrf2);
                    for (int lm2 = Utils::lm_by_l_m(l2, -l2); lm2 <= Utils::lm_by_l_m(l2, l2); lm2++, xi2++) {
                        int xi1 = atom_type__.indexb().index_by_idxrf(idxrf1);
                        for (int lm1 = Utils::lm_by_l_m(l1, -l1); lm1 <= Utils::lm_by_l_m(l1, l1); lm1++, xi1++) {
                            for (int k = 0; k < gaunt_coeffs__.num_gaunt(lm1, lm2); k++) {
                                int lm3 = gaunt_coeffs__.gaunt(lm1, lm2, k).lm3;
                                T gc = gaunt_coeffs__.gaunt(lm1, lm2, k).coef;
                                switch (num_mag_dims) {
                                    case 3: {
                                        mt_density_matrix__(lm3, offs, 2) += 2.0 * std::real(zdens__(xi1, xi2, 2, ia__) * gc); 
                                        mt_density_matrix__(lm3, offs, 3) -= 2.0 * std::imag(zdens__(xi1, xi2, 2, ia__) * gc);
                                    }
                                    case 1: {
                                        mt_density_matrix__(lm3, offs, 1) += std::real(zdens__(xi1, xi2, 1, ia__) * gc);
                                    }
                                    case 0: {
                                        mt_density_matrix__(lm3, offs, 0) += std::real(zdens__(xi1, xi2, 0, ia__) * gc);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Add k-point contribution to the auxiliary density matrix.
        /** In case of full-potential LAPW complex density matrix has the following expression:
         *  \f[
         *      d_{\xi \sigma, \xi' \sigma'}^{\alpha} = \sum_{j{\bf k}} n_{j{\bf k}}
         *          S_{\xi}^{\sigma j {\bf k},\alpha*} S_{\xi'}^{\sigma' j {\bf k},\alpha}
         *  \f]
         * 
         *  where \f$ S_{\xi}^{\sigma j {\bf k},\alpha} \f$ are the expansion coefficients of
         *  spinor wave functions inside muffin-tin spheres.
         *  
         *  In case of LDA+U the occupation matrix is also computed. It has the following expression:
         *  \f[
         *      n_{\ell,mm'}^{\sigma \sigma'} = \sum_{i {\bf k}}^{occ} \int_{0}^{R_{MT}} r^2 dr 
         *                \Psi_{\ell m}^{i{\bf k}\sigma *}({\bf r}) \Psi_{\ell m'}^{i{\bf k}\sigma'}({\bf r})
         *  \f] 
         *
         * In case of ultrasoft pseudopotential the following density matrix has to be computed for each atom:
         *  \f[
         *      d_{\xi \xi'}^{\alpha} = \langle \beta_{\xi}^{\alpha} | \hat N | \beta_{\xi'}^{\alpha} \rangle = 
         *        \sum_{j {\bf k}} \langle \beta_{\xi}^{\alpha} | \Psi_{j{\bf k}} \rangle n_{j{\bf k}} 
         *        \langle \Psi_{j{\bf k}} | \beta_{\xi'}^{\alpha} \rangle
         *  \f]
         *  Here \f$ \hat N = \sum_{j{\bf k}} | \Psi_{j{\bf k}} \rangle n_{j{\bf k}} \langle \Psi_{j{\bf k}} | \f$ is 
         *  the occupancy operator written in spectral representation. */
        template <typename T> 
        inline void add_k_point_contribution_dm(K_point* kp__,
                                                mdarray<double_complex, 4>& density_matrix__);

        /// Add k-point contribution to the density and magnetization defined on the regular FFT grid.
        template <bool mt_spheres>
        inline void add_k_point_contribution_rg(K_point* kp__);

        /// Generate valence density in the muffin-tins 
        void generate_valence_density_mt(K_set& ks);
        
        /// Generate charge density of core states
        void generate_core_charge_density()
        {
            PROFILE_WITH_TIMER("sirius::Density::generate_core_charge_density");

            for (int icloc = 0; icloc < unit_cell_.spl_num_atom_symmetry_classes().local_size(); icloc++) {
                int ic = unit_cell_.spl_num_atom_symmetry_classes(icloc);
                unit_cell_.atom_symmetry_class(ic).generate_core_charge_density(ctx_.core_relativity());
            }

            for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
                int rank = unit_cell_.spl_num_atom_symmetry_classes().local_rank(ic);
                unit_cell_.atom_symmetry_class(ic).sync_core_charge_density(ctx_.comm(), rank);
            }
        }

        void generate_pseudo_core_charge_density()
        {
            PROFILE_WITH_TIMER("sirius::Density::generate_pseudo_core_charge_density");

            auto rho_core_radial_integrals = generate_rho_radial_integrals(2);

            std::vector<double_complex> v = unit_cell_.make_periodic_function(rho_core_radial_integrals, ctx_.gvec());
            ctx_.fft().prepare(ctx_.gvec().partition());
            ctx_.fft().transform<1>(ctx_.gvec().partition(), &v[ctx_.gvec().partition().gvec_offset_fft()]);
            ctx_.fft().output(&rho_pseudo_core_->f_rg(0));
            ctx_.fft().dismiss();
        }

        /// Initialize \rho_{ij} - density matrix, occupation on basis of beta-projectors (used for PAW).
        void initialize_beta_density_matrix();

    public:

        /// Constructor
        Density(Simulation_context& ctx_);
        
        /// Destructor
        ~Density();
       
        /// Set pointers to muffin-tin and interstitial charge density arrays
        void set_charge_density_ptr(double* rhomt, double* rhoir)
        {
            if (ctx_.full_potential()) {
                rho_->set_mt_ptr(rhomt);
            }
            rho_->set_rg_ptr(rhoir);
        }
        
        /// Set pointers to muffin-tin and interstitial magnetization arrays
        void set_magnetization_ptr(double* magmt, double* magir)
        {
            if (ctx_.num_mag_dims() == 0) {
                return;
            }
            assert(ctx_.num_spins() == 2);

            // set temporary array wrapper
            mdarray<double, 4> magmt_tmp(magmt, ctx_.lmmax_rho(), unit_cell_.max_num_mt_points(), 
                                         unit_cell_.num_atoms(), ctx_.num_mag_dims());
            mdarray<double, 2> magir_tmp(magir, ctx_.fft().size(), ctx_.num_mag_dims());
            
            if (ctx_.num_mag_dims() == 1) {
                /* z component is the first and only one */
                magnetization_[0]->set_mt_ptr(&magmt_tmp(0, 0, 0, 0));
                magnetization_[0]->set_rg_ptr(&magir_tmp(0, 0));
            }

            if (ctx_.num_mag_dims() == 3) {
                /* z component is the first */
                magnetization_[0]->set_mt_ptr(&magmt_tmp(0, 0, 0, 2));
                magnetization_[0]->set_rg_ptr(&magir_tmp(0, 2));
                /* x component is the second */
                magnetization_[1]->set_mt_ptr(&magmt_tmp(0, 0, 0, 0));
                magnetization_[1]->set_rg_ptr(&magir_tmp(0, 0));
                /* y component is the third */
                magnetization_[2]->set_mt_ptr(&magmt_tmp(0, 0, 0, 1));
                magnetization_[2]->set_rg_ptr(&magir_tmp(0, 1));
            }
        }
        
        /// Zero density and magnetization
        void zero()
        {
            rho_->zero();
            for (int i = 0; i < ctx_.num_mag_dims(); i++) {
                magnetization_[i]->zero();
            }
        }
        
        /// Find the total leakage of the core states out of the muffin-tins
        double core_leakage()
        {
            double sum = 0.0;
            for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
                sum += core_leakage(ic) * unit_cell_.atom_symmetry_class(ic).num_atoms();
            }
            return sum;
        }

        /// Return core leakage for a specific atom symmetry class
        double core_leakage(int ic)
        {
            return unit_cell_.atom_symmetry_class(ic).core_leakage();
        }

        /// Generate initial charge density and magnetization
        void initial_density();

        void initial_density_pseudo();

        void initial_density_full_pot();

        /// Generate full charge density (valence + core) and magnetization from the wave functions.
        inline void generate(K_set& ks__);

        /// Generate valence charge density and magnetization from the wave functions.
        inline void generate_valence(K_set& ks__);
        
        /// Add augmentation charge Q(r)
        /** Restore valence density by adding the Q-operator constribution.
         *  The following term is added to the valence density, generated by the pseudo wave-functions:
         *  \f[
         *      \tilde \rho({\bf G}) = \sum_{\alpha} \sum_{\xi \xi'} d_{\xi \xi'}^{\alpha} Q_{\xi' \xi}^{\alpha}({\bf G})
         *  \f]
         *  Plane-wave coefficients of the Q-operator for a given atom \f$ \alpha \f$ can be obtained from the 
         *  corresponding coefficients of the Q-operator for a given atom \a type A:
         *  \f[
         *       Q_{\xi' \xi}^{\alpha(A)}({\bf G}) = e^{-i{\bf G}\tau_{\alpha(A)}} Q_{\xi' \xi}^{A}({\bf G})
         *  \f]
         *  We use this property to split the sum over atoms into sum over atom types and inner sum over atoms of the 
         *  same type:
         *  \f[
         *       \tilde \rho({\bf G}) = \sum_{A} \sum_{\xi \xi'} Q_{\xi' \xi}^{A}({\bf G}) \sum_{\alpha(A)} 
         *          d_{\xi \xi'}^{\alpha(A)} e^{-i{\bf G}\tau_{\alpha(A)}} = 
         *          \sum_{A} \sum_{\xi \xi'} Q_{\xi' \xi}^{A}({\bf G}) d_{\xi \xi'}^{A}({\bf G})
         *  \f]
         *  where
         *  \f[
         *      d_{\xi \xi'}^{A}({\bf G}) = \sum_{\alpha(A)} d_{\xi \xi'}^{\alpha(A)} e^{-i{\bf G}\tau_{\alpha(A)}} 
         *  \f]
         */
        void augment(K_set& ks__) // TODO: skip when norm-conserving potential is used for all species
        {
            PROFILE_WITH_TIMER("sirius::Density::augment");

            /* split G-vectors between ranks */
            splindex<block> spl_gvec(ctx_.gvec().num_gvec(), ctx_.comm().size(), ctx_.comm().rank());
            
            /* collect density and magnetization into single array */
            std::vector<Periodic_function<double>*> rho_vec(ctx_.num_mag_dims() + 1);
            rho_vec[0] = rho_;
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                rho_vec[1 + j] = magnetization_[j];
            }

            #ifdef __PRINT_OBJECT_CHECKSUM
            for (auto e: rho_vec) {
                auto cs = e->checksum_pw();
                DUMP("checksum(rho_vec_pw): %20.14f %20.14f", cs.real(), cs.imag());
            }
            #endif

            mdarray<double_complex, 2> rho_aug(spl_gvec.local_size(), ctx_.num_mag_dims() + 1);

            #ifdef __GPU
            if (ctx_.processing_unit() == GPU) {
                rho_aug.allocate(memory_t::device);
            }
            #endif
            
            switch (ctx_.processing_unit()) {
                case CPU: {
                    generate_rho_aug<CPU>(rho_vec, rho_aug);
                    break;
                }
                case GPU: {
                    generate_rho_aug<GPU>(rho_vec, rho_aug);
                    break;
                }
            }

            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                #pragma omp parallel for
                for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++) {
                    rho_vec[iv]->f_pw(spl_gvec[igloc]) += rho_aug(igloc, iv);
                }
            }

            runtime::Timer t5("sirius::Density::augment|mpi");
            for (auto e: rho_vec) {
                ctx_.comm().allgather(&e->f_pw(0), spl_gvec.global_offset(), spl_gvec.local_size());

                #ifdef __PRINT_OBJECT_CHECKSUM
                {
                    auto cs = e->checksum_pw();
                    DUMP("checksum(rho_vec_pw): %20.14f %20.14f", cs.real(), cs.imag());
                }
                #endif
            }
            t5.stop();
        }

        template <device_t pu>
        inline void generate_rho_aug(std::vector<Periodic_function<double>*> rho__,
                              mdarray<double_complex, 2>& rho_aug__);
        
        /// generate n_1 and \tilda{n}_1 in lm components
        void generate_paw_loc_density();

        /// Check density at MT boundary
        void check_density_continuity_at_mt();

        mdarray<double, 2> generate_rho_radial_integrals(int type__);

        void generate_pw_coefs()
        {
            rho_->fft_transform(-1);
        }
         
        void save()
        {
            if (ctx_.comm().rank() == 0)
            {
                HDF5_tree fout(storage_file_name, false);
                rho_->hdf5_write(fout["density"]);
                for (int j = 0; j < ctx_.num_mag_dims(); j++)
                    magnetization_[j]->hdf5_write(fout["magnetization"].create_node(j));
            }
            ctx_.comm().barrier();
        }
        
        void load()
        {
            HDF5_tree fout(storage_file_name, false);
            rho_->hdf5_read(fout["density"]);
            for (int j = 0; j < ctx_.num_mag_dims(); j++)
                magnetization_[j]->hdf5_read(fout["magnetization"][j]);
        }

        inline size_t size()
        {
            size_t s = rho_->size();
            for (int i = 0; i < ctx_.num_mag_dims(); i++) s += magnetization_[i]->size();
            return s;
        }

        Periodic_function<double>* rho()
        {
            return rho_;
        }
        
        Periodic_function<double>* rho_pseudo_core()
        {
            return rho_pseudo_core_;
        }
        
        Periodic_function<double>** magnetization()
        {
            return magnetization_;
        }

        Periodic_function<double>* magnetization(int i)
        {
            return magnetization_[i];
        }

        Spheric_function<spectral, double> const& density_mt(int ialoc) const
        {
            return rho_->f_mt(ialoc);
        }

        std::vector< mdarray<double, 2> >* get_paw_ae_local_density()
        {
            return &paw_ae_local_density_;
        }

        std::vector< mdarray<double, 2> >* get_paw_ps_local_density()
        {
            return &paw_ps_local_density_;
        }

        std::vector< mdarray<double, 3> >* get_paw_ae_local_magnetization()
        {
            return &paw_ae_local_magnetization_;
        }

        std::vector< mdarray<double, 3> >* get_paw_ps_local_magnetization()
        {
            return &paw_ps_local_magnetization_;
        }

        void allocate()
        {
            rho_->allocate_mt(true);
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                magnetization_[j]->allocate_mt(true);
            }
        }

        void mixer_input()
        {
            if (mixer_ != nullptr) {
                size_t n = rho_->pack(0, mixer_);
                for (int i = 0; i < ctx_.num_mag_dims(); i++) {
                    n += magnetization_[i]->pack(n, mixer_);
                }
            } else {
                int k = 0;
                for (int ig: lf_gvec_) {
                    low_freq_mixer_->input(k++, rho_->f_pw(ig));
                }
                for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                    for (int ig: lf_gvec_) {
                        low_freq_mixer_->input(k++, magnetization_[j]->f_pw(ig));
                    }
                }
                for (size_t i = 0; i < density_matrix_.size(); i++) {
                     low_freq_mixer_->input(k++, density_matrix_[i]);
                }

                k = 0;
                for (int ig: hf_gvec_) {
                    high_freq_mixer_->input(k++, rho_->f_pw(ig));
                }
                for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                    for (int ig: hf_gvec_) {
                        high_freq_mixer_->input(k++, magnetization_[j]->f_pw(ig));
                    }
                }
            }
        }

        void mixer_output()
        {
            if (mixer_ != nullptr) {
                size_t n = rho_->unpack(mixer_->output_buffer());
                for (int i = 0; i < ctx_.num_mag_dims(); i++) {
                    n += magnetization_[i]->unpack(&mixer_->output_buffer()[n]);
                }
            } else {
                int k = 0;
                for (int ig: lf_gvec_) {
                    rho_->f_pw(ig) = low_freq_mixer_->output_buffer(k++);
                }
                for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                    for (int ig: lf_gvec_) {
                        magnetization_[j]->f_pw(ig) = low_freq_mixer_->output_buffer(k++);
                    }
                }
                for (size_t i = 0; i < density_matrix_.size(); i++) {
                    density_matrix_[i] = low_freq_mixer_->output_buffer(k++);
                }

                k = 0;
                for (int ig: hf_gvec_) {
                    rho_->f_pw(ig) = high_freq_mixer_->output_buffer(k++);
                }
                for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                    for (int ig: hf_gvec_) {
                        magnetization_[j]->f_pw(ig) = high_freq_mixer_->output_buffer(k++);
                    }
                }
            }
        }

        void mixer_init()
        {
            mixer_input();

            if (mixer_ != nullptr) {
                mixer_->initialize();
            } else {
                low_freq_mixer_->initialize();
                high_freq_mixer_->initialize();
            }
        }

        double mix()
        {
            double rms;

            if (mixer_ != nullptr) {
                /* mix in real-space in case of FP-LAPW */
                mixer_input();
                rms = mixer_->mix();
                mixer_output();
                /* get rho(G) after mixing */
                rho_->fft_transform(-1);
            } else {
                /* mix in G-space in case of PP */
                mixer_input();
                rms = low_freq_mixer_->mix();
                rms += high_freq_mixer_->mix();
                mixer_output();
                ctx_.fft().prepare(ctx_.gvec().partition());
                rho_->fft_transform(1);
                for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                    magnetization_[j]->fft_transform(1);
                }
                ctx_.fft().dismiss();
            }

            return rms;
        }

        inline double dr2()
        {
            return low_freq_mixer_->rss();
        }

        mdarray<double_complex, 4> const& density_matrix() const
        {
            return density_matrix_;
        }
};

template <typename T> 
inline void Density::add_k_point_contribution_dm(K_point* kp__,
                                                 mdarray<double_complex, 4>& density_matrix__)
{
    PROFILE_WITH_TIMER("sirius::Density::add_k_point_contribution_dm");
    
    if (ctx_.esm_type() == electronic_structure_method_t::full_potential_lapwlo) {
        /* non-magnetic or spin-collinear case */
        if (ctx_.num_mag_dims() != 3) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                int nbnd = kp__->spinor_wave_functions<true>(ispn).spl_num_swapped().local_size();

                mdarray<double_complex, 2> wf1(unit_cell_.max_mt_basis_size(), nbnd);
                mdarray<double_complex, 2> wf2(unit_cell_.max_mt_basis_size(), nbnd);

                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    int offset_wf = unit_cell_.atom(ia).offset_wf();
                    int mt_basis_size = unit_cell_.atom(ia).type().mt_basis_size();

                    for (int i = 0; i < nbnd; i++) {
                        int j = kp__->spinor_wave_functions<true>(ispn).spl_num_swapped()[i];
 
                        for (int xi = 0; xi < mt_basis_size; xi++) {
                            wf1(xi, i) = std::conj(kp__->spinor_wave_functions<true>(ispn)[i][offset_wf + xi]);
                            wf2(xi, i) = kp__->spinor_wave_functions<true>(ispn)[i][offset_wf + xi] * 
                                         kp__->band_occupancy(j + ispn * ctx_.num_fv_states()) *
                                         kp__->weight();
                        }
                    }
                    /* add |psi_j> n_j <psi_j| to density matrix */
                    linalg<CPU>::gemm(0, 1, mt_basis_size, mt_basis_size, nbnd, complex_one, 
                                          &wf1(0, 0), wf1.ld(), &wf2(0, 0), wf2.ld(), complex_one, 
                                          density_matrix__.at<CPU>(0, 0, ispn, ia), density_matrix__.ld());
                }
            }
        } else {
            assert(kp__->spinor_wave_functions<true>(0).spl_num_swapped().local_size() ==
                   kp__->spinor_wave_functions<true>(1).spl_num_swapped().local_size());

            int nbnd = kp__->spinor_wave_functions<true>(0).spl_num_swapped().local_size();

            mdarray<double_complex, 3> wf1(unit_cell_.max_mt_basis_size(), nbnd, ctx_.num_spins());
            mdarray<double_complex, 3> wf2(unit_cell_.max_mt_basis_size(), nbnd, ctx_.num_spins());

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                int offset_wf = unit_cell_.atom(ia).offset_wf();
                int mt_basis_size = unit_cell_.atom(ia).type().mt_basis_size();

                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    for (int i = 0; i < nbnd; i++) {
                        int j = kp__->spinor_wave_functions<true>(ispn).spl_num_swapped()[i];

                        for (int xi = 0; xi < mt_basis_size; xi++) {
                            wf1(xi, i, ispn) = std::conj(kp__->spinor_wave_functions<true>(ispn)[i][offset_wf + xi]);
                            wf2(xi, i, ispn) = kp__->spinor_wave_functions<true>(ispn)[i][offset_wf + xi] * 
                                               kp__->band_occupancy(j) * kp__->weight();
                        }
                    }
                }
                /* compute diagonal terms */
                for (int ispn = 0; ispn < 2; ispn++) {
                    linalg<CPU>::gemm(0, 1, mt_basis_size, mt_basis_size, nbnd, complex_one, 
                                      &wf1(0, 0, ispn), wf1.ld(), &wf2(0, 0, ispn), wf2.ld(), complex_one, 
                                      density_matrix__.at<CPU>(0, 0, ispn, ia), density_matrix__.ld());
                }
                /* offdiagonal term */
                linalg<CPU>::gemm(0, 1, mt_basis_size, mt_basis_size, nbnd, complex_one, 
                                  &wf1(0, 0, 1), wf1.ld(), &wf2(0, 0, 0), wf2.ld(), complex_one, 
                                  density_matrix__.at<CPU>(0, 0, 2, ia), density_matrix__.ld());
            }
        }
    }

    if (ctx_.esm_type() == electronic_structure_method_t::ultrasoft_pseudopotential ||
        ctx_.esm_type() == electronic_structure_method_t::norm_conserving_pseudopotential ||
        ctx_.esm_type() == electronic_structure_method_t::paw_pseudopotential)
    {
        if (!ctx_.unit_cell().mt_lo_basis_size()) {
            return;
        }

        kp__->beta_projectors().prepare();

        #ifdef __GPU
        bool allocate_on_gpu[] = {false, false};
        if (ctx_.processing_unit() == GPU) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                int nbnd = kp__->num_occupied_bands(ispn);
                if (!kp__->spinor_wave_functions<false>(ispn).coeffs().on_device()) {
                    allocate_on_gpu[ispn] = true;
                    kp__->spinor_wave_functions<false>(ispn).allocate_on_device();
                    kp__->spinor_wave_functions<false>(ispn).copy_to_device(0, nbnd);
                }
            }
        }
        #endif

        if (ctx_.num_mag_dims() != 3) {
            for (int chunk = 0; chunk < kp__->beta_projectors().num_beta_chunks(); chunk++) {
                kp__->beta_projectors().generate(chunk);

                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    /* total number of occupied bands for this spin */
                    int nbnd = kp__->num_occupied_bands(ispn);
                    /* compute <beta|psi> */
                    kp__->beta_projectors().inner<T>(chunk, kp__->spinor_wave_functions<false>(ispn), 0, nbnd);

                    /* number of beta projectors */
                    int nbeta = kp__->beta_projectors().beta_chunk(chunk).num_beta_;

                    auto beta_psi = kp__->beta_projectors().beta_phi<T>(chunk, nbnd);

                    splindex<block> spl_nbnd(nbnd, kp__->comm().size(), kp__->comm().rank());

                    int nbnd_loc = spl_nbnd.local_size();
                    if (nbnd_loc) { // TODO: this part can also be moved to GPU
                        #pragma omp parallel
                        {
                            /* auxiliary arrays */
                            mdarray<double_complex, 2> bp1(nbeta, nbnd_loc);
                            mdarray<double_complex, 2> bp2(nbeta, nbnd_loc);
                            #pragma omp for
                            for (int ia = 0; ia < kp__->beta_projectors().beta_chunk(chunk).num_atoms_; ia++) {
                                int nbf = kp__->beta_projectors().beta_chunk(chunk).desc_(0, ia);
                                int offs = kp__->beta_projectors().beta_chunk(chunk).desc_(1, ia);
                                int ja = kp__->beta_projectors().beta_chunk(chunk).desc_(3, ia);

                                for (int i = 0; i < nbnd_loc; i++) {
                                    int j = spl_nbnd[i];

                                    for (int xi = 0; xi < nbf; xi++) {
                                        bp1(xi, i) = beta_psi(offs + xi, j);
                                        bp2(xi, i) = std::conj(bp1(xi, i)) * kp__->weight() *
                                                     kp__->band_occupancy(j + ispn * ctx_.num_fv_states());
                                    }
                                }

                                linalg<CPU>::gemm(0, 1, nbf, nbf, nbnd_loc, complex_one, &bp1(0, 0), bp1.ld(),
                                                  &bp2(0, 0), bp2.ld(), complex_one, &density_matrix__(0, 0, ispn, ja), 
                                                  density_matrix__.ld());
                            }
                        }
                    }
                }
            }
        } else {
            STOP();
        }

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                if (allocate_on_gpu[ispn]) {
                    kp__->spinor_wave_functions<false>(ispn).deallocate_on_device();
                }
            }
        }
        #endif

        kp__->beta_projectors().dismiss();
    }
}

template <device_t pu>
inline void Density::generate_rho_aug(std::vector<Periodic_function<double>*> rho__,
                                      mdarray<double_complex, 2>& rho_aug__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_rho_aug");

    splindex<block> spl_gvec(ctx_.gvec().num_gvec(), ctx_.comm().size(), ctx_.comm().rank());

    if (pu == CPU) {
        rho_aug__.zero();
    }

    #ifdef __GPU
    if (pu == GPU) {
        rho_aug__.zero_on_device();
    }
    #endif
    
    ctx_.augmentation_op(0).prepare(0);

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        if (!atom_type.uspp().augmentation_) {
            continue;
        }

        int nbf = atom_type.mt_basis_size();
        
        /* convert to real matrix */
        mdarray<double, 3> dm(nbf * (nbf + 1) / 2, atom_type.num_atoms(), ctx_.num_mag_dims() + 1);
        #pragma omp parallel for
        for (int i = 0; i < atom_type.num_atoms(); i++) {
            int ia = atom_type.atom_id(i);

            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 <= xi2; xi1++) {
                    int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                    switch (ctx_.num_mag_dims()) {
                        case 0: {
                            dm(idx12, i, 0) = density_matrix_(xi2, xi1, 0, ia).real();
                            break;
                        }
                        case 1: {
                            dm(idx12, i, 0) = std::real(density_matrix_(xi2, xi1, 0, ia) + density_matrix_(xi2, xi1, 1, ia));
                            dm(idx12, i, 1) = std::real(density_matrix_(xi2, xi1, 0, ia) - density_matrix_(xi2, xi1, 1, ia));
                            break;
                        }
                    }
                }
            }
        }

        if (pu == CPU) {
            runtime::Timer t2("sirius::Density::generate_rho_aug|phase_fac");
            /* treat phase factors as real array with x2 size */
            mdarray<double, 2> phase_factors(atom_type.num_atoms(), spl_gvec.local_size() * 2);

            #pragma omp parallel for
            for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++) {
                int ig = spl_gvec[igloc];
                for (int i = 0; i < atom_type.num_atoms(); i++) {
                    int ia = atom_type.atom_id(i);
                    double_complex z = std::conj(ctx_.gvec_phase_factor(ig, ia));
                    phase_factors(i, 2 * igloc)     = z.real();
                    phase_factors(i, 2 * igloc + 1) = z.imag();
                }
            }
            t2.stop();
            
            /* treat auxiliary array as double with x2 size */
            mdarray<double, 2> dm_pw(nbf * (nbf + 1) / 2, spl_gvec.local_size() * 2);

            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                runtime::Timer t3("sirius::Density::generate_rho_aug|gemm");
                linalg<CPU>::gemm(0, 0, nbf * (nbf + 1) / 2, spl_gvec.local_size() * 2, atom_type.num_atoms(), 
                                  &dm(0, 0, iv), dm.ld(),
                                  &phase_factors(0, 0), phase_factors.ld(), 
                                  &dm_pw(0, 0), dm_pw.ld());
                t3.stop();

                #ifdef __PRINT_OBJECT_CHECKSUM
                {
                    auto cs = dm_pw.checksum();
                    ctx_.comm().allreduce(&cs, 1);
                    DUMP("checksum(dm_pw) : %18.10f", cs);
                }
                #endif

                runtime::Timer t4("sirius::Density::generate_rho_aug|sum");
                #pragma omp parallel for
                for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++) {
                    double_complex zsum(0, 0);
                    /* get contribution from non-diagonal terms */
                    for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                        double_complex z1 = double_complex(ctx_.augmentation_op(iat).q_pw(i, 2 * igloc),
                                                           ctx_.augmentation_op(iat).q_pw(i, 2 * igloc + 1));
                        double_complex z2(dm_pw(i, 2 * igloc), dm_pw(i, 2 * igloc + 1));

                        zsum += z1 * z2 * ctx_.augmentation_op(iat).sym_weight(i);
                    }
                    rho_aug__(igloc, iv) += zsum;
                }
                t4.stop();
            }
        }

        #ifdef __GPU
        if (pu == GPU) {
            dm.allocate(memory_t::device);
            dm.copy_to_device();

            /* treat auxiliary array as double with x2 size */
            mdarray<double, 2> dm_pw(nullptr, nbf * (nbf + 1) / 2, spl_gvec.local_size() * 2);
            dm_pw.allocate(memory_t::device);

            mdarray<double, 1> phase_factors(nullptr, atom_type.num_atoms() * spl_gvec.local_size() * 2);
            phase_factors.allocate(memory_t::device);

            acc::sync_stream(0);
            if (iat + 1 != unit_cell_.num_atom_types()) {
                ctx_.augmentation_op(iat + 1).prepare(0);
            }

            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                generate_dm_pw_gpu(atom_type.num_atoms(),
                                   spl_gvec.local_size(),
                                   nbf,
                                   ctx_.atom_coord(iat).at<GPU>(),
                                   ctx_.gvec_coord().at<GPU>(),
                                   phase_factors.at<GPU>(),
                                   dm.at<GPU>(0, 0, iv),
                                   dm_pw.at<GPU>(),
                                   1);
                sum_q_pw_dm_pw_gpu(spl_gvec.local_size(), 
                                   nbf,
                                   ctx_.augmentation_op(iat).q_pw().at<GPU>(),
                                   dm_pw.at<GPU>(),
                                   ctx_.augmentation_op(iat).sym_weight().at<GPU>(),
                                   rho_aug__.at<GPU>(0, iv),
                                   1);
            }
            acc::sync_stream(1);
            ctx_.augmentation_op(iat).dismiss();
        }
        #endif
    }

    #ifdef __GPU
    if (pu == GPU) {
        rho_aug__.copy_to_host();
    }
    #endif
    
    #ifdef __PRINT_OBJECT_CHECKSUM
    {
         auto cs = rho_aug__.checksum();
         DUMP("checksum(rho_aug): %20.14f %20.14f", cs.real(), cs.imag());
    }
    #endif
}

inline void Density::generate_valence(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_valence");

    double wt{0};
    double occ_val{0};
    for (int ik = 0; ik < ks__.num_kpoints(); ik++) {
        wt += ks__[ik]->weight();
        for (int j = 0; j < ctx_.num_bands(); j++) {
            occ_val += ks__[ik]->weight() * ks__[ik]->band_occupancy(j);
        }
    }

    if (std::abs(wt - 1.0) > 1e-12) {
        TERMINATE("K_point weights don't sum to one");
    }

    if (std::abs(occ_val - unit_cell_.num_valence_electrons()) > 1e-8) {
        std::stringstream s;
        s << "wrong occupancies" << std::endl
          << "  computed : " << occ_val << std::endl
          << "  required : " << unit_cell_.num_valence_electrons() << std::endl
          << "  difference : " << std::abs(occ_val - unit_cell_.num_valence_electrons());
        WARNING(s);
    }
    
    density_matrix_.zero();

    /* zero density and magnetization */
    zero();
    
    /* start the main loop over k-points */
    for (int ikloc = 0; ikloc < ks__.spl_num_kpoints().local_size(); ikloc++) {
        int ik = ks__.spl_num_kpoints(ikloc);
        auto kp = ks__[ik];

        /* swap wave functions */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int nbnd = kp->num_occupied_bands(ispn);
            if (ctx_.full_potential()) {
                kp->spinor_wave_functions<true>(ispn).swap_forward(0, nbnd);
            } else {
                #ifdef __GPU
                if (ctx_.processing_unit() == GPU) {
                    kp->spinor_wave_functions<false>(ispn).allocate_on_device();
                    kp->spinor_wave_functions<false>(ispn).copy_to_device(0, nbnd);
                }
                #endif
                kp->spinor_wave_functions<false>(ispn).swap_forward(0, nbnd, kp->gkvec().partition(),
                                                                    ctx_.mpi_grid_fft().communicator(1 << 1));
            }
        }
        
        if (ctx_.esm_type() == electronic_structure_method_t::full_potential_lapwlo) {
            add_k_point_contribution_dm<double_complex>(kp, density_matrix_);
        }
        
        if (ctx_.esm_type() == electronic_structure_method_t::ultrasoft_pseudopotential ||
            ctx_.esm_type() == electronic_structure_method_t::paw_pseudopotential) {
            if (ctx_.gamma_point()) {
                add_k_point_contribution_dm<double>(kp, density_matrix_);
            } else {
                add_k_point_contribution_dm<double_complex>(kp, density_matrix_);
            }
        }

        /* add contribution from regular space grid */
        if (ctx_.full_potential()) {
            add_k_point_contribution_rg<true>(kp);
        } else {
            add_k_point_contribution_rg<false>(kp);
        }

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            #ifdef __GPU
            if (ctx_.processing_unit() == GPU) {
                kp->spinor_wave_functions<false>(ispn).deallocate_on_device();
            }
            #endif
        }
    }

    if (density_matrix_.size()) {
        ctx_.comm().allreduce(density_matrix_.at<CPU>(), static_cast<int>(density_matrix_.size()));
    }

    /* reduce arrays; assume that each rank did its own fraction of the density */
    auto& comm = (ctx_.fft().parallel()) ? ctx_.mpi_grid().communicator(1 << _mpi_dim_k_ | 1 << _mpi_dim_k_col_)
                                         : ctx_.comm();

    comm.allreduce(&rho_->f_rg(0), ctx_.fft().local_size()); 
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        comm.allreduce(&magnetization_[j]->f_rg(0), ctx_.fft().local_size()); 
    }

    /* for muffin-tin part */
    switch (ctx_.esm_type()) {
        case electronic_structure_method_t::full_potential_lapwlo: {
            generate_valence_density_mt(ks__);
            break;
        }
        case electronic_structure_method_t::full_potential_pwlo: {
            STOP();
        }
        default: {
            break;
        }
    }

    ctx_.fft().prepare(ctx_.gvec().partition());
    /* get rho(G) and mag(G)
     * they are required to symmetrize density and magnetization */
    rho_->fft_transform(-1);
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        magnetization_[j]->fft_transform(-1);
    }
    ctx_.fft().dismiss();

    //== printf("number of electrons: %f\n", rho_->f_pw(0).real() * unit_cell_.omega());
    //== STOP();

    if (!ctx_.full_potential()) {
        augment(ks__);
    }

    if (ctx_.esm_type() == electronic_structure_method_t::paw_pseudopotential) {
        symmetrize_density_matrix();
    }
}

inline void Density::generate(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate");

    generate_valence(ks__);

    if (ctx_.full_potential()) {
        /* find the core states */
        generate_core_charge_density();
        /* add core contribution */
        for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            for (int ir = 0; ir < unit_cell_.atom(ia).num_mt_points(); ir++) {
                rho_->f_mt<index_domain_t::local>(0, ir, ialoc) += unit_cell_.atom(ia).symmetry_class().core_charge_density(ir) / y00;
            }
        }
        /* synchronize muffin-tin part */
        rho_->sync_mt();
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            magnetization_[j]->sync_mt();
        }
    }
    
    double nel{0};
    if (ctx_.full_potential()) {
        std::vector<double> nel_mt;
        double nel_it;
        nel = rho_->integrate(nel_mt, nel_it);
    } else {
        nel = rho_->f_pw(0).real() * unit_cell_.omega();
    }

    if (std::abs(nel - unit_cell_.num_electrons()) > 1e-5) {
        std::stringstream s;
        s << "wrong charge density after k-point summation" << std::endl
          << "obtained value : " << nel << std::endl 
          << "target value : " << unit_cell_.num_electrons() << std::endl
          << "difference : " << fabs(nel - unit_cell_.num_electrons()) << std::endl;
        if (ctx_.full_potential()) {
            s << "total core leakage : " << core_leakage();
            for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
                s << std::endl << "  atom class : " << ic << ", core leakage : " << core_leakage(ic);
            }
        }
        WARNING(s);
    }

    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(rhomt): %16llX", rho_->f_mt().hash());
    DUMP("hash(rhoit): %16llX", rho_->f_it().hash());
    #endif

    //if (debug_level > 1) check_density_continuity_at_mt();
}

template <bool mt_spheres>
inline void Density::add_k_point_contribution_rg(K_point* kp__)
{
    PROFILE_WITH_TIMER("sirius::Density::add_k_point_contribution_rg");

    int nfv = ctx_.num_fv_states();
    double omega = unit_cell_.omega();

    mdarray<double, 2> density_rg(ctx_.fft().local_size(), ctx_.num_mag_dims() + 1);
    density_rg.zero();

    #ifdef __GPU
    if (ctx_.fft().hybrid()) {
        density_rg.allocate(memory_t::device);
        density_rg.zero_on_device();
    }
    #endif

    ctx_.fft().prepare(kp__->gkvec().partition());

    int wf_pw_offset = kp__->wf_pw_offset();
        
    /* non-magnetic or collinear case */
    if (ctx_.num_mag_dims() != 3) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            if (!kp__->spinor_wave_functions<mt_spheres>(ispn).spl_num_swapped().global_index_size()) {
                continue;
            }

            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < kp__->spinor_wave_functions<mt_spheres>(ispn).spl_num_swapped().local_size(); i++) {
                int j = kp__->spinor_wave_functions<mt_spheres>(ispn).spl_num_swapped()[i];
                double w = kp__->band_occupancy(j + ispn * nfv) * kp__->weight() / omega;

                /* transform to real space; in case of GPU wave-function stays in GPU memory */
                if (ctx_.fft().gpu_only()) {
                    ctx_.fft().transform<1>(kp__->gkvec().partition(),
                                            kp__->spinor_wave_functions<mt_spheres>(ispn).coeffs_swapped().template at<GPU>(wf_pw_offset, i));
                } else {
                    ctx_.fft().transform<1>(kp__->gkvec().partition(),
                                            kp__->spinor_wave_functions<mt_spheres>(ispn)[i] + wf_pw_offset);
                }

                if (ctx_.fft().hybrid()) {
                    #ifdef __GPU
                    update_density_rg_1_gpu(ctx_.fft().local_size(), ctx_.fft().buffer<GPU>(), w, density_rg.at<GPU>(0, ispn));
                    #else
                    TERMINATE_NO_GPU
                    #endif
                } else {
                    #pragma omp parallel for
                    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                        auto z = ctx_.fft().buffer(ir);
                        density_rg(ir, ispn) += w * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                    }
                }
            }
        }
    } else {
        assert(kp__->spinor_wave_functions<mt_spheres>(0).spl_num_swapped().local_size() ==
               kp__->spinor_wave_functions<mt_spheres>(1).spl_num_swapped().local_size());

        std::vector<double_complex> psi_r(ctx_.fft().local_size());

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < kp__->spinor_wave_functions<mt_spheres>(0).spl_num_swapped().local_size(); i++)
        {
            int j = kp__->spinor_wave_functions<mt_spheres>(0).spl_num_swapped()[i];
            double w = kp__->band_occupancy(j) * kp__->weight() / omega;

            /* transform up- component of spinor function to real space; in case of GPU wave-function stays in GPU memory */
            ctx_.fft().transform<1>(kp__->gkvec().partition(), kp__->spinor_wave_functions<mt_spheres>(0)[i] + wf_pw_offset);
            /* save in auxiliary buffer */
            ctx_.fft().output(&psi_r[0]);
            /* transform dn- component of spinor wave function */
            ctx_.fft().transform<1>(kp__->gkvec().partition(), kp__->spinor_wave_functions<mt_spheres>(1)[i] + wf_pw_offset);

            if (ctx_.fft().hybrid())
            {
                STOP();
                //#ifdef __GPU
                //update_it_density_matrix_1_gpu(ctx_.fft(thread_id)->local_size(), ispn, ctx_.fft(thread_id)->buffer<GPU>(), w,
                //                               it_density_matrix_gpu.at<GPU>());
                //#else
                //TERMINATE_NO_GPU
                //#endif
            }
            else
            {
                #pragma omp parallel for
                for (int ir = 0; ir < ctx_.fft().local_size(); ir++)
                {
                    auto r0 = (std::pow(psi_r[ir].real(), 2) + std::pow(psi_r[ir].imag(), 2)) * w;
                    auto r1 = (std::pow(ctx_.fft().buffer(ir).real(), 2) +
                               std::pow(ctx_.fft().buffer(ir).imag(), 2)) * w;

                    auto z2 = psi_r[ir] * std::conj(ctx_.fft().buffer(ir)) * w;

                    density_rg(ir, 0) += r0;
                    density_rg(ir, 1) += r1;
                    density_rg(ir, 2) += 2.0 * std::real(z2);
                    density_rg(ir, 3) -= 2.0 * std::imag(z2);
                }
            }
        }
    }

    #ifdef __GPU
    if (ctx_.fft().hybrid()) density_rg.copy_to_host();
    #endif
    
    /* switch from real density matrix to density and magnetization */
    switch (ctx_.num_mag_dims()) {
        case 3: {
            #pragma omp parallel for
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                magnetization_[1]->f_rg(ir) += density_rg(ir, 2);
                magnetization_[2]->f_rg(ir) += density_rg(ir, 3);
            }
        }
        case 1: {
            #pragma omp parallel for
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                rho_->f_rg(ir) += (density_rg(ir, 0) + density_rg(ir, 1));
                magnetization_[0]->f_rg(ir) += (density_rg(ir, 0) - density_rg(ir, 1));
            }
            break;
        }
        case 0: {
            #pragma omp parallel for
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                rho_->f_rg(ir) += density_rg(ir, 0);
            }
        }
    }

    ctx_.fft().dismiss();
}

}

#endif // __DENSITY_H__
