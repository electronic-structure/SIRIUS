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

/** \file density.h
 *   
 *  \brief Contains definition and partial implementation of sirius::Density class.
 */

#ifndef __DENSITY_H__
#define __DENSITY_H__

#include "periodic_function.h"
#include "k_point_set.h"
#include "simulation_context.h"
#include "mixer.h"

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
                                        double_complex const* psi_rg__, 
                                        double wt__, 
                                        double* density_rg__);

extern "C" void update_density_rg_2_gpu(int size__, 
                                        double_complex const* psi_rg_up__, 
                                        double_complex const* psi_rg_dn__, 
                                        double wt__, 
                                        double* density_x_rg__,
                                        double* density_y_rg__);
#endif

namespace sirius {

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

        /// Context of the simulation.
        Simulation_context& ctx_;
        
        /// Alias to ctx_.unit_cell()
        Unit_cell& unit_cell_;

        /// Density matrix for all atoms.
        mdarray<double_complex, 4> density_matrix_; // TODO: make it local for LAPW

        struct paw_density_data_t
        {
            Atom *atom_{nullptr};

            int ia{-1};

            /// ae and ps local densities
            mdarray<double, 2> ae_density_; // TODO: use Spheric_function
            mdarray<double, 2> ps_density_;

            mdarray<double, 3> ae_magnetization_;
            mdarray<double, 3> ps_magnetization_;
        };

        std::vector<paw_density_data_t> paw_density_data_;

        /// Pointer to charge density.
        /** In the case of full-potential calculation this is the full (valence + core) electron charge density.
         *  In the case of pseudopotential this is the valence charge density. */ 
        std::unique_ptr<Periodic_function<double>> rho_{nullptr};

        /// Magnetization
        std::array<std::unique_ptr<Periodic_function<double>>, 3> magnetization_;

        /// Alias for density and magnetization.
        std::array<Periodic_function<double>*, 4> rho_vec_{{nullptr, nullptr, nullptr, nullptr}};
        
        /// Density and magnetization on the coarse FFT mesh.
        /** Coarse FFT grid is enough to generate density and magnetization from the wave-functions. The components
         *  of the <tt>rho_mag_coarse</tt> vector have the following order:
         *  \f$ \{\rho({\bf r}), m_z({\bf r}), m_x({\bf r}), m_y({\bf r}) \} \f$. */
        std::array<std::unique_ptr<Smooth_periodic_function<double>>, 4> rho_mag_coarse_;

        /// Pointer to pseudo core charge density
        /** In the case of pseudopotential we need to know the non-linear core correction to the 
         *  exchange-correlation energy which is introduced trough the pseudo core density: 
         *  \f$ E_{xc}[\rho_{val} + \rho_{core}] \f$. The 'pseudo' reflects the fact that 
         *  this density integrated does not reproduce the total number of core elctrons. */
        std::unique_ptr<Smooth_periodic_function<double>> rho_pseudo_core_{nullptr};
        
        /// Non-zero Gaunt coefficients.
        std::unique_ptr<Gaunt_coefficients<double_complex>> gaunt_coefs_{nullptr};
        
        /// Fast mapping between composite lm index and corresponding orbital quantum number.
        mdarray<int, 1> l_by_lm_;

        /// High-frequency mixer for the pseudopotential density mixing.
        std::unique_ptr<Mixer<double_complex>> hf_mixer_{nullptr};

        /// Low-frequency mixer for the pseudopotential density mixing.
        std::unique_ptr<Mixer<double_complex>> lf_mixer_{nullptr};

        /// Mixer for the full-potential density mixing.
        std::unique_ptr<Mixer<double>> mixer_{nullptr};
        
        /// List of local low-fequency G-vectors.
        std::vector<int> lf_gvec_;

        /// List of local high-fequency G-vectors.
        std::vector<int> hf_gvec_;

        /// Weights of local low-frequency G-vectors.
        std::vector<double> lf_gvec_weights_;

        /// Allocate PAW data.
        void init_paw();

        void generate_paw_atom_density(paw_density_data_t &pdd);

        /// Initialize \rho_{ij} - density matrix, occupation on basis of beta-projectors (used for PAW).
        void init_density_matrix_for_paw();

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
         *      n_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha} = 
         *          \sum_{mm'} D_{\ell \lambda m, \ell' \lambda' m'}^{\alpha} 
         *          \langle Y_{\ell m} | R_{\ell_3 m_3} | Y_{\ell' m'} \rangle
         *  \f] 
         */
        template <int num_mag_dims>
        void reduce_density_matrix(Atom_type const&                          atom_type__,
                                   int                                       ia__,
                                   mdarray<double_complex, 4> const&         zdens__,
                                   Gaunt_coefficients<double_complex> const& gaunt_coeffs__,
                                   mdarray<double, 3>&                       mt_density_matrix__)
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
                                auto gc = gaunt_coeffs__.gaunt(lm1, lm2, k).coef;
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

        /// Add k-point contribution to the density matrix in the canonical form.
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
        inline void add_k_point_contribution_rg(K_point* kp__);

        /// Generate valence density in the muffin-tins 
        void generate_valence_mt(K_point_set& ks);
        
        /// Generate charge density of core states
        void generate_core_charge_density()
        {
            PROFILE("sirius::Density::generate_core_charge_density");

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
            PROFILE("sirius::Density::generate_pseudo_core_charge_density");

            auto ri = Radial_integrals_rho_core_pseudo<false>(ctx_.unit_cell(), ctx_.pw_cutoff(), ctx_.settings().nprii_rho_core_);

            auto v = ctx_.make_periodic_function<index_domain_t::local>([&ri](int iat, double g)
                                                                        {
                                                                            return ri.value(iat, g);
                                                                        });
            std::copy(v.begin(), v.end(), &rho_pseudo_core_->f_pw_local(0));
            rho_pseudo_core_->fft_transform(1);
        }

    public:

        /// Constructor
        Density(Simulation_context& ctx__)
            : ctx_(ctx__)
            , unit_cell_(ctx_.unit_cell())
        {
            /* allocate charge density */
            rho_ = std::unique_ptr<Periodic_function<double>>(new Periodic_function<double>(ctx_, ctx_.lmmax_rho()));
            rho_vec_[0] = rho_.get();
            
            /* allocate magnetization density */
            for (int i = 0; i < ctx_.num_mag_dims(); i++) {
                magnetization_[i] = std::unique_ptr<Periodic_function<double>>(new Periodic_function<double>(ctx_, ctx_.lmmax_rho()));
                rho_vec_[i + 1] = magnetization_[i].get();
            }
            
            /*  allocate charge density and magnetization on a coarse grid */
            for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
                rho_mag_coarse_[i] = std::unique_ptr<Smooth_periodic_function<double>>(new Smooth_periodic_function<double>(ctx_.fft_coarse(), ctx_.gvec_coarse()));
            }

            /* core density of the pseudopotential method */
            if (!ctx_.full_potential()) {
                rho_pseudo_core_ = std::unique_ptr<Smooth_periodic_function<double>>(new Smooth_periodic_function<double>(ctx_.fft(), ctx_.gvec()));
                rho_pseudo_core_->zero();

                generate_pseudo_core_charge_density();
            }

            if (ctx_.full_potential()) {
                using gc_z = Gaunt_coefficients<double_complex>;
                gaunt_coefs_ = std::unique_ptr<gc_z>(new gc_z(ctx_.lmax_apw(), ctx_.lmax_rho(), ctx_.lmax_apw(), SHT::gaunt_hybrid));
            }

            l_by_lm_ = Utils::l_by_lm(ctx_.lmax_rho());

            density_matrix_ = mdarray<double_complex, 4>(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), 
                                                         ctx_.num_mag_comp(), unit_cell_.num_atoms());
            density_matrix_.zero();

            /* split local G-vectors to low-frequency and high-frequency */
            for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                int ig = ctx_.gvec().offset() + igloc;
                auto gv = ctx_.gvec().gvec_cart(ig);
                if (gv.length() <= 2 * ctx_.gk_cutoff()) {
                    lf_gvec_.push_back(igloc);
                    if (ig) {
                        lf_gvec_weights_.push_back(fourpi * unit_cell_.omega() / std::pow(gv.length(), 2));
                    } else {
                        lf_gvec_weights_.push_back(0);
                    }
                } else {
                    hf_gvec_.push_back(igloc);
                }
            }
        }
        
        /// Set pointers to muffin-tin and interstitial charge density arrays
        void set_charge_density_ptr(double* rhomt, double* rhorg)
        {
            if (ctx_.full_potential() && rhomt) {
                rho_->set_mt_ptr(rhomt);
            }
            if (rhorg) {
                rho_->set_rg_ptr(rhorg);
            }
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
                if (magmt) {
                    magnetization_[0]->set_mt_ptr(&magmt_tmp(0, 0, 0, 0));
                }
                if (magir) {
                    magnetization_[0]->set_rg_ptr(&magir_tmp(0, 0));
                }
            }

            if (ctx_.num_mag_dims() == 3) {
                if (magmt) {
                    /* z component is the first */
                    magnetization_[0]->set_mt_ptr(&magmt_tmp(0, 0, 0, 2));
                    /* x component is the second */
                    magnetization_[1]->set_mt_ptr(&magmt_tmp(0, 0, 0, 0));
                    /* y component is the third */
                    magnetization_[2]->set_mt_ptr(&magmt_tmp(0, 0, 0, 1));
                }
                if (magir) {
                    /* z component is the first */
                    magnetization_[0]->set_rg_ptr(&magir_tmp(0, 2));
                    /* x component is the second */
                    magnetization_[1]->set_rg_ptr(&magir_tmp(0, 0));
                    /* y component is the third */
                    magnetization_[2]->set_rg_ptr(&magir_tmp(0, 1));
                }
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

        /// Check total density for the correct number of electrons.
        inline void check_num_electrons()
        {
            double nel{0};
            if (ctx_.full_potential()) {
                std::vector<double> nel_mt;
                double nel_it;
                nel = rho_->integrate(nel_mt, nel_it);
            } else {
                nel = rho_->f_0().real() * unit_cell_.omega();
            }
            
            /* check the number of electrons */
            if (std::abs(nel - unit_cell_.num_electrons()) > 1e-5) {
                std::stringstream s;
                s << "wrong number of electrons" << std::endl
                  << "  obtained value : " << nel << std::endl 
                  << "  target value : " << unit_cell_.num_electrons() << std::endl
                  << "  difference : " << std::abs(nel - unit_cell_.num_electrons()) << std::endl;
                if (ctx_.full_potential()) {
                    s << "  total core leakage : " << core_leakage();
                    for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
                        s << std::endl << "    atom class : " << ic << ", core leakage : " << core_leakage(ic);
                    }
                }
                WARNING(s);
            }
        }

        /// Generate full charge density (valence + core) and magnetization from the wave functions.
        /** This function calls generate_valence() and then in case of full-potential LAPW method adds a core density
         *  to get the full charge density of the system. */
        inline void generate(K_point_set& ks__)
        {
            PROFILE("sirius::Density::generate");

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
        }

        /// Generate valence charge density and magnetization from the wave functions.
        /** The interstitial density is generated on the coarse FFT grid and then transformed to the PW domain.
         *  After symmetrization and mixing and before the generation of the XC potential density is transformted to the
         *  real-space domain and checked for the number of electrons. */
        inline void generate_valence(K_point_set& ks__);

        /// Add augmentation charge Q(r).
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
        void augment(K_point_set& ks__)
        {
            PROFILE("sirius::Density::augment");

            /*check if we need to augment charge density and magnetization */
            bool need_to_augment{false};
            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                need_to_augment |= unit_cell_.atom_type(iat).pp_desc().augment;
            }
            if (!need_to_augment) {
                return;
            }

            //if (ctx_.control().print_checksum_) {
            //    for (auto e: rho_vec_) {
            //        auto cs = e->checksum_pw();
            //        DUMP("checksum(rho_vec_pw): %20.14f %20.14f", cs.real(), cs.imag());
            //    }
            //}
            
            mdarray<double_complex, 2> rho_aug(ctx_.gvec().count(), ctx_.num_mag_dims() + 1, ctx_.dual_memory_t());

            switch (ctx_.processing_unit()) {
                case CPU: {
                    generate_rho_aug<CPU>(rho_aug);
                    break;
                }
                case GPU: {
                    generate_rho_aug<GPU>(rho_aug);
                    break;
                }
            }

            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                #pragma omp parallel for
                for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                    rho_vec_[iv]->f_pw_local(igloc) += rho_aug(igloc, iv);
                }
            }
        }

        template <device_t pu>
        inline void generate_rho_aug(mdarray<double_complex, 2>& rho_aug__);

        /// Check density at MT boundary
        void check_density_continuity_at_mt();

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

        void save_to_xsf()
        {
            //== FILE* fout = fopen("unit_cell.xsf", "w");
            //== fprintf(fout, "CRYSTAL\n");
            //== fprintf(fout, "PRIMVEC\n");
            //== auto& lv = unit_cell_.lattice_vectors();
            //== for (int i = 0; i < 3; i++)
            //== {
            //==     fprintf(fout, "%18.12f %18.12f %18.12f\n", lv(0, i), lv(1, i), lv(2, i));
            //== }
            //== fprintf(fout, "CONVVEC\n");
            //== for (int i = 0; i < 3; i++)
            //== {
            //==     fprintf(fout, "%18.12f %18.12f %18.12f\n", lv(0, i), lv(1, i), lv(2, i));
            //== }
            //== fprintf(fout, "PRIMCOORD\n");
            //== fprintf(fout, "%i 1\n", unit_cell_.num_atoms());
            //== for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
            //== {
            //==     auto pos = unit_cell_.get_cartesian_coordinates(unit_cell_.atom(ia).position());
            //==     fprintf(fout, "%i %18.12f %18.12f %18.12f\n", unit_cell_.atom(ia).zn(), pos[0], pos[1], pos[2]);
            //== }
            //== fclose(fout);
        }

        void save_to_xdmf()
        {
            //== mdarray<double, 3> rho_grid(&rho_->f_it<global>(0), fft_->size(0), fft_->size(1), fft_->size(2));
            //== mdarray<double, 4> pos_grid(3, fft_->size(0), fft_->size(1), fft_->size(2));

            //== mdarray<double, 4> mag_grid(3, fft_->size(0), fft_->size(1), fft_->size(2));
            //== mag_grid.zero();

            //== // loop over 3D array (real space)
            //== for (int j0 = 0; j0 < fft_->size(0); j0++)
            //== {
            //==     for (int j1 = 0; j1 < fft_->size(1); j1++)
            //==     {
            //==         for (int j2 = 0; j2 < fft_->size(2); j2++)
            //==         {
            //==             int ir = static_cast<int>(j0 + j1 * fft_->size(0) + j2 * fft_->size(0) * fft_->size(1));
            //==             // get real space fractional coordinate
            //==             double frv[] = {double(j0) / fft_->size(0),
            //==                             double(j1) / fft_->size(1),
            //==                             double(j2) / fft_->size(2)};
            //==             vector3d<double> rv = ctx_.unit_cell()->get_cartesian_coordinates(vector3d<double>(frv));
            //==             for (int x = 0; x < 3; x++) pos_grid(x, j0, j1, j2) = rv[x];
            //==             if (ctx_.num_mag_dims() == 1) mag_grid(2, j0, j1, j2) = magnetization_[0]->f_it<global>(ir);
            //==             if (ctx_.num_mag_dims() == 3)
            //==             {
            //==                 mag_grid(0, j0, j1, j2) = magnetization_[1]->f_it<global>(ir);
            //==                 mag_grid(1, j0, j1, j2) = magnetization_[2]->f_it<global>(ir);
            //==             }
            //==         }
            //==     }
            //== }

            //== HDF5_tree h5_rho("rho.hdf5", true);
            //== h5_rho.write("rho", rho_grid);
            //== h5_rho.write("pos", pos_grid);
            //== h5_rho.write("mag", mag_grid);

            //== FILE* fout = fopen("rho.xdmf", "w");
            //== //== fprintf(fout, "<?xml version=\"1.0\" ?>\n"
            //== //==               "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">\n"
            //== //==               "<Xdmf>\n"
            //== //==               "  <Domain Name=\"name1\">\n"
            //== //==               "    <Grid Name=\"fft_fine_grid\" Collection=\"Unknown\">\n"
            //== //==               "      <Topology TopologyType=\"3DSMesh\" NumberOfElements=\" %i %i %i \"/>\n"
            //== //==               "      <Geometry GeometryType=\"XYZ\">\n"
            //== //==               "        <DataItem Dimensions=\"%i %i %i 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">rho.hdf5:/pos</DataItem>\n"
            //== //==               "      </Geometry>\n"
            //== //==               "      <Attribute\n"
            //== //==               "           AttributeType=\"Scalar\"\n"
            //== //==               "           Center=\"Node\"\n"
            //== //==               "           Name=\"rho\">\n"
            //== //==               "          <DataItem\n"
            //== //==               "             NumberType=\"Float\"\n"
            //== //==               "             Precision=\"8\"\n"
            //== //==               "             Dimensions=\"%i %i %i\"\n"
            //== //==               "             Format=\"HDF\">\n"
            //== //==               "             rho.hdf5:/rho\n"
            //== //==               "          </DataItem>\n"
            //== //==               "        </Attribute>\n"
            //== //==               "    </Grid>\n"
            //== //==               "  </Domain>\n"
            //== //==               "</Xdmf>\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2));
            //== fprintf(fout, "<?xml version=\"1.0\" ?>\n"
            //==               "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">\n"
            //==               "<Xdmf>\n"
            //==               "  <Domain Name=\"name1\">\n"
            //==               "    <Grid Name=\"fft_fine_grid\" Collection=\"Unknown\">\n"
            //==               "      <Topology TopologyType=\"3DSMesh\" NumberOfElements=\" %i %i %i \"/>\n"
            //==               "      <Geometry GeometryType=\"XYZ\">\n"
            //==               "        <DataItem Dimensions=\"%i %i %i 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">rho.hdf5:/pos</DataItem>\n"
            //==               "      </Geometry>\n"
            //==               "      <Attribute\n"
            //==               "           AttributeType=\"Vector\"\n"
            //==               "           Center=\"Node\"\n"
            //==               "           Name=\"mag\">\n"
            //==               "          <DataItem\n"
            //==               "             NumberType=\"Float\"\n"
            //==               "             Precision=\"8\"\n"
            //==               "             Dimensions=\"%i %i %i 3\"\n"
            //==               "             Format=\"HDF\">\n"
            //==               "             rho.hdf5:/mag\n"
            //==               "          </DataItem>\n"
            //==               "        </Attribute>\n"
            //==               "    </Grid>\n"
            //==               "  </Domain>\n"
            //==               "</Xdmf>\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2));
            //== fclose(fout);
        }

        Periodic_function<double>* rho()
        {
            return rho_.get();
        }
        
        Smooth_periodic_function<double>& rho_pseudo_core()
        {
            return *rho_pseudo_core_;
        }
        
        std::array<Periodic_function<double>*, 3> magnetization()
        {
            return {magnetization_[0].get(), magnetization_[1].get(), magnetization_[2].get()};
        }

        Periodic_function<double>* magnetization(int i)
        {
            return magnetization_[i].get();
        }

        Spheric_function<spectral, double> const& density_mt(int ialoc) const
        {
            return rho_->f_mt(ialoc);
        }

        /// Generate \f$ n_1 \f$  and \f$ \tilde{n}_1 \f$ in lm components.
        void generate_paw_loc_density();

        mdarray<double, 2> const& ae_paw_atom_density(int spl_paw_ind) const
        {
            return paw_density_data_[spl_paw_ind].ae_density_;
        }

        mdarray<double, 2> const& ps_paw_atom_density(int spl_paw_ind) const
        {
            return paw_density_data_[spl_paw_ind].ps_density_;
        }

        mdarray<double, 3> const& ae_paw_atom_magn(int spl_paw_ind) const
        {
            return paw_density_data_[spl_paw_ind].ae_magnetization_;
        }

        mdarray<double, 3> const& ps_paw_atom_magn(int spl_paw_ind) const
        {
            return paw_density_data_[spl_paw_ind].ps_magnetization_;
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
            if (ctx_.full_potential()) {
                STOP();
            } else {
                int ld = static_cast<int>(hf_gvec_.size());
                /* input high-frequency components */
                for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                    for (int i = 0; i < static_cast<int>(hf_gvec_.size()); i++) {
                        int igloc = hf_gvec_[i];
                        hf_mixer_->input_local(i + j * ld, rho_vec_[j]->f_pw_local(igloc));
                    }
                }
                ld = static_cast<int>(lf_gvec_.size());
                /* input low-frequency components */
                for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                    if (j == 0) {
                        for (int i = 0; i < static_cast<int>(lf_gvec_.size()); i++) {
                            int igloc = lf_gvec_[i];
                            lf_mixer_->input_local(i + j * ld, rho_vec_[j]->f_pw_local(igloc), lf_gvec_weights_[i]);
                        }
                    } else {
                        for (int i = 0; i < static_cast<int>(lf_gvec_.size()); i++) {
                            int igloc = lf_gvec_[i];
                            lf_mixer_->input_local(i + j * ld, rho_vec_[j]->f_pw_local(igloc));
                        }
                    }
                }
                /* input commonly shared data */
                for (int i = 0; i < static_cast<int>(density_matrix_.size()); i++) {
                     lf_mixer_->input_shared(i, density_matrix_[i], 0);
                }
            }
        }

        void mixer_output()
        {
            if (ctx_.full_potential()) {
                STOP();
            } else {
                int ld = static_cast<int>(hf_gvec_.size());
                /* get high-frequency components */
                for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                    for (int i = 0; i < static_cast<int>(hf_gvec_.size()); i++) {
                        int igloc = hf_gvec_[i];
                        rho_vec_[j]->f_pw_local(igloc) = hf_mixer_->output_local(i + j * ld);
                    }
                }

                ld = static_cast<int>(lf_gvec_.size());
                /* get low-frequency components */
                for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                    for (int i = 0; i < static_cast<int>(lf_gvec_.size()); i++) {
                        int igloc = lf_gvec_[i];
                        rho_vec_[j]->f_pw_local(igloc) = lf_mixer_->output_local(i + j * ld);
                    }
                }

                for (int i = 0; i < static_cast<int>(density_matrix_.size()); i++) {
                    density_matrix_[i] = lf_mixer_->output_shared(i);
                }
            }
        }

        void mixer_init()
        {
            if (!ctx_.full_potential()) {
                hf_mixer_ = Mixer_factory<double_complex>("linear",
                                                          0,
                                                          static_cast<int>(hf_gvec_.size() * (1 + ctx_.num_mag_dims())),
                                                          ctx_.mixer_input(),
                                                          ctx_.comm());

                lf_mixer_ = Mixer_factory<double_complex>(ctx_.mixer_input().type_,
                                                          static_cast<int>(density_matrix_.size()),
                                                          static_cast<int>(lf_gvec_.size() * (1 + ctx_.num_mag_dims())),
                                                          ctx_.mixer_input(),
                                                          ctx_.comm());
            } else {
                //mixer_ = Mixer_factory<double>(ctx_.mixer_input().type_, size(), ctx_.mixer_input(), ctx_.comm());
            }

            mixer_input();

            if (ctx_.full_potential()) {
                mixer_->initialize();
            } else {
                lf_mixer_->initialize();
                if (hf_mixer_) {
                    hf_mixer_->initialize();
                }
            }
        }

        double mix()
        {
            double rms;

            if (ctx_.full_potential()) {
                STOP();
                /* mix in real-space in case of FP-LAPW */
                mixer_input();
                rms = mixer_->mix();
                mixer_output();
                /* get rho(G) after mixing */
                rho_->fft_transform(-1);
            } else {
                /* mix in G-space in case of PP */
                mixer_input();
                rms = lf_mixer_->mix();
                if (hf_mixer_) {
                    rms += hf_mixer_->mix();
                }
                mixer_output();
            }

            return rms;
        }

        inline double dr2()
        {
            return lf_mixer_->rss();
        }

        mdarray<double_complex, 4> const& density_matrix() const
        {
            return density_matrix_;
        }

        mdarray<double_complex, 4>& density_matrix()
        {
            return density_matrix_;
        }

        inline void fft_transform(int direction__)
        {
            rho_->fft_transform(direction__);
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                magnetization_[j]->fft_transform(direction__);
            }
        }

        /// Return density matrix in auxiliary form.
        inline mdarray<double, 3> density_matrix_aux(int iat__)
        {
            auto& atom_type = unit_cell_.atom_type(iat__);
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
                            case 3: {
                                dm(idx12, i, 2) = 2 * std::real(density_matrix_(xi2, xi1, 2, ia));
                                dm(idx12, i, 3) = -2 * std::imag(density_matrix_(xi2, xi1, 2, ia));
                            }
                            case 1: {
                                dm(idx12, i, 0) = std::real(density_matrix_(xi2, xi1, 0, ia) + density_matrix_(xi2, xi1, 1, ia));
                                dm(idx12, i, 1) = std::real(density_matrix_(xi2, xi1, 0, ia) - density_matrix_(xi2, xi1, 1, ia));
                                break;
                            }
                            case 0: {
                                dm(idx12, i, 0) = density_matrix_(xi2, xi1, 0, ia).real();
                                break;
                            }
                        }
                    }
                }
            }
            return std::move(dm);
        }

};

#include "Density/initial_density.hpp"
#include "Density/add_k_point_contribution_rg.hpp"
#include "Density/add_k_point_contribution_dm.hpp"
#include "Density/generate_valence.hpp"
#include "Density/generate_rho_aug.hpp"
#include "Density/symmetrize_density_matrix.hpp"
#include "Density/generate_valence_mt.hpp"
#include "Density/check_density_continuity_at_mt.hpp"
#include "Density/paw_density.hpp"
}

#endif // __DENSITY_H__
