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

/** \file density.h
 *   
 *  \brief Contains definition and partial implementation of sirius::Density class.
 */

#ifndef __DENSITY_H__
#define __DENSITY_H__

#include "periodic_function.h"
#include "band.h"
#include "k_set.h"
#include "simulation_context.h"

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
 *  At this point it is straightforward to compute the density and magnetization in the interstitial (see add_k_point_contribution_it()).
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
        
        /// Global set of parameters.
        Simulation_parameters const& parameters_;

        Unit_cell& unit_cell_;

        /// Alias for FFT driver.
        //FFT3D_CPU* fft_;
        
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
        
        std::vector< std::pair<int, int> > dmat_spins_;

        /// Non-zero Gaunt coefficients.
        Gaunt_coefficients<double_complex>* gaunt_coefs_;
        
        /// fast mapping between composite lm index and corresponding orbital quantum number
        std::vector<int> l_by_lm_;

        Mixer<double_complex>* high_freq_mixer_;
        Mixer<double_complex>* low_freq_mixer_;
        Mixer<double>* mixer_;

        std::vector<int> lf_gvec_;
        std::vector<int> hf_gvec_;

        /// Reduce complex density matrix over magnetic quantum numbers
        /** The following operation is performed:
         *  \f[
         *      d_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha} = 
         *          \sum_{mm'} D_{\ell \lambda m, \ell' \lambda' m'}^{\alpha} 
         *          \langle Y_{\ell m} | R_{\ell_3 m_3} | Y_{\ell' m'} \rangle
         *  \f] 
         */
        template <int num_mag_dims> 
        void reduce_density_matrix(Atom_type* atom_type, int ialoc, mdarray<double_complex, 4>& zdens, mdarray<double, 3>& mt_density_matrix);

        /// Add k-point contribution to the auxiliary density matrix.
        /** In case of full-potential LAPW complex density matrix has the following expression:
         *  \f[
         *      D_{\xi \sigma, \xi' \sigma'}^{\alpha} = \sum_{j{\bf k}} n_{j{\bf k}}
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
        template <electronic_structure_method_t basis>
        void add_k_point_contribution(K_point* kp__,
                                      occupied_bands_descriptor const& occupied_bands__,
                                      mdarray<double_complex, 4>& density_matrix__);

        /// Restore valence density by adding the Q-operator constribution
        /** The following term is added to the valence density, generated by the pseudo wave-functions:
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
        void add_q_contribution_to_valence_density(K_set& kset);

        /// Add k-point contribution to the interstitial density and magnetization
        template <bool mt_spheres>
        void add_k_point_contribution_it(K_point* kp__);

        #ifdef __GPU
        void add_q_contribution_to_valence_density_gpu(K_set& ks);
        #endif

        /// Generate valence density in the muffin-tins 
        void generate_valence_density_mt(K_set& ks);
        
        /// Generate valence density in the interstitial
        void generate_valence_density_it(K_set& ks);
       
        /// Add band contribution to the muffin-tin density
        void add_band_contribution_mt(Band* band, double weight, mdarray<double_complex, 3>& fylm, 
                                      std::vector<Periodic_function<double>*>& dens);
        
        /// Generate charge density of core states
        void generate_core_charge_density();

        void generate_pseudo_core_charge_density();

    public:

        /// Constructor
        Density(Simulation_context& ctx_);
        
        /// Destructor
        ~Density();
       
        /// Set pointers to muffin-tin and interstitial charge density arrays
        void set_charge_density_ptr(double* rhomt, double* rhoir);
        
        /// Set pointers to muffin-tin and interstitial magnetization arrays
        void set_magnetization_ptr(double* magmt, double* magir);
        
        /// Zero density and magnetization
        void zero();
        
        /// Generate initial charge density and magnetization
        void initial_density();

        /// Find the total leakage of the core states out of the muffin-tins
        double core_leakage();
        
        /// Return core leakage for a specific atom symmetry class
        double core_leakage(int ic);

        /// Generate full charge density (valence + core) and magnetization from the wave functions.
        void generate(K_set& ks__);

        /// Generate valence charge density and magnetization from the wave functions.
        void generate_valence(K_set& ks__);
        
        /// Add augmentation charge Q(r)
        void augment(K_set& ks__);
        
        /// Integrtae charge density to get total and partial charges
        //** void integrate();

        /// Check density at MT boundary
        void check_density_continuity_at_mt();

        void generate_pw_coefs();
         
        void save();
        
        void load();

        mdarray<double, 2> generate_rho_radial_integrals(int type__);

        inline size_t size()
        {
            size_t s = rho_->size();
            for (int i = 0; i < parameters_.num_mag_dims(); i++) s += magnetization_[i]->size();
            return s;
        }

        inline void pack(Mixer<double>* mixer__)
        {
            size_t n = rho_->pack(0, mixer__);
            for (int i = 0; i < parameters_.num_mag_dims(); i++) n += magnetization_[i]->pack(n, mixer__);
        }

        inline void unpack(double const* buffer__)
        {
            size_t n = rho_->unpack(buffer__);
            for (int i = 0; i < parameters_.num_mag_dims(); i++) n += magnetization_[i]->unpack(&buffer__[n]);
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

        Spheric_function<spectral, double>& density_mt(int ialoc)
        {
            return rho_->f_mt(ialoc);
        }

        void allocate()
        {
            rho_->allocate(true);
            for (int j = 0; j < parameters_.num_mag_dims(); j++) magnetization_[j]->allocate(true);
        }

        void mixer_input()
        {
            if (mixer_ != nullptr)
            {
                pack(mixer_);
            }
            else
            {
                int k = 0;
                for (int i = 0; i < ctx_.gvec_coarse().num_gvec(); i++)
                {
                    //low_freq_mixer_->input(k++, rho_->f_pw(ig));
                    low_freq_mixer_->input(k++, rho_->f_pw(lf_gvec_[i]));
                }

                k = 0;
                //for (int ig = ctx_.fft_coarse()->num_gvec(); ig < ctx_.fft()->num_gvec(); ig++)
                for (int i = 0; i < ctx_.gvec().num_gvec() - ctx_.gvec_coarse().num_gvec(); i++)
                {
                    high_freq_mixer_->input(k++, rho_->f_pw(hf_gvec_[i]));
                }
            }
        }

        void mixer_output()
        {
            if (mixer_ != nullptr)
            {
                unpack(mixer_->output_buffer());
            }
            else
            {
                int ngv = ctx_.gvec().num_gvec();
                int ngvc = ctx_.gvec_coarse().num_gvec();
                
                for (int i = 0; i < ngvc; i++)
                {
                    rho_->f_pw(lf_gvec_[i]) = low_freq_mixer_->output_buffer(i);
                }
                for (int i = 0; i < ngv - ngvc; i++)
                {
                    rho_->f_pw(hf_gvec_[i]) = high_freq_mixer_->output_buffer(i);
                }

                //memcpy(&rho_->f_pw(0), low_freq_mixer_->output_buffer(), ngvc * sizeof(double_complex));
                //memcpy(&rho_->f_pw(ngvc), high_freq_mixer_->output_buffer(), (ngv - ngvc) * sizeof(double_complex));
            }
        }

        void mixer_init()
        {
            mixer_input();

            if (mixer_ != nullptr)
            {
                mixer_->initialize();
            }
            else
            {
                low_freq_mixer_->initialize();
                high_freq_mixer_->initialize();
            }
        }

        double mix()
        {
            double rms;

            if (mixer_ != nullptr)
            {
                /* mix in real-space in case of FP-LAPW */
                mixer_input();
                rms = mixer_->mix();
                mixer_output();
                rho_->fft_transform(-1);
            }
            else
            {
                /* mix in G-space in case of PP */
                mixer_input();
                rms = low_freq_mixer_->mix();
                rms += high_freq_mixer_->mix();
                mixer_output();
                rho_->fft_transform(1);
            }

            return rms;
        }

        inline double dr2()
        {
            return low_freq_mixer_->rss();
        }
};

}

#endif // __DENSITY_H__
