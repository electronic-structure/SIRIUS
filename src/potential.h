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

/** \file potential.h
 *   
 *  \brief Contains declaration and partial implementation of sirius::Potential class.
 */

#ifndef __POTENTIAL_H__
#define __POTENTIAL_H__

#include "periodic_function.h"
#include "spheric_function.h"
#include "simulation_context.h"
#include "density.h"

namespace sirius {

/// Generate effective potential from charge density and magnetization.
/** \note At some point we need to update the atomic potential with the new MT potential. This is simple if the 
          effective potential is a global function. Otherwise we need to pass the effective potential between MPI ranks.
          This is also simple, but requires some time. It is also easier to mix the global functions.  */
class Potential 
{
    private:

        Simulation_context& ctx_;

        Unit_cell& unit_cell_;

        Communicator const& comm_;

        std::unique_ptr<Periodic_function<double>> effective_potential_;

        Periodic_function<double>* effective_magnetic_field_[3];
 
        Periodic_function<double>* hartree_potential_;
        Periodic_function<double>* xc_potential_;
        Periodic_function<double>* xc_energy_density_;

        /// Local part of pseudopotential.
        std::unique_ptr<Periodic_function<double>> local_potential_;

        mdarray<double, 3> sbessel_mom_;

        mdarray<double, 3> sbessel_mt_;

        mdarray<double, 2> gamma_factors_R_;

        int lmax_;

        std::unique_ptr<SHT> sht_;

        int pseudo_density_order_{9};

        std::vector<double_complex> zil_;

        std::vector<double_complex> zilm_;

        mdarray<int, 1> l_by_lm_;

        mdarray<double_complex, 2> gvec_ylm_;

        double energy_vha_;

        /// Electronic part of Hartree potential.
        /** Used to compute electron-nuclear contribution to the total energy */
        mdarray<double, 1> vh_el_;

        std::unique_ptr<Mixer<double>> mixer_{nullptr};

        std::vector<XC_functional> xc_func_;

        /// Plane-wave coefficients of the effective potential weighted by the unit step-function.
        mdarray<double_complex, 1> veff_pw_;

        /// Plane-wave coefficients of the inverse relativistic mass weighted by the unit step-function.
        mdarray<double_complex, 1> rm_inv_pw_;
 
        /// Plane-wave coefficients of the squared inverse relativistic mass weighted by the unit step-function.
        mdarray<double_complex, 1> rm2_inv_pw_;

        struct paw_potential_data_t
        {
            Atom *atom_{nullptr};

            int ia{-1};

            int ia_paw{-1};

            mdarray<double, 3> ae_potential_; // TODO: -> Spheric_function
            mdarray<double, 3> ps_potential_;

            double hartree_energy_{0.0};
            double xc_energy_{0.0};
            double core_energy_{0.0};
            double one_elec_energy_{0.0};
        };

        std::vector<double> paw_hartree_energies_;
        std::vector<double> paw_xc_energies_;
        std::vector<double> paw_core_energies_;
        std::vector<double> paw_one_elec_energies_;

        double paw_hartree_total_energy_{0.0};
        double paw_xc_total_energy_{0.0};
        double paw_total_core_energy_{0.0};
        double paw_one_elec_energy_{0.0};

        std::vector<paw_potential_data_t> paw_potential_data_;

        mdarray<double_complex, 4> paw_dij_;

        int max_paw_basis_size_{0};

        void init_PAW();

        double xc_mt_PAW_nonmagnetic(Radial_grid<double> const& rgrid,
                                     mdarray<double, 3>& out_atom_pot,
                                     mdarray<double, 2> const& full_rho_lm,
                                     std::vector<double> const& rho_core);


        double xc_mt_PAW_collinear(Radial_grid<double> const& rgrid,
                                   mdarray<double,3> &out_atom_pot,
                                   mdarray<double,2> const& full_rho_lm,
                                   mdarray<double,3> const& magnetization_lm,
                                   std::vector<double> const& rho_core);

        // TODO DO
        void xc_mt_PAW_noncollinear(    )   {     };

        void calc_PAW_local_potential(paw_potential_data_t &pdd,
                                      mdarray<double, 2> const& ae_full_density,
                                      mdarray<double, 2> const& ps_full_density,
                                      mdarray<double, 3> const& ae_local_magnetization,
                                      mdarray<double, 3> const& ps_local_magnetization);

        void calc_PAW_local_Dij(paw_potential_data_t &pdd, mdarray<double_complex, 4>& paw_dij);

        double calc_PAW_hartree_potential(Atom& atom, const Radial_grid<double>& grid,
                                          mdarray<double, 2> const& full_density,
                                          mdarray<double, 3>& out_atom_pot);

        double calc_PAW_one_elec_energy(paw_potential_data_t &pdd,
                                        const mdarray<double_complex, 4>& density_matrix,
                                        const mdarray<double_complex, 4>& paw_dij);


        void add_paw_Dij_to_atom_Dmtrx();
        
        /// Compute MT part of the potential and MT multipole moments
        inline void poisson_vmt(Periodic_function<double>* rho__, 
                                Periodic_function<double>* vh__,
                                mdarray<double_complex, 2>& qmt__)
        {
            PROFILE("sirius::Potential::poisson_vmt");

            qmt__.zero();
            
            for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
                int ia = unit_cell_.spl_num_atoms(ialoc);

                auto qmt = poisson_vmt<false>(unit_cell_.atom(ia), rho__->f_mt(ialoc),
                                              const_cast<Spheric_function<function_domain_t::spectral, double>&>(vh__->f_mt(ialoc)));

                SHT::convert(ctx_.lmax_rho(), &qmt[0], &qmt__(0, ia));
            }

            ctx_.comm().allreduce(&qmt__(0, 0), (int)qmt__.size());
        }

        /// Perform a G-vector summation of plane-wave coefficiens multiplied by radial integrals.
        inline void poisson_sum_G(int lmmax__, 
                                  double_complex* fpw__, 
                                  mdarray<double, 3>& fl__, 
                                  mdarray<double_complex, 2>& flm__);
        
        /// Add contribution from the pseudocharge to the plane-wave expansion
        inline void poisson_add_pseudo_pw(mdarray<double_complex, 2>& qmt, mdarray<double_complex, 2>& qit, double_complex* rho_pw);
        
        /// Generate local part of pseudo potential.
        /** Total local potential is a lattice sum:
         * \f[
         *    V({\bf r}) = \sum_{{\bf T},\alpha} V_{\alpha}({\bf r} - {\bf T} - {\bf \tau}_{\alpha})
         * \f]
         * We want to compute it's plane-wave expansion coefficients:
         * \f[
         *    V({\bf G}) = \frac{1}{V} \int e^{-i{\bf Gr}} V({\bf r}) d{\bf r} =
         *      \frac{1}{V} \sum_{{\bf T},\alpha} \int e^{-i{\bf Gr}}V_{\alpha}({\bf r} - {\bf T} - {\bf \tau}_{\alpha})d{\bf r}
         * \f]
         * Standard change of variables: \f$ {\bf r}' = {\bf r} - {\bf T} - {\bf \tau}_{\alpha},\; {\bf r} = {\bf r}' + {\bf T} + {\bf \tau}_{\alpha} \f$ 
         * leads to:
         * \f[
         *    V({\bf G}) = \frac{1}{V} \sum_{{\bf T},\alpha} \int e^{-i{\bf G}({\bf r}' + {\bf T} + {\bf \tau}_{\alpha})}V_{\alpha}({\bf r}')d{\bf r'} = 
         *    \frac{N}{V} \sum_{\alpha} \int e^{-i{\bf G}({\bf r}' + {\bf \tau}_{\alpha})}V_{\alpha}({\bf r}')d{\bf r'} = 
         *    \frac{1}{\Omega} \sum_{\alpha} e^{-i {\bf G} {\bf \tau}_{\alpha} } \int e^{-i{\bf G}{\bf r}}V_{\alpha}({\bf r})d{\bf r} 
         * \f]
         * Using the well-known expansion of a plane wave in terms of spherical Bessel functions:
         * \f[
         *   e^{i{\bf G}{\bf r}}=4\pi \sum_{\ell m} i^\ell j_{\ell}(Gr)Y_{\ell m}^{*}({\bf \hat G})Y_{\ell m}({\bf \hat r})
         * \f]
         * and remembering that for \f$ \ell = 0 \f$ (potential is sphericla) \f$ j_{0}(x) = \sin(x) / x \f$ we have:
         * \f[
         *   V_{\alpha}({\bf G}) =  \int V_{\alpha}(r) 4\pi \frac{\sin(Gr)}{Gr} Y^{*}_{00} Y_{00}  r^2 \sin(\theta) dr d \phi d\theta = 
         *     4\pi \int V_{\alpha}(r) \frac{\sin(Gr)}{Gr} r^2 dr
         * \f]
         * The tricky part comes next: \f$ V_{\alpha}({\bf r}) \f$ is a long-range potential -- it decays slowly as 
         * \f$ -Z_{\alpha}^{p}/r \f$ and the straightforward integration with sperical Bessel function is numerically 
         * unstable. For \f$ {\bf G} = 0 \f$ an extra term \f$ Z_{\alpha}^p/r \f$, corresponding to the potential of 
         * pseudo-ion, is added to and removed from the local part of the atomic pseudopotential \f$ V_{\alpha}({\bf r}) \f$:
         * \f[
         *    V_{\alpha}({\bf G} = 0) = \int V_{\alpha}({\bf r})d{\bf r} \Rightarrow 
         *       4\pi \int \Big( V_{\alpha}(r) + \frac{Z_{\alpha}^p}{r} \Big) r^2 dr - 
         *       4\pi \int \Big( \frac{Z_{\alpha}^p}{r} \Big) r^2 dr 
         * \f]
         * Second term corresponds to the average electrostatic potential of ions and it is ignored 
         * (like the \f$ {\bf G} = 0 \f$ term in the Hartree potential of electrons). 
         * For \f$ G \ne 0 \f$ the following trick is done: \f$ Z_{\alpha}^p {\rm erf}(r) / r \f$ is added to and
         * removed from \f$ V_{\alpha}(r) \f$. The idea is to make potential decay quickly and then take the extra
         * contribution analytically. We have: 
         * \f[
         *    V_{\alpha}({\bf G}) = 4\pi \int \Big(V_{\alpha}(r) + Z_{\alpha}^p \frac{{\rm erf}(r)} {r} - 
         *       Z_{\alpha}^p \frac{{\rm erf}(r)}{r}\Big) \frac{\sin(Gr)}{Gr} r^2 dr
         * \f]
         * Analytical contribution from the error function is computed using the 1D Fourier transform in complex plane:
         * \f[
         *   \frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{\infty} {\rm erf}(t) e^{i\omega t} dt = 
         *     \frac{i e^{-\frac{\omega ^2}{4}} \sqrt{\frac{2}{\pi }}}{\omega }
         * \f]
         * from which we immediately get
         * \f[
         *   \int_{0}^{\infty} \frac{{\rm erf}(r)}{r} \frac{\sin(Gr)}{Gr} r^2 dr = \frac{e^{-\frac{G^2}{4}}}{G^2}
         * \f] 
         * The final expression for the local potential radial integrals for \f$ G \ne 0 \f$ take the following form:
         * \f[
         *   4\pi \int \Big(V_{\alpha}(r) r + Z_{\alpha}^p {\rm erf}(r) \Big) \frac{\sin(Gr)}{G} dr -  Z_{\alpha}^p \frac{e^{-\frac{G^2}{4}}}{G^2}
         * \f]
         */
        inline void generate_local_potential()
        {
            PROFILE("sirius::Potential::generate_local_potential");
            
            Radial_integrals_vloc ri(ctx_.unit_cell(), ctx_.pw_cutoff(), 100);
            auto v = ctx_.make_periodic_function<index_domain_t::local>([&ri](int iat, double g)
                                                                        {
                                                                            return ri.value(iat, g);
                                                                        });
            std::copy(v.begin(), v.end(), &local_potential_->f_pw_local(0));
            local_potential_->fft_transform(1);

            if (ctx_.control().print_checksum_) {
                //auto cs = local_potential_->checksum_pw();
                auto cs1 = local_potential_->checksum_rg();
                if (ctx_.comm().rank() == 0) {
                    //DUMP("checksum(local_potential_pw): %18.10f %18.10f", cs.real(), cs.imag());
                    DUMP("checksum(local_potential_rg): %18.10f", cs1);
                }
            }
        }
        
        inline void xc_mt_nonmagnetic(Radial_grid<double> const& rgrid,
                                      std::vector<XC_functional>& xc_func,
                                      Spheric_function<spectral, double> const& rho_lm,
                                      Spheric_function<spatial, double>& rho_tp,
                                      Spheric_function<spatial, double>& vxc_tp, 
                                      Spheric_function<spatial, double>& exc_tp);

        inline void xc_mt_magnetic(Radial_grid<double> const& rgrid, 
                                   std::vector<XC_functional>& xc_func,
                                   Spheric_function<spectral, double>& rho_up_lm, 
                                   Spheric_function<spatial, double>& rho_up_tp, 
                                   Spheric_function<spectral, double>& rho_dn_lm, 
                                   Spheric_function<spatial, double>& rho_dn_tp, 
                                   Spheric_function<spatial, double>& vxc_up_tp, 
                                   Spheric_function<spatial, double>& vxc_dn_tp, 
                                   Spheric_function<spatial, double>& exc_tp);

        inline void xc_mt(Periodic_function<double>* rho, 
                          Periodic_function<double>* magnetization[3], 
                          std::vector<XC_functional>& xc_func,
                          Periodic_function<double>* vxc, 
                          Periodic_function<double>* bxc[3], 
                          Periodic_function<double>* exc);
    
        inline void xc_it_nonmagnetic(Periodic_function<double>* rho, 
                                      std::vector<XC_functional>& xc_func,
                                      Periodic_function<double>* vxc, 
                                      Periodic_function<double>* exc);

        inline void xc_it_magnetic(Periodic_function<double>* rho, 
                                   Periodic_function<double>* magnetization[3], 
                                   std::vector<XC_functional>& xc_func,
                                   Periodic_function<double>* vxc, 
                                   Periodic_function<double>* bxc[3], 
                                   Periodic_function<double>* exc);

        inline void init();

    public:

        /// Constructor
        Potential(Simulation_context& ctx__)
            : ctx_(ctx__)
            , unit_cell_(ctx__.unit_cell())
            , comm_(ctx__.comm())
        {
            PROFILE("sirius::Potential::Potential");

            lmax_ = std::max(ctx_.lmax_rho(), ctx_.lmax_pot());
            sht_ = std::unique_ptr<SHT>(new SHT(lmax_));

            if (lmax_ >= 0) {
                l_by_lm_ = Utils::l_by_lm(lmax_);

                /* precompute i^l */
                zil_.resize(lmax_ + 1);
                for (int l = 0; l <= lmax_; l++) {
                    zil_[l] = std::pow(double_complex(0, 1), l);
                }
                
                zilm_.resize(Utils::lmmax(lmax_));
                for (int l = 0, lm = 0; l <= lmax_; l++) {
                    for (int m = -l; m <= l; m++, lm++) {
                        zilm_[lm] = zil_[l];
                    }
                }
            }

            effective_potential_ = std::unique_ptr<Periodic_function<double>>(new Periodic_function<double>(ctx_, ctx_.lmmax_pot()));
            
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                effective_magnetic_field_[j] = new Periodic_function<double>(ctx_, ctx_.lmmax_pot());
            }
            
            hartree_potential_ = new Periodic_function<double>(ctx_, ctx_.lmmax_pot());
            hartree_potential_->allocate_mt(false);
            
            xc_potential_ = new Periodic_function<double>(ctx_, ctx_.lmmax_pot());
            xc_potential_->allocate_mt(false);
            
            xc_energy_density_ = new Periodic_function<double>(ctx_, ctx_.lmmax_pot());
            xc_energy_density_->allocate_mt(false);

            if (!ctx_.full_potential()) {
                local_potential_ = std::unique_ptr<Periodic_function<double>>(new Periodic_function<double>(ctx_, 0));
                local_potential_->zero();

                generate_local_potential();
            }

            vh_el_ = mdarray<double, 1>(unit_cell_.num_atoms());

            if (ctx_.full_potential()) {
                gvec_ylm_ = mdarray<double_complex, 2>(ctx_.lmmax_pot(), ctx_.gvec().count(), memory_t::host, "gvec_ylm_");
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                    int ig = ctx_.gvec().offset() + igloc;
                    auto rtp = SHT::spherical_coordinates(ctx_.gvec().gvec_cart(ig));
                    SHT::spherical_harmonics(ctx_.lmax_pot(), rtp[1], rtp[2], &gvec_ylm_(0, igloc));
                }
            }

            if (ctx_.full_potential()) {
                switch (ctx_.valence_relativity()) {
                    case relativity_t::iora: {
                        rm2_inv_pw_ = mdarray<double_complex, 1>(ctx_.gvec().num_gvec());
                    }
                    case relativity_t::zora: {
                        rm_inv_pw_ = mdarray<double_complex, 1>(ctx_.gvec().num_gvec());
                    }
                    default: {
                        veff_pw_ = mdarray<double_complex, 1>(ctx_.gvec().num_gvec());
                    }
                }
            }

            init();

            /* create list of XC functionals */
            for (auto& xc_label: ctx_.xc_functionals()) {
                xc_func_.push_back(std::move(XC_functional(xc_label, ctx_.num_spins())));
            }

            /* in case of PAW */
            init_PAW();
        }

        ~Potential()
        {
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                delete effective_magnetic_field_[j];
            }
            delete hartree_potential_;
            delete xc_potential_;
            delete xc_energy_density_;
        }

        inline void set_effective_potential_ptr(double* veffmt, double* veffit)
        {
            if (ctx_.full_potential() && veffmt) {
                effective_potential_->set_mt_ptr(veffmt);
            }
            if (veffit) {
                effective_potential_->set_rg_ptr(veffit);
            }
        }

        inline void set_effective_magnetic_field_ptr(double* beffmt, double* beffit)
        {
            if (ctx_.num_mag_dims() == 0) {
                return;
            }
            assert(ctx_.num_spins() == 2);
            
            /* set temporary array wrapper */
            mdarray<double, 4> beffmt_tmp(beffmt, ctx_.lmmax_pot(), unit_cell_.max_num_mt_points(), 
                                          unit_cell_.num_atoms(), ctx_.num_mag_dims());
            mdarray<double, 2> beffit_tmp(beffit, ctx_.fft().size(), ctx_.num_mag_dims());
            
            if (ctx_.num_mag_dims() == 1) {
                /* z-component */
                if (beffmt) {
                    effective_magnetic_field_[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
                }
                if (beffit) {
                    effective_magnetic_field_[0]->set_rg_ptr(&beffit_tmp(0, 0));
                }
            }
            
            if (ctx_.num_mag_dims() == 3) {
                if (beffmt) {
                    /* z-component */
                    effective_magnetic_field_[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 2));
                    /* x-component */
                    effective_magnetic_field_[1]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
                    /* y-component */
                    effective_magnetic_field_[2]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 1));
                }
                if (beffit) {
                    /* z-component */
                    effective_magnetic_field_[0]->set_rg_ptr(&beffit_tmp(0, 2));
                    /* x-component */
                    effective_magnetic_field_[1]->set_rg_ptr(&beffit_tmp(0, 0));
                    /* y-component */
                    effective_magnetic_field_[2]->set_rg_ptr(&beffit_tmp(0, 1));
                }
            }
        }
         
        /// Zero effective potential and magnetic field.
        inline void zero()
        {
            effective_potential_->zero();
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                effective_magnetic_field_[j]->zero();
            }
        }

        /// Solve Poisson equation for a single atom.
        template <bool free_atom, typename T>
        inline std::vector<T>
        poisson_vmt(Atom const& atom__,
                    Spheric_function<function_domain_t::spectral, T> const& rho_mt__,
                    Spheric_function<function_domain_t::spectral, T>& vha_mt__) const
        {
            const bool use_r_prefact{false};

            int lmmax_rho = rho_mt__.angular_domain_size();
            int lmmax_pot = vha_mt__.angular_domain_size();
            assert((int)l_by_lm_.size() >= lmmax_rho);
            if (lmmax_rho > ctx_.lmmax_rho()) {
                std::stringstream s;
                s << "wrong angular size of rho_mt for atom of " << atom__.type().symbol() << std::endl
                  << "  lmmax_rho: " << lmmax_rho << std::endl
                  << "  ctx.lmmax_rho(): " << ctx_.lmmax_rho();
                TERMINATE(s);
            }
            std::vector<T> qmt(ctx_.lmmax_rho(), 0);

            double R = atom__.mt_radius();
            int nmtp = atom__.num_mt_points();

            #pragma omp parallel
            {
                std::vector<T> g1;
                std::vector<T> g2;

                #pragma omp for
                for (int lm = 0; lm < lmmax_rho; lm++) {
                    int l = l_by_lm_[lm];

                    auto rholm = rho_mt__.component(lm);

                    if (use_r_prefact) {
                        /* save multipole moment */
                        qmt[lm] = rholm.integrate(g1, l + 2);
                    } else {
                        for (int ir = 0; ir < nmtp; ir++) {
                            double r = atom__.radial_grid(ir);
                            rholm[ir] *= std::pow(r, l + 2);
                        }
                        qmt[lm] = rholm.interpolate().integrate(g1, 0);
                    }
                    
                    if (lm < lmmax_pot) {

                        if (use_r_prefact) {
                            rholm.integrate(g2, 1 - l);
                        } else {
                            rholm = rho_mt__.component(lm);
                            for (int ir = 0; ir < nmtp; ir++) {
                                double r = atom__.radial_grid(ir);
                                rholm[ir] *= std::pow(r, 1 - l);
                            }
                            rholm.interpolate().integrate(g2, 0);
                        }

                        double fact = fourpi / double(2 * l + 1); 
                        T vlm;

                        for (int ir = 0; ir < nmtp; ir++) {
                            double r = atom__.radial_grid(ir);

                            if (free_atom) {
                                vlm = g1[ir] / std::pow(r, l + 1) + (g2.back() - g2[ir]) * std::pow(r, l);
                            } else {
                                double d1 = 1.0 / std::pow(R, 2 * l + 1); 
                                vlm = (1.0 - std::pow(r / R, 2 * l + 1)) * g1[ir] / std::pow(r, l + 1) +
                                      (g2.back() - g2[ir]) * std::pow(r, l) - (g1.back() - g1[ir]) * std::pow(r, l) * d1;
                            }
                            vha_mt__(lm, ir) = vlm * fact;
                        }
                    }
                }
            }
            if (!free_atom) {
                /* constant part of nuclear potential -z*(1/r - 1/R) */
                for (int ir = 0; ir < nmtp; ir++) {
                    #ifdef __VHA_AUX
                    double r = atom__.radial_grid(ir);
                    vha_mt__(0, ir) += atom__.zn() * (1 / R - 1 / r) / y00;
                    #else
                    vha_mt__(0, ir) += atom__.zn() / R / y00;
                    #endif
                }
            }
            /* nuclear multipole moment */
            qmt[0] -= atom__.zn() * y00;

            return std::move(qmt);
        }

        /// Poisson solver.
        /** Detailed explanation is available in:
         *      - Weinert, M. (1981). Solution of Poisson's equation: beyond Ewald-type methods. 
         *        Journal of Mathematical Physics, 22(11), 2433–2439. doi:10.1063/1.524800
         *      - Classical Electrodynamics Third Edition by J. D. Jackson.
         *
         *  Solution of Poisson's equation for the muffin-tin geometry is carried out in several steps:
         *      - True multipole moments \f$ q_{\ell m}^{\alpha} \f$ of the muffin-tin charge density are computed.
         *      - Pseudocharge density is introduced. Pseudocharge density coincides with the true charge density 
         *        in the interstitial region and it's multipole moments inside muffin-tin spheres coincide with the 
         *        true multipole moments.
         *      - Poisson's equation for the pseudocharge density is solved in the plane-wave domain. It gives the 
         *        correct interstitial potential and correct muffin-tin boundary values.
         *      - Finally, muffin-tin part of potential is found by solving Poisson's equation in spherical coordinates
         *        with Dirichlet boundary conditions.
         *  
         *  We start by computing true multipole moments of the charge density inside the muffin-tin spheres:
         *  \f[
         *      q_{\ell m}^{\alpha} = \int Y_{\ell m}^{*}(\hat {\bf r}) r^{\ell} \rho({\bf r}) d {\bf r} = 
         *          \int \rho_{\ell m}^{\alpha}(r) r^{\ell + 2} dr
         *  \f]
         *  and for the nucleus with charge density \f$ \rho(r, \theta, \phi) = -\frac{Z \delta(r)}{4 \pi r^2} \f$:
         *  \f[
         *      q_{00}^{\alpha} = \int Y_{0 0} \frac{-Z_{\alpha} \delta(r)}{4 \pi r^2} r^2 \sin \theta dr d\phi d\theta = 
         *        -Z_{\alpha} Y_{00}
         *  \f]
         *
         *  Now we need to get the multipole moments of the interstitial charge density \f$ \rho^{I}({\bf r}) \f$ inside 
         *  muffin-tin spheres. We need this in order to estimate the amount of pseudocharge to be added to 
         *  \f$ \rho^{I}({\bf r}) \f$ to get the pseudocharge multipole moments equal to the true multipole moments. 
         *  We want to compute
         *  \f[
         *      q_{\ell m}^{I,\alpha} = \int Y_{\ell m}^{*}(\hat {\bf r}) r^{\ell} \rho^{I}({\bf r}) d {\bf r}
         *  \f]
         *  where
         *  \f[
         *      \rho^{I}({\bf r}) = \sum_{\bf G}e^{i{\bf Gr}} \rho({\bf G})
         *  \f]
         *
         *  Recall the spherical plane wave expansion:
         *  \f[
         *      e^{i{\bf G r}}=4\pi e^{i{\bf G r}_{\alpha}} \sum_{\ell m} i^\ell 
         *          j_{\ell}(G|{\bf r}-{\bf r}_{\alpha}|)
         *          Y_{\ell m}^{*}({\bf \hat G}) Y_{\ell m}(\widehat{{\bf r}-{\bf r}_{\alpha}})
         *  \f]
         *  Multipole moments of each plane-wave are computed as:
         *  \f[
         *      q_{\ell m}^{\alpha}({\bf G}) = 4 \pi e^{i{\bf G r}_{\alpha}} Y_{\ell m}^{*}({\bf \hat G}) i^{\ell}
         *          \int_{0}^{R} j_{\ell}(Gr) r^{\ell + 2} dr = 4 \pi e^{i{\bf G r}_{\alpha}} Y_{\ell m}^{*}({\bf \hat G}) i^{\ell}
         *          \left\{\begin{array}{ll} \frac{R^{\ell + 2} j_{\ell + 1}(GR)}{G} & G \ne 0 \\
         *                                   \frac{R^3}{3} \delta_{\ell 0} & G = 0 \end{array} \right.
         *  \f]
         *
         *  Final expression for the muffin-tin multipole moments of the interstitial charge denisty:
         *  \f[
         *      q_{\ell m}^{I,\alpha} = \sum_{\bf G}\rho({\bf G}) q_{\ell m}^{\alpha}({\bf G}) 
         *  \f]
         *
         *  Now we are going to modify interstitial charge density inside the muffin-tin region in order to
         *  get the true multipole moments. We will add a pseudodensity of the form:
         *  \f[
         *      P({\bf r}) = \sum_{\ell m} p_{\ell m}^{\alpha} Y_{\ell m}(\hat {\bf r}) r^{\ell} \left(1-\frac{r^2}{R^2}\right)^n
         *  \f]
         *  Radial functions of the pseudodensity are chosen in a special way. First, they produce a confined and 
         *  smooth functions inside muffin-tins and second (most important) plane-wave coefficients of the
         *  pseudodensity can be computed analytically. Let's find the relation between \f$ p_{\ell m}^{\alpha} \f$
         *  coefficients and true and interstitial multipole moments first. We are searching for the pseudodensity which restores
         *  the true multipole moments:
         *  \f[
         *      \int Y_{\ell m}^{*}(\hat {\bf r}) r^{\ell} \Big(\rho^{I}({\bf r}) + P({\bf r})\Big) d {\bf r} = q_{\ell m}^{\alpha}
         *  \f]
         *  Then 
         *  \f[
         *      p_{\ell m}^{\alpha} = \frac{q_{\ell m}^{\alpha} - q_{\ell m}^{I,\alpha}}
         *                  {\int r^{2 \ell + 2} \left(1-\frac{r^2}{R^2}\right)^n dr} = 
         *         (q_{\ell m}^{\alpha} - q_{\ell m}^{I,\alpha}) \frac{2 \Gamma(5/2 + \ell + n)}{R^{2\ell + 3}\Gamma(3/2 + \ell) \Gamma(n + 1)} 
         *  \f]
         *  
         *  Now let's find the plane-wave coefficients of \f$ P({\bf r}) \f$ inside each muffin-tin:
         *  \f[
         *      P^{\alpha}({\bf G}) = \frac{4\pi e^{-i{\bf G r}_{\alpha}}}{\Omega} \sum_{\ell m} (-i)^{\ell} Y_{\ell m}({\bf \hat G})  
         *         p_{\ell m}^{\alpha} \int_{0}^{R} j_{\ell}(G r) r^{\ell} \left(1-\frac{r^2}{R^2}\right)^n r^2 dr
         *  \f]
         *
         *  Integral of the spherical Bessel function with the radial pseudodensity component is taken analytically:
         *  \f[
         *      \int_{0}^{R} j_{\ell}(G r) r^{\ell} \left(1-\frac{r^2}{R^2}\right)^n r^2 dr = 
         *          2^n R^{\ell + 3} (GR)^{-n - 1} \Gamma(n + 1) j_{n + \ell + 1}(GR)
         *  \f]
         *
         *  The final expression for the pseudodensity plane-wave component is:
         *  \f[
         *       P^{\alpha}({\bf G}) = \frac{4\pi e^{-i{\bf G r}_{\alpha}}}{\Omega} \sum_{\ell m} (-i)^{\ell} Y_{\ell m}({\bf \hat G})  
         *          (q_{\ell m}^{\alpha} - q_{\ell m}^{I,\alpha}) \Big( \frac{2}{GR} \Big)^{n+1} 
         *          \frac{ \Gamma(5/2 + n + \ell) } {R^{\ell} \Gamma(3/2+\ell)} j_{n + \ell + 1}(GR)
         *  \f]
         *
         *  For \f$ G=0 \f$ only \f$ \ell = 0 \f$ contribution survives:
         *  \f[
         *       P^{\alpha}({\bf G}=0) = \frac{4\pi}{\Omega} Y_{00} (q_{00}^{\alpha} - q_{00}^{I,\alpha})
         *  \f]
         *
         *  We can now sum the contributions from all muffin-tin spheres and obtain a modified charge density,
         *  which is equal to the exact charge density in the interstitial region and which has correct multipole
         *  moments inside muffin-tin spheres:
         *  \f[
         *      \tilde \rho({\bf G}) = \rho({\bf G}) + \sum_{\alpha} P^{\alpha}({\bf G})
         *  \f]
         *  This density is used to solve the Poisson's equation in the plane-wave domain:
         *  \f[
         *      V_{H}({\bf G}) = \frac{4 \pi \tilde \rho({\bf G})}{G^2}
         *  \f]
         *  The potential is correct in the interstitial region and also on the muffin-tin surface. We will use
         *  it to find the boundary conditions for the potential inside the muffin-tins. Using spherical
         *  plane-wave expansion we get:
         *  \f[
         *      V^{\alpha}_{\ell m}(R) = \sum_{\bf G} V_{H}({\bf G})  
         *          4\pi e^{i{\bf G r}_{\alpha}} i^\ell 
         *          j_{\ell}^{\alpha}(GR) Y_{\ell m}^{*}({\bf \hat G}) 
         *  \f]
         *
         *  As soon as the muffin-tin boundary conditions for the potential are known, we can find the potential 
         *  inside spheres using Dirichlet Green's function:
         *  \f[
         *      V({\bf x}) = \int \rho({\bf x'})G_D({\bf x},{\bf x'}) d{\bf x'} - \frac{1}{4 \pi} \int_{S} V({\bf x'}) 
         *          \frac{\partial G_D}{\partial n'} d{\bf S'}
         *  \f]
         *  where Dirichlet Green's function for the sphere is defined as:
         *  \f[
         *      G_D({\bf x},{\bf x'}) = 4\pi \sum_{\ell m} \frac{Y_{\ell m}^{*}({\bf \hat x'}) 
         *          Y_{\ell m}(\hat {\bf x})}{2\ell + 1}
         *          \frac{r_{<}^{\ell}}{r_{>}^{\ell+1}}\Biggl(1 - \Big( \frac{r_{>}}{R} \Big)^{2\ell + 1} \Biggr)
         *  \f]
         *  and it's normal derivative at the surface is equal to:
         *  \f[
         *       \frac{\partial G_D}{\partial n'} = -\frac{4 \pi}{R^2} \sum_{\ell m} \Big( \frac{r}{R} \Big)^{\ell} 
         *          Y_{\ell m}^{*}({\bf \hat x'}) Y_{\ell m}(\hat {\bf x})
         *  \f]
         */
        inline void poisson(Periodic_function<double>* rho, Periodic_function<double>* vh);
        
        /// Generate XC potential and energy density
        /** In case of spin-unpolarized GGA the XC potential has the following expression:
         *  \f[
         *      V_{XC}({\bf r}) = \frac{\partial}{\partial \rho} \varepsilon_{xc}(\rho, \nabla \rho) - 
         *        \nabla \frac{\partial}{\partial (\nabla \rho)} \varepsilon_{xc}(\rho, \nabla \rho) 
         *  \f]
         *  LibXC packs the gradient information into the so-called \a sigma array:
         *  \f[
         *      \sigma = \nabla \rho \nabla \rho
         *  \f]
         *  Changing variables in \f$ V_{XC} \f$ expression gives:
         *  \f{eqnarray*}{
         *      V_{XC}({\bf r}) &=& \frac{\partial}{\partial \rho} \varepsilon_{xc}(\rho, \sigma) - 
         *        \nabla \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma}
         *        \frac{\partial \sigma}{ \partial (\nabla \rho)} \\
         *                      &=& \frac{\partial}{\partial \rho} \varepsilon_{xc}(\rho, \sigma) - 
         *        2 \nabla \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma} \nabla \rho - 
         *        2 \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma} \nabla^2 \rho
         *  \f}
         *  The following sequence of functions must be computed:
         *      - density on the real space grid
         *      - gradient of density (in spectral representation)
         *      - gradient of density on the real space grid
         *      - laplacian of density (in spectral representation)
         *      - laplacian of density on the real space grid
         *      - \a sigma array
         *      - a call to Libxc must be performed \a sigma derivatives must be obtained
         *      - \f$ \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma} \f$ in spectral representation
         *      - gradient of \f$ \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma} \f$ in spectral representation
         *      - gradient of \f$ \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma} \f$ on the real space grid
         *
         *  Expression for spin-polarized potential has a bit more complicated form:
         *  \f{eqnarray*}
         *      V_{XC}^{\gamma} &=& \frac{\partial \varepsilon_{xc}}{\partial \rho_{\gamma}} - \nabla
         *        \Big( 2 \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \gamma}} \nabla \rho_{\gamma} +
         *        \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \delta}} \nabla \rho_{\delta} \Big) \\
         *                      &=& \frac{\partial \varepsilon_{xc}}{\partial \rho_{\gamma}} 
         *        -2 \nabla \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \gamma}} \nabla \rho_{\gamma} 
         *        -2 \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \gamma}} \nabla^2 \rho_{\gamma} 
         *        - \nabla \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \delta}} \nabla \rho_{\delta}
         *        - \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \delta}} \nabla^2 \rho_{\delta} 
         *  \f}
         *  In magnetic case the "up" and "dn" density and potential decomposition is used. Using the fact that the
         *  effective magnetic field is parallel to magnetization at each point in space, we can write the coupling
         *  of density and magnetization with XC potential and XC magentic field as:
         *  \f[
         *      V_{xc}({\bf r}) \rho({\bf r}) + {\bf B}_{xc}({\bf r}){\bf m}({\bf r}) =
         *        V_{xc}({\bf r}) \rho({\bf r}) + {\rm B}_{xc}({\bf r}) {\rm m}({\bf r}) = 
         *        V^{\uparrow}({\bf r})\rho^{\uparrow}({\bf r}) + V^{\downarrow}({\bf r})\rho^{\downarrow}({\bf r})
         *  \f]
         *  where
         *  \f{eqnarray*}{
         *      \rho^{\uparrow}({\bf r}) &=& \frac{1}{2}\Big( \rho({\bf r}) + {\rm m}({\bf r}) \Big) \\
         *      \rho^{\downarrow}({\bf r}) &=& \frac{1}{2}\Big( \rho({\bf r}) - {\rm m}({\bf r}) \Big)
         *  \f}
         *  and
         *  \f{eqnarray*}{
         *      V^{\uparrow}({\bf r}) &=& V_{xc}({\bf r}) + {\rm B}_{xc}({\bf r}) \\
         *      V^{\downarrow}({\bf r}) &=& V_{xc}({\bf r}) - {\rm B}_{xc}({\bf r}) 
         *  \f}
         */
        void xc(Periodic_function<double>* rho,
                Periodic_function<double>* magnetization[3], 
                Periodic_function<double>* vxc,
                Periodic_function<double>* bxc[3],
                Periodic_function<double>* exc);
        
        /// Generate effective potential and magnetic field from charge density and magnetization.
        inline void generate(Density& density__)
        {
            PROFILE("sirius::Potential::generate");

            /* zero effective potential and magnetic field */
            zero();

            /* solve Poisson equation */
            poisson(density__.rho(), hartree_potential_);

            /* add Hartree potential to the total potential */
            effective_potential_->add(hartree_potential_);

            if (ctx_.full_potential()) {
                xc(density__.rho(), density__.magnetization(), xc_potential_, effective_magnetic_field_, xc_energy_density_);
            } else {
                /* add local ionic potential to the effective potential */
                effective_potential_->add(local_potential_.get());
                /* create temporary function for rho + rho_core */
                Periodic_function<double> rhovc(ctx_, 0);
                rhovc.zero();
                rhovc.add(density__.rho());
                rhovc.add(&density__.rho_pseudo_core());
                /* construct XC potentials from rho + rho_core */
                xc(&rhovc, density__.magnetization(), xc_potential_, effective_magnetic_field_, xc_energy_density_);
            }
            /* add XC potential to the effective potential */
            effective_potential_->add(xc_potential_);
    
            if (ctx_.full_potential()) {
                effective_potential_->sync_mt();
                for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                    effective_magnetic_field_[j]->sync_mt();
                }
            }

            /* get plane-wave coefficients of effective potential;
             * they will be used in three places:
             *  1) compute D-matrix
             *  2) establish a mapping between fine and coarse FFT grid for the Hloc operator 
             *  3) symmetrize effective potential */
            effective_potential_->fft_transform(-1);
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                effective_magnetic_field_[j]->fft_transform(-1);
            }

            if (!ctx_.full_potential()) {
                generate_D_operator_matrix();
                generate_PAW_effective_potential(density__);
            }
        }

        inline void save()
        {
            if (comm_.rank() == 0) {
                HDF5_tree fout(storage_file_name, false);

                effective_potential_->hdf5_write(fout["effective_potential"]);

                for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                    effective_magnetic_field_[j]->hdf5_write(fout["effective_magnetic_field"].create_node(j));
                }
            }
            comm_.barrier();
        }
        
        inline void load()
        {
            HDF5_tree fout(storage_file_name, false);
            
            effective_potential_->hdf5_read(fout["effective_potential"]);

            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                effective_magnetic_field_[j]->hdf5_read(fout["effective_magnetic_field"][j]);
            }
            
            if (ctx_.full_potential()) {
                update_atomic_potential();
            }
        }
        
        inline void update_atomic_potential()
        {
            for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
                int ia = unit_cell_.atom_symmetry_class(ic).atom_id(0);
                int nmtp = unit_cell_.atom(ia).num_mt_points();
               
                std::vector<double> veff(nmtp);
               
                for (int ir = 0; ir < nmtp; ir++) {
                    veff[ir] = y00 * effective_potential_->f_mt<index_domain_t::global>(0, ir, ia);
                }

               unit_cell_.atom_symmetry_class(ic).set_spherical_potential(veff);
            }
            
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                double* veff = &effective_potential_->f_mt<index_domain_t::global>(0, 0, ia);
                
                double* beff[] = {nullptr, nullptr, nullptr};
                for (int i = 0; i < ctx_.num_mag_dims(); i++) {
                    beff[i] = &effective_magnetic_field_[i]->f_mt<index_domain_t::global>(0, 0, ia);
                }
                
                unit_cell_.atom(ia).set_nonspherical_potential(veff, beff);
            }
        }
        
        template <device_t pu> 
        void add_mt_contribution_to_pw();

        /// Generate plane-wave coefficients of the potential in the interstitial region.
        void generate_pw_coefs();
        
        /// Calculate D operator from potential and augmentation charge.
        /** The following real symmetric matrix is computed:
         *  \f[
         *      D_{\xi \xi'}^{\alpha} = \int V({\bf r}) Q_{\xi \xi'}^{\alpha}({\bf r}) d{\bf r}
         *  \f]
         *  In the plane-wave domain this integrals transform into sum over Fourier components:
         *  \f[
         *      D_{\xi \xi'}^{\alpha} = \sum_{\bf G} \langle V |{\bf G}\rangle \langle{\bf G}|Q_{\xi \xi'}^{\alpha} \rangle = 
         *        \sum_{\bf G} V^{*}({\bf G}) e^{-i{\bf r}_{\alpha}{\bf G}} Q_{\xi \xi'}^{A}({\bf G}) = 
         *        \sum_{\bf G} Q_{\xi \xi'}^{A}({\bf G}) \tilde V_{\alpha}^{*}({\bf G})
         *  \f]
         *  where \f$ \alpha \f$ is the atom, \f$ A \f$ is the atom type and 
         *  \f[
         *      \tilde V_{\alpha}({\bf G}) = e^{i{\bf r}_{\alpha}{\bf G}} V({\bf G})  
         *  \f]
         *  Both \f$ V({\bf r}) \f$ and \f$ Q({\bf r}) \f$ functions are real and the following condition is fulfilled:
         *  \f[
         *      \tilde V_{\alpha}({\bf G}) = \tilde V_{\alpha}^{*}(-{\bf G})
         *  \f]
         *  \f[
         *      Q_{\xi \xi'}({\bf G}) = Q_{\xi \xi'}^{*}(-{\bf G})
         *  \f]
         *  In the sum over plane-wave coefficients the \f$ {\bf G} \f$ and \f$ -{\bf G} \f$ contributions will give:
         *  \f[
         *       Q_{\xi \xi'}^{A}({\bf G}) \tilde V_{\alpha}^{*}({\bf G}) + Q_{\xi \xi'}^{A}(-{\bf G}) \tilde V_{\alpha}^{*}(-{\bf G}) =
         *          2 \Re \Big( Q_{\xi \xi'}^{A}({\bf G}) \Big) \Re \Big( \tilde V_{\alpha}^{*}({\bf G}) \Big) + 
         *          2 \Im \Big( Q_{\xi \xi'}^{A}({\bf G}) \Big) \Im \Big( \tilde V_{\alpha}^{*}({\bf G}) \Big) 
         *  \f]
         *  This allows the use of a <b>dgemm</b> instead of a <b>zgemm</b> when \f$  D_{\xi \xi'}^{\alpha} \f$ matrix
         *  is calculated for all atoms of the same type.
         */
        void generate_D_operator_matrix();

        void generate_PAW_effective_potential(Density& density);

        std::vector<double> const& PAW_hartree_energies() const
        {
            return paw_hartree_energies_;
        }

        std::vector<double> const& PAW_xc_energies() const
        {
            return paw_xc_energies_;
        }

        std::vector<double> const& PAW_core_energies() const
        {
            return paw_core_energies_;
        }

        std::vector<double> const& PAW_one_elec_energies()
        {
            return paw_one_elec_energies_;
        }

        double PAW_hartree_total_energy() const
        {
            return paw_hartree_total_energy_;
        }

        double PAW_xc_total_energy() const
        {
            return paw_xc_total_energy_;
        }

        double PAW_total_core_energy() const
        {
            return paw_total_core_energy_;
        }

        double PAW_total_energy()
        {
            return paw_hartree_total_energy_ + paw_xc_total_energy_ ;
        }

        double PAW_one_elec_energy()
        {
            return paw_one_elec_energy_;
        }

        void check_potential_continuity_at_mt();

        Periodic_function<double>* effective_potential()
        {
            return effective_potential_.get();
        }

        Periodic_function<double>& local_potential()
        {
            return *local_potential_;
        }

        Spheric_function<spectral, double> const& effective_potential_mt(int ialoc) const
        {
            return effective_potential_->f_mt(ialoc);
        }

        Periodic_function<double>** effective_magnetic_field()
        {
            return effective_magnetic_field_;
        }
        
        Periodic_function<double>* effective_magnetic_field(int i)
        {
            return effective_magnetic_field_[i];
        }

        Periodic_function<double>* hartree_potential()
        {
            return hartree_potential_;
        }
        
        Spheric_function<spectral, double> const& hartree_potential_mt(int ialoc) const
        {
            return hartree_potential_->f_mt(ialoc);
        }
        
        Periodic_function<double>* xc_potential()
        {
            return xc_potential_;
        }

        Periodic_function<double>* xc_energy_density()
        {
            return xc_energy_density_;
        }
        
        void allocate()
        {
            effective_potential_->allocate_mt(true);
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                effective_magnetic_field_[j]->allocate_mt(true);
            }
        }

        inline double vh_el(int ia)
        {
            return vh_el_(ia);
        }

        inline double energy_vha()
        {
            return energy_vha_;
        }

        void mixer_input()
        {
            /* collect density and magnetization into single array */
            std::vector<Periodic_function<double>*> veff_vec(ctx_.num_mag_dims() + 1);
            veff_vec[0] = effective_potential_.get();
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                veff_vec[1 + j] = effective_magnetic_field_[j];
            }
            
            int k{0};
            
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                for (int ialoc = 0; ialoc < ctx_.unit_cell().spl_num_atoms().local_size(); ialoc++) {
                    for (int i = 0; i < static_cast<int>(veff_vec[j]->f_mt(ialoc).size()); i++) {
                        mixer_->input_local(k++, veff_vec[j]->f_mt(ialoc)[i]);
                    }
                }
                for (int i = 0; i < ctx_.fft().local_size(); i++) {
                    mixer_->input_local(k++, veff_vec[j]->f_rg(i));
                }
            }
        }

        void mixer_output()
        {
            /* collect density and magnetization into single array */
            std::vector<Periodic_function<double>*> veff_vec(ctx_.num_mag_dims() + 1);
            veff_vec[0] = effective_potential_.get();
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                veff_vec[1 + j] = effective_magnetic_field_[j];
            }
            
            int k{0};
            
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                for (int ialoc = 0; ialoc < ctx_.unit_cell().spl_num_atoms().local_size(); ialoc++) {
                    auto& f_mt = const_cast<Spheric_function<spectral, double>&>(veff_vec[j]->f_mt(ialoc));
                    for (int i = 0; i < static_cast<int>(veff_vec[j]->f_mt(ialoc).size()); i++) {
                        f_mt[i] = mixer_->output_local(k++);
                    }
                }
                for (int i = 0; i < ctx_.fft().local_size(); i++) {
                    veff_vec[j]->f_rg(i) = mixer_->output_local(k++);
                }
            }
            for (auto e: veff_vec) {
                e->sync_mt();
            }
        }

        void mixer_init()
        {
            int sz{0};
            for (int ialoc = 0; ialoc < ctx_.unit_cell().spl_num_atoms().local_size(); ialoc++) {
                sz += static_cast<int>(effective_potential_->f_mt(ialoc).size());
            }
            sz += ctx_.fft().local_size();

            mixer_ = Mixer_factory<double>(ctx_.mixer_input().type_, 0, (ctx_.num_mag_dims() + 1) * sz, ctx_.mixer_input(), comm_);
            mixer_input();
            mixer_->initialize();
        }

        double mix()
        {
            mixer_input();
            double rms = mixer_->mix();
            mixer_output();
            return rms;
        }

        double_complex const& veff_pw(int ig__) const
        {
            return veff_pw_(ig__);
        }

        inline void set_veff_pw(double_complex* veff_pw__)
        {
            std::copy(veff_pw__, veff_pw__ + ctx_.gvec().num_gvec(), veff_pw_.at<CPU>());
        }

        double_complex const& rm_inv_pw(int ig__) const
        {
            return rm_inv_pw_(ig__);
        }

        inline void set_rm_inv_pw(double_complex* rm_inv_pw__)
        {
            std::copy(rm_inv_pw__, rm_inv_pw__ + ctx_.gvec().num_gvec(), rm_inv_pw_.at<CPU>());
        }

        double_complex const& rm2_inv_pw(int ig__) const
        {
            return rm2_inv_pw_(ig__);
        }

        inline void set_rm2_inv_pw(double_complex* rm2_inv_pw__)
        {
            std::copy(rm2_inv_pw__, rm2_inv_pw__ + ctx_.gvec().num_gvec(), rm2_inv_pw_.at<CPU>());
        }

        inline void fft_transform(int direction__)
        {
            effective_potential_->fft_transform(direction__);
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                effective_magnetic_field_[j]->fft_transform(direction__);
            }
        }
};

#include "Potential/init.hpp"
#include "Potential/generate_d_operator_matrix.hpp"
#include "Potential/generate_pw_coefs.hpp"
#include "Potential/xc.hpp"
#include "Potential/poisson.hpp"
#include "Potential/paw_potential.hpp"

};

#endif // __POTENTIAL_H__

