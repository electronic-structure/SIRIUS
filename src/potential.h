// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

        Step_function const* step_function_;

        Communicator const& comm_;

        Periodic_function<double>* effective_potential_;

        Periodic_function<double>* effective_magnetic_field_[3];
 
        Periodic_function<double>* hartree_potential_;
        Periodic_function<double>* xc_potential_;
        Periodic_function<double>* xc_energy_density_;
        
        /// Local part of pseudopotential.
        Periodic_function<double>* local_potential_;

        mdarray<double, 3> sbessel_mom_;

        mdarray<double, 3> sbessel_mt_;

        mdarray<double, 2> gamma_factors_R_;

        int lmax_;
        
        std::unique_ptr<SHT> sht_;

        int pseudo_density_order;

        std::vector<double_complex> zil_;
        
        std::vector<double_complex> zilm_;

        std::vector<int> l_by_lm_;

        mdarray<double_complex, 2> gvec_ylm_;

        double energy_vha_;
        
        /// Electronic part of Hartree potential.
        /** Used to compute electron-nuclear contribution to the total energy */
        mdarray<double, 1> vh_el_;

        Mixer<double>* mixer_;

        std::vector<XC_functional*> xc_func_;

        // Store form factors because it is needed by forces calculation
        mdarray<double, 2> vloc_radial_integrals_;

        //------------------------------------------------
        //--- PAW parameters and functions -----------------------
        //------------------------------------------------
        std::vector< mdarray<double,3> > ae_paw_local_potential_;
        std::vector< mdarray<double,3> > ps_paw_local_potential_;


        // because one need to call allreduce for a whole matrix
        mdarray<double_complex,4>  paw_dij_;

        std::vector< double > paw_hartree_energies_;
        std::vector< double > paw_xc_energies_;
        std::vector< double > paw_core_energies_;
        std::vector< double > paw_one_elec_energies_;

        double paw_hartree_total_energy_;
        double paw_xc_total_energy_;
        double paw_total_core_energy_;
        double paw_one_elec_energy_;

        //--- PAW  functions ---
        void init_PAW();

        double xc_mt_PAW_nonmagnetic(Radial_grid const& rgrid,
                                     mdarray<double, 3> &out_atom_pot,
                                     mdarray<double, 2> &full_rho_lm,
                                     const std::vector<double> &rho_core);


        double xc_mt_PAW_collinear(Radial_grid const& rgrid,
                                   mdarray<double,3> &out_atom_pot,
                                   mdarray<double,2> &full_rho_lm,
                                   mdarray<double,3> &magnetization_lm,
                                   const std::vector<double> &rho_core);

        // TODO DO
        void xc_mt_PAW_noncollinear(    )   {     };

        void calc_PAW_local_potential(int spl_atom_index,
                                      mdarray<double, 2> &ae_full_density,
                                      mdarray<double, 2> &ps_full_density,
                                      mdarray<double, 3> &ae_local_magnetization,
                                      mdarray<double, 3> &ps_local_magnetization);



        void calc_PAW_local_Dij(int spl_atom_index, mdarray<double_complex,4>& paw_dij);

        double calc_PAW_hartree_potential(Atom& atom, const Radial_grid& grid,
                                             mdarray<double, 2> &full_density,
                                             mdarray<double, 3> &out_atom_pot);

        double calc_PAW_one_elec_energy(int atom_index,
                                        const mdarray<double_complex,4>& density_matrix,
                                        const mdarray<double_complex,4>& paw_dij);


        void add_paw_Dij_to_atom_Dmtrx();

        //------------------------------------------------
        //------------------------------------------------
        //Spheric_function<function_domain_t::spectral,double>
        /// Compute MT part of the potential and MT multipole moments
        void poisson_vmt(Periodic_function<double>* rho__, 
                         Periodic_function<double>* vh__,
                         mdarray<double_complex, 2>& qmt__);

        /// Compute MT part of the potential and MT multipole moments
        void poisson_atom_vmt(Spheric_function<function_domain_t::spectral,double> &rho_mt,
                        Spheric_function<function_domain_t::spectral,double> &vh_mt,
                        mdarray<double_complex, 1>& qmt_mt,
                        Atom &atom);

        /// Perform a G-vector summation of plane-wave coefficiens multiplied by radial integrals.
        void poisson_sum_G(int lmmax__, 
                           double_complex* fpw__, 
                           mdarray<double, 3>& fl__, 
                           mdarray<double_complex, 2>& flm__);
        
        /// Add contribution from the pseudocharge to the plane-wave expansion
        void poisson_add_pseudo_pw(mdarray<double_complex, 2>& qmt, mdarray<double_complex, 2>& qit, double_complex* rho_pw);

        void generate_local_potential();
        
        void xc_mt_nonmagnetic(Radial_grid const& rgrid,
                               std::vector<XC_functional*>& xc_func,
                               Spheric_function<spectral, double> const& rho_lm,
                               Spheric_function<spatial, double>& rho_tp,
                               Spheric_function<spatial, double>& vxc_tp, 
                               Spheric_function<spatial, double>& exc_tp);

        void xc_mt_magnetic(Radial_grid const& rgrid, 
                            std::vector<XC_functional*>& xc_func,
                            Spheric_function<spectral, double>& rho_up_lm, 
                            Spheric_function<spatial, double>& rho_up_tp, 
                            Spheric_function<spectral, double>& rho_dn_lm, 
                            Spheric_function<spatial, double>& rho_dn_tp, 
                            Spheric_function<spatial, double>& vxc_up_tp, 
                            Spheric_function<spatial, double>& vxc_dn_tp, 
                            Spheric_function<spatial, double>& exc_tp);

        void xc_mt(Periodic_function<double>* rho, 
                   Periodic_function<double>* magnetization[3], 
                   std::vector<XC_functional*>& xc_func,
                   Periodic_function<double>* vxc, 
                   Periodic_function<double>* bxc[3], 
                   Periodic_function<double>* exc);
    
        void xc_it_nonmagnetic(Periodic_function<double>* rho, 
                               std::vector<XC_functional*>& xc_func,
                               Periodic_function<double>* vxc, 
                               Periodic_function<double>* exc);

        void xc_it_magnetic(Periodic_function<double>* rho, 
                            Periodic_function<double>* magnetization[3], 
                            std::vector<XC_functional*>& xc_func,
                            Periodic_function<double>* vxc, 
                            Periodic_function<double>* bxc[3], 
                            Periodic_function<double>* exc);

        void init();




    public:

        /// Constructor
        Potential(Simulation_context& ctx__);

        ~Potential();

        void set_effective_potential_ptr(double* veffmt, double* veffit);
        
        void set_effective_magnetic_field_ptr(double* beffmt, double* beffit);
         
        /// Zero effective potential and magnetic field.
        void zero();

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
         *          \frac{ \Gamma(5/2 + n + \ell) } {R^{\ell} \Gamma(3/2+\ell)}
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
        void poisson(Periodic_function<double>* rho, Periodic_function<double>* vh);
        
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
        void xc(Periodic_function<double>* rho, Periodic_function<double>* magnetization[3], 
                Periodic_function<double>* vxc, Periodic_function<double>* bxc[3], Periodic_function<double>* exc);
        
        /// Generate effective potential and magnetic field from charge density and magnetization.
        void generate_effective_potential(Periodic_function<double>* rho, Periodic_function<double>* magnetization[3]);
        
        void generate_effective_potential(Periodic_function<double>* rho, Periodic_function<double>* rho_core, 
                                          Periodic_function<double>* magnetization[3]);

        void save();
        
        void load();
        
        void update_atomic_potential();
        
        template <device_t pu> 
        void add_mt_contribution_to_pw();

        /// Generate plane-wave coefficients of the potential in the interstitial region
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

        //------------------------------------------------
        //--- PAW functions ------------------------------
        //------------------------------------------------
        void generate_PAW_effective_potential(Density& density);

        const std::vector< double >&  PAW_hartree_energies(){ return paw_hartree_energies_; }

        const std::vector< double >&  PAW_xc_energies(){ return paw_xc_energies_; }

        const std::vector< double >&  PAW_core_energies(){ return paw_core_energies_; }

        const std::vector< double >&  PAW_one_elec_energies(){ return paw_one_elec_energies_; }

        double PAW_hartree_total_energy(){ return paw_hartree_total_energy_; }

        double PAW_xc_total_energy(){ return paw_xc_total_energy_; }

        double PAW_total_core_energy(){ return paw_total_core_energy_; }

        double PAW_total_energy(){ return paw_hartree_total_energy_ + paw_xc_total_energy_ ; }

        double PAW_one_elec_energy(){ return paw_one_elec_energy_; }

        //------------------------------------------------
        //--- For forces ------------------------------
        //------------------------------------------------

        mdarray<double, 2> const& get_vloc_radial_integrals(){ return vloc_radial_integrals_; }

        //------------------------------------------------
        //--- next ------------------------------
        //------------------------------------------------

        void check_potential_continuity_at_mt();

        //void copy_to_global_ptr(double* fmt, double* fit, Periodic_function<double>* src);
       
        /// Total size (number of elements) of the potential and effective magnetic field. 
        inline size_t size()
        {
            size_t s = effective_potential_->size();
            for (int i = 0; i < ctx_.num_mag_dims(); i++) s += effective_magnetic_field_[i]->size();
            return s;
        }

        inline void pack(Mixer<double>* mixer)
        {
            size_t n = effective_potential_->pack(0, mixer);
            for (int i = 0; i < ctx_.num_mag_dims(); i++) n += effective_magnetic_field_[i]->pack(n, mixer);
        }

        inline void unpack(double const* buffer)
        {
            size_t n = effective_potential_->unpack(buffer);
            for (int i = 0; i < ctx_.num_mag_dims(); i++) n += effective_magnetic_field_[i]->unpack(&buffer[n]);
        }

        //void copy_xc_potential(double* vxcmt, double* vxcir);

        //void copy_effective_magnetic_field(double* beffmt, double* beffit);
        
        Periodic_function<double>* effective_potential()
        {
            return effective_potential_;
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
            for (int j = 0; j < ctx_.num_mag_dims(); j++) effective_magnetic_field_[j]->allocate_mt(true);
        }

        inline double vh_el(int ia)
        {
            return vh_el_(ia);
        }

        inline double energy_vha()
        {
            return energy_vha_;
        }

        void mixer_init()
        {
            /* create mixer */
            if (ctx_.mixer_input_section().type_ == "linear")
            {
                mixer_ = new Linear_mixer<double>(size(), ctx_.mixer_input_section().beta_, comm_);
            }
            else if (ctx_.mixer_input_section().type_ == "broyden1")
            {
                std::vector<double> weights;
                mixer_ = new Broyden1<double>(size(),
                                              ctx_.mixer_input_section().max_history_,
                                              ctx_.mixer_input_section().beta_,
                                              weights,
                                              comm_);
            }
            else if (ctx_.mixer_input_section().type_ == "broyden2")
            {
                std::vector<double> weights;
                mixer_ = new Broyden2<double>(size(),
                                              ctx_.mixer_input_section().max_history_,
                                              ctx_.mixer_input_section().beta_,
                                              ctx_.mixer_input_section().beta0_,
                                              ctx_.mixer_input_section().linear_mix_rms_tol_,
                                              weights,
                                              comm_);
            }
            else
            {
                TERMINATE("wrong mixer type");
            }
            pack(mixer_);
            mixer_->initialize();
        }

        double mix()
        {
            pack(mixer_);
            double rms = mixer_->mix();
            unpack(mixer_->output_buffer());
            return rms;
        }

        //inline void checksum()
        //{
        //    DUMP("checksum(veff): %18.10f %18.10f\n", effective_potential_->f_mt().checksum(), effective_potential_->f_it().checksum());
        //}
};

#include "Potential/init.hpp"
#include "Potential/generate_d_operator_matrix.hpp"
#include "Potential/generate_pw_coefs.hpp"

};

#endif // __POTENTIAL_H__

