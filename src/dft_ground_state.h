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

/** \file dft_ground_state.h
 *
 *  \brief Contains definition and partial implementation of sirius::DFT_ground_state class.
 */

#ifndef __DFT_GROUND_STATE_H__
#define __DFT_GROUND_STATE_H__

#include "potential.h"
#include "density.h"
#include "k_set.h"
#include "force.h"

namespace sirius
{

class DFT_ground_state
{
    private:

        Simulation_context& ctx_;

        Simulation_parameters const& parameters_;

        Unit_cell& unit_cell_;

        Potential* potential_;

        Density* density_;

        K_set* kset_;

        int use_symmetry_;

        double ewald_energy_;

        double ewald_energy()
        {
            Timer t("sirius::DFT_ground_state::ewald_energy");

            double alpha = 1.5;
            
            double ewald_g = 0;

            auto rl = ctx_.reciprocal_lattice();

            splindex<block> spl_num_gvec(ctx_.gvec().num_gvec(), ctx_.comm().size(), ctx_.comm().rank());

            #pragma omp parallel
            {
                double ewald_g_pt = 0;

                #pragma omp for
                for (int igloc = 0; igloc < (int)spl_num_gvec.local_size(); igloc++)
                {
                    int ig = (int)spl_num_gvec[igloc];

                    double_complex rho(0, 0);
                    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
                    {
                        rho += rl->gvec_phase_factor(ig, ia) * double(unit_cell_.atom(ia)->zn());
                    }
                    double g2 = std::pow(ctx_.gvec().shell_len(ctx_.gvec().shell(ig)), 2);
                    if (ig)
                    {
                        ewald_g_pt += std::pow(std::abs(rho), 2) * std::exp(-g2 / 4 / alpha) / g2;
                    }
                    else
                    {
                        ewald_g_pt -= std::pow(unit_cell_.num_electrons(), 2) / alpha / 4; // constant term in QE comments
                    }
                }

                #pragma omp critical
                ewald_g += ewald_g_pt;
            }
            ctx_.comm().allreduce(&ewald_g, 1);
            ewald_g *= (twopi / unit_cell_.omega());

            /* remove self-interaction */
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
            {
                ewald_g -= sqrt(alpha / pi) * std::pow(unit_cell_.atom(ia)->zn(), 2);
            }

            double ewald_r = 0;
            #pragma omp parallel
            {
                double ewald_r_pt = 0;

                #pragma omp for
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
                {
                    for (int i = 1; i < unit_cell_.num_nearest_neighbours(ia); i++)
                    {
                        int ja = unit_cell_.nearest_neighbour(i, ia).atom_id;
                        double d = unit_cell_.nearest_neighbour(i, ia).distance;
                        ewald_r_pt += 0.5 * unit_cell_.atom(ia)->zn() * 
                                            unit_cell_.atom(ja)->zn() * gsl_sf_erfc(sqrt(alpha) * d) / d;
                    }
                }

                #pragma omp critical
                ewald_r += ewald_r_pt;
            }

            return (ewald_g + ewald_r);
        }

    public:

        DFT_ground_state(Simulation_context& ctx__,
                         Potential* potential__,
                         Density* density__,
                         K_set* kset__,
                         int use_symmetry__)
            : ctx_(ctx__),
              parameters_(ctx__.parameters()),
              unit_cell_(ctx__.unit_cell()),
              potential_(potential__), 
              density_(density__), 
              kset_(kset__),
              use_symmetry_(use_symmetry__)
        {
            if (parameters_.esm_type() == ultrasoft_pseudopotential) ewald_energy_ = ewald_energy();
        }

        void move_atoms(int istep);

        void forces(mdarray<double, 2>& atom_force);

        void scf_loop(double potential_tol, double energy_tol, int num_dft_iter);

        void relax_atom_positions();

        void print_info();

        /// Return nucleus energy in the electrostatic field.
        /** Compute energy of nucleus in the electrostatic potential generated by the total (electrons + nuclei) 
         *  charge density. Diverging self-interaction term z*z/|r=0| is excluded. */
        double energy_enuc();
        
        /// Return eigen-value sum of core states.
        double core_eval_sum();
        
        double energy_vha()
        {
            //return inner(parameters_, density_->rho(), potential_->hartree_potential());
            return potential_->energy_vha();
        }
        
        double energy_vxc()
        {
            return Periodic_function<double>::inner(density_->rho(), potential_->xc_potential());
        }
        
        double energy_exc()
        {
            double exc = Periodic_function<double>::inner(density_->rho(), potential_->xc_energy_density());
            if (parameters_.esm_type() == ultrasoft_pseudopotential) 
                exc += Periodic_function<double>::inner(density_->rho_pseudo_core(), potential_->xc_energy_density());
            return exc;
        }

        double energy_bxc()
        {
            double ebxc = 0.0;
            for (int j = 0; j < parameters_.num_mag_dims(); j++) 
                ebxc += Periodic_function<double>::inner(density_->magnetization(j), potential_->effective_magnetic_field(j));
            return ebxc;
        }

        double energy_veff()
        {
            //return inner(parameters_, density_->rho(), potential_->effective_potential());
            return energy_vha() + energy_vxc();
        }

        /// Full eigen-value sum (core + valence)
        double eval_sum()
        {
            return (core_eval_sum() + kset_->valence_eval_sum());
        }
        
        /// Kinetic energy
        /** more doc here
        */
        double energy_kin()
        {
            return (eval_sum() - energy_veff() - energy_bxc());
        }

        double energy_ewald() const
        {
            return ewald_energy_;
        }

        /// Total energy of the electronic subsystem.
        /** From the definition of the density functional we have:
         *  
         *  \f[
         *      E[\rho] = T[\rho] + E^{H}[\rho] + E^{XC}[\rho] + E^{ext}[\rho]
         *  \f]
         *  where \f$ T[\rho] \f$ is the kinetic energy, \f$ E^{H}[\rho] \f$ - electrostatic energy of
         *  electron-electron density interaction, \f$ E^{XC}[\rho] \f$ - exchange-correlation energy
         *  and \f$ E^{ext}[\rho] \f$ - energy in the external field of nuclei.
         *  
         *  Electrostatic and external field energies are grouped in the following way:
         *  \f[
         *      \frac{1}{2} \int \int \frac{\rho({\bf r})\rho({\bf r'}) d{\bf r} d{\bf r'}}{|{\bf r} - {\bf r'}|} + 
         *          \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r} = \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} + 
         *          \frac{1}{2} \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r}
         *  \f]
         *  Here \f$ V^{H}({\bf r}) \f$ is the total (electron + nuclei) electrostatic potential returned by the 
         *  poisson solver. Next we transform the remaining term:
         *  \f[
         *      \frac{1}{2} \int \rho({\bf r}) V^{nuc}({\bf r}) d{\bf r} = 
         *      \frac{1}{2} \int \int \frac{\rho({\bf r})\rho^{nuc}({\bf r'}) d{\bf r} d{\bf r'}}{|{\bf r} - {\bf r'}|} = 
         *      \frac{1}{2} \int V^{H,el}({\bf r}) \rho^{nuc}({\bf r}) d{\bf r}
         *  \f]
         */
        double total_energy()
        {
            switch (parameters_.esm_type())
            {
                case full_potential_lapwlo:
                case full_potential_pwlo:
                {
                    return (energy_kin() + energy_exc() + 0.5 * energy_vha() + energy_enuc());
                }
                case ultrasoft_pseudopotential:
                case norm_conserving_pseudopotential:
                {
                    return (kset_->valence_eval_sum() - (energy_vxc() + energy_vha()) + 0.5 * energy_vha() + 
                            energy_exc() + ewald_energy_);
                }
                default:
                {
                    STOP();
                }
            }
            return 0; // make compiler happy
        }

        void generate_effective_potential()
        {
            switch(parameters_.esm_type())
            {
                case full_potential_lapwlo:
                case full_potential_pwlo:
                {
                    potential_->generate_effective_potential(density_->rho(), density_->magnetization());
                    break;
                }
                case ultrasoft_pseudopotential:
                case norm_conserving_pseudopotential:
                {
                    potential_->generate_effective_potential(density_->rho(), density_->rho_pseudo_core(), density_->magnetization());
                    break;
                }
            }
        }

        void symmetrize_density()
        {
            PROFILE();

            auto& comm = ctx_.comm();

            if (parameters_.full_potential())
            {
                for (int j = 0; j < parameters_.num_mag_dims(); j++)
                    density_->magnetization(j)->fft_transform(-1);
            }

            unit_cell_.symmetry()->symmetrize_function(&density_->rho()->f_pw(0), ctx_.gvec(), comm);

            if (parameters_.full_potential())
                unit_cell_.symmetry()->symmetrize_function(density_->rho()->f_mt(), comm);

            if (parameters_.num_mag_dims() == 1)
            {
                unit_cell_.symmetry()->symmetrize_vector_z_component(&density_->magnetization(0)->f_pw(0), ctx_.gvec(), comm);
                unit_cell_.symmetry()->symmetrize_vector_z_component(density_->magnetization(0)->f_mt(), comm);
            }
            if (parameters_.num_mag_dims() == 3)
            {
                unit_cell_.symmetry()->symmetrize_vector(&density_->magnetization(1)->f_pw(0),
                                                         &density_->magnetization(2)->f_pw(0), 
                                                         &density_->magnetization(0)->f_pw(0),
                                                         ctx_.gvec(), comm);
                unit_cell_.symmetry()->symmetrize_vector(density_->magnetization(1)->f_mt(),
                                                         density_->magnetization(2)->f_mt(),
                                                         density_->magnetization(0)->f_mt(),
                                                         comm);
            }

            if (parameters_.esm_type() == full_potential_lapwlo || parameters_.esm_type() == full_potential_pwlo)
            {
                density_->rho()->fft_transform(1);
                for (int j = 0; j < parameters_.num_mag_dims(); j++)
                    density_->magnetization(j)->fft_transform(1);
            }

            #ifdef __PRINT_OBJECT_HASH
            DUMP("hash(rhomt): %16llX", density_->rho()->f_mt().hash());
            DUMP("hash(rhoit): %16llX", density_->rho()->f_it().hash());
            DUMP("hash(rhopw): %16llX", density_->rho()->f_pw().hash());
            #endif
        }
};

};

#endif // __DFT_GROUND_STATE_H__

/** \page DFT Spin-polarized DFT
 *  \section section1 Preliminary notes
 *
 *  \note Here and below sybol \f$ {\boldsymbol \sigma} \f$ is reserved for the vector of Pauli matrices. Spin components 
 *        are labeled with \f$ \alpha \f$ or \f$ \beta\f$.
 *
 *  Wave-function of spin-1/2 particle is a two-component spinor:
 *  \f[
 *      {\bf \varphi}({\bf r})=\left( \begin{array}{c} \varphi_1({\bf r}) \\ \varphi_2({\bf r}) \end{array} \right)
 *  \f]
 *  Operator of spin:
 *  \f[
 *      {\bf \hat S}=\frac{\hbar}{2}{\bf \sigma},
 *  \f]
 *  Pauli matrices:
 *  \f[
 *      \sigma_x=\left( \begin{array}{cc}
 *         0 & 1 \\
 *         1 & 0 \\ \end{array} \right) \,
 *           \sigma_y=\left( \begin{array}{cc}
 *         0 & -i \\
 *         i & 0 \\ \end{array} \right) \,
 *           \sigma_z=\left( \begin{array}{cc}
 *         1 & 0 \\
 *         0 & -1 \\ \end{array} \right)
 *  \f]
 *
 *  \section section2 Density and magnetization
 *  Density is defined as:
 *  \f[
 *      \rho({\bf r}) = \sum_{j}^{occ} \Psi_{j}^{*}({\bf r}){\bf I} \Psi_{j}({\bf r}) = 
          \sum_{j}^{occ} \psi_{j}^{\uparrow *} \psi_{j}^{\uparrow} + \psi_{j}^{\downarrow *} \psi_{j}^{\downarrow} 

 *  \f]
 *  Magnetization is defined as:
 *  \f[
 *      {\bf m}({\bf r}) = \sum_{j}^{occ} \Psi_{j}^{*}({\bf r}) {\boldsymbol \sigma} \Psi_{j}({\bf r})
 *  \f]
 *  \f[
 *      m_x({\bf r}) = \sum_{j}^{occ} \psi_{j}^{\uparrow *} \psi_{j}^{\downarrow} + \psi_{j}^{\downarrow *} \psi_{j}^{\uparrow} 
 *  \f]
 *  \f[
 *      m_y({\bf r}) = \sum_{j}^{occ} -i \psi_{j}^{\uparrow *} \psi_{j}^{\downarrow} + i \psi_{j}^{\downarrow *} \psi_{j}^{\uparrow} 
 *  \f]
 *  \f[
 *      m_z({\bf r}) = \sum_{j}^{occ} \psi_{j}^{\uparrow *} \psi_{j}^{\uparrow} - \psi_{j}^{\downarrow *} \psi_{j}^{\downarrow} 
 *  \f]
 *  Density matrix is defined as:
 *  \f[
 *      {\boldsymbol \rho}({\bf r}) = \frac{1}{2} \Big( {\bf I}\rho({\bf r}) + {\boldsymbol \sigma} {\bf m}({\bf r})\Big) = 
 *        \frac{1}{2} \sum_{j}^{occ} \left( \begin{array}{cc} \psi_{j}^{\uparrow *} \psi_{j}^{\uparrow} & 
 *                                                            \psi_{j}^{\downarrow *} \psi_{j}^{\uparrow} \\
 *                                                            \psi_{j}^{\uparrow *} \psi_{j}^{\downarrow} &
 *                                                            \psi_{j}^{\downarrow *} \psi_{j}^{\downarrow} \end{array} \right)
 *  \f]
 *  Pay attention to the order of spin indices in the \f$ 2 \times 2 \f$ density matrix:
 *  \f[
 *    \rho_{\alpha \beta}({\bf r}) = \frac{1}{2} \sum_{j}^{occ} \psi_{j}^{\beta *}({\bf r})\psi_{j}^{\alpha}({\bf r})
 *  \f]
 */
