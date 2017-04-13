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
#include "k_point_set.h"
#include "force.h"
#include "json.hpp"
#include "Geometry/Forces_PS.h"
#include "Geometry/stress.h"

using json = nlohmann::json;

namespace sirius
{

class DFT_ground_state
{
    private:

        Simulation_context& ctx_;

        Unit_cell& unit_cell_;

        Potential& potential_;

        Density& density_;

        K_point_set& kset_;

        Band band_;

        //std::unique_ptr<Forces_PS> forces_;

        int use_symmetry_;

        double ewald_energy_{0};
        
        /// Compute the ion-ion electrostatic energy using Ewald method.
        /** The following contribution (per unit cell) to the total energy has to be computed:
         *  \f[
         *    E^{ion-ion} = \frac{1}{N} \frac{1}{2} \sum_{i \neq j} \frac{Z_i Z_j}{|{\bf r}_i - {\bf r}_j|} = 
         *      \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} \frac{Z_{\alpha} Z_{\beta}}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|}
         *  \f]
         *  where \f$ N \f$ is the number of unit cells in the crystal.
         *  Following the idea of Ewald the Coulomb interaction is split into two terms:
         *  \f[
         *     \frac{1}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|} = 
         *       \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|} + 
         *       \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|}
         *  \f]
         *  Second term is computed directly. First term is computed in the reciprocal space. Remembering that
         *  \f[
         *    \frac{1}{\Omega} \sum_{\bf G} e^{i{\bf Gr}} = \sum_{\bf T} \delta({\bf r - T})
         *  \f]
         *  we rewrite the first term as
         *  \f[
         *    \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} 
         *      \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|} = 
         *    \frac{1}{2} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta}  \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|} -
         *      \frac{1}{2} \sum_{\alpha} Z_{\alpha}^2 2 \sqrt{\frac{\lambda}{\pi}} = \\
         *    \frac{1}{2} \sum_{\alpha \beta} Z_{\alpha} Z_{\beta} \frac{1}{\Omega} \sum_{\bf G} \int e^{i{\bf Gr}} \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}_{\alpha\beta} + {\bf r}|)}{|{\bf r}_{\alpha\beta} + {\bf r}|} d{\bf r}  -
         *      \sum_{\alpha} Z_{\alpha}^2  \sqrt{\frac{\lambda}{\pi}}
         *  \f]
         *  The integral is computed using the \f$ \ell=0 \f$ term of the spherical expansion of the plane-wave:
         *  \f[
         *    \int e^{i{\bf Gr}} \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}_{\alpha\beta} + {\bf r}|)}{|{\bf r}_{\alpha\beta} + {\bf r}|} d{\bf r} = 
         *      \int e^{-i{\bf r}_{\alpha \beta}{\bf G}} e^{i{\bf Gr}} \frac{{\rm erf}(\sqrt{\lambda} |{\bf r}|)}{|{\bf r}|} d{\bf r} = 
         *    e^{-i{\bf r}_{\alpha \beta}{\bf G}} 4 \pi \int_0^{\infty} \frac{\sin({G r})}{G} {\rm erf}(\sqrt{\lambda} r ) dr 
         *  \f]
         *  We will split integral in two parts:
         *  \f[
         *    \int_0^{\infty} \sin({G r}) {\rm erf}(\sqrt{\lambda} r ) dr = \int_0^{b} \sin({G r}) {\rm erf}(\sqrt{\lambda} r ) dr + 
         *      \int_b^{\infty} \sin({G r}) dr = \frac{1}{G} e^{-\frac{G^2}{4 \lambda}}
         *  \f]
         *  where \f$ b \f$ is sufficiently large. To reproduce in Mathrmatica:
            \verbatim
            Limit[Limit[
              Integrate[Sin[g*x]*Erf[Sqrt[nu] * x], {x, 0, b},
                  Assumptions -> {nu > 0, g >= 0, b > 0}] +
                     Integrate[Sin[g*(x + I*a)], {x, b, \[Infinity]},
                         Assumptions -> {a > 0, nu > 0, g >= 0, b > 0}], a -> 0],
                          b -> \[Infinity], Assumptions -> {nu > 0, g >= 0}]
            \endverbatim
         *  The first term of the Ewald sum thus becomes:
         *  \f[
         *    \frac{2 \pi}{\Omega} \sum_{{\bf G}} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 -
         *       \sum_{\alpha} Z_{\alpha}^2 \sqrt{\frac{\lambda}{\pi}}
         *  \f]
         *  For \f$ G=0 \f$ the following is done:
         *  \f[
         *    \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \approx \frac{1}{G^2}-\frac{1}{4 \lambda }
         *  \f]
         *  The term \f$ \frac{1}{G^2} \f$ is compensated together with the corresponding Hartree terms in electron-electron and
         *  electron-ion interactions (cell should be neutral) and we are left with the following conribution:
         *  \f[
         *    -\frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda}
         *  \f]
         *  Final expression for the Ewald energy:
         *  \f[
         *    E^{ion-ion} = \frac{1}{2} \sideset{}{'} \sum_{\alpha \beta {\bf T}} Z_{\alpha} Z_{\beta} 
         *      \frac{{\rm erfc}(\sqrt{\lambda} |{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|)}{|{\bf r}_{\alpha} - {\bf r}_{\beta} + {\bf T}|} +
         *      \frac{2 \pi}{\Omega} \sum_{{\bf G}\neq 0} \frac{e^{-\frac{G^2}{4 \lambda}}}{G^2} \Big| \sum_{\alpha} Z_{\alpha} e^{-i{\bf r}_{\alpha}{\bf G}} \Big|^2 -
         *      \sum_{\alpha} Z_{\alpha}^2 \sqrt{\frac{\lambda}{\pi}} - \frac{2\pi}{\Omega}\frac{N_{el}^2}{4 \lambda}
         *  \f]
         */
        double ewald_energy();

    public:

        DFT_ground_state(Simulation_context& ctx__,
                         Potential& potential__,
                         Density& density__,
                         K_point_set& kset__,
                         int use_symmetry__)
            : ctx_(ctx__)
            , unit_cell_(ctx__.unit_cell())
            , potential_(potential__)
            , density_(density__)
            , kset_(kset__)
            , band_(ctx_)
            , use_symmetry_(use_symmetry__)
        {
            if (!ctx_.full_potential()) {
                ewald_energy_ = ewald_energy();
            }

            //forces_ = std::unique_ptr<Forces_PS>(new Forces_PS(ctx_, density_, potential_, kset_));
        }

        mdarray<double, 2> forces();

        void forces(mdarray<double, 2>& inout_forces);

        int find(double potential_tol, double energy_tol, int num_dft_iter, bool write_state);

        void print_info();
        
        /// Return nucleus energy in the electrostatic field.
        /** Compute energy of nucleus in the electrostatic potential generated by the total (electrons + nuclei) 
         *  charge density. Diverging self-interaction term z*z/|r=0| is excluded. */
        double energy_enuc() const
        {
            double enuc{0};
            if (ctx_.full_potential()) {
                for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
                    int ia = unit_cell_.spl_num_atoms(ialoc);
                    int zn = unit_cell_.atom(ia).zn();
                    enuc -= 0.5 * zn * potential_.vh_el(ia); // * y00;
                }
                ctx_.comm().allreduce(&enuc, 1);
            }
            return enuc;
        }
        
        /// Return eigen-value sum of core states.
        double core_eval_sum() const
        {
            double sum{0};
            for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
                sum += unit_cell_.atom_symmetry_class(ic).core_eval_sum() * 
                       unit_cell_.atom_symmetry_class(ic).num_atoms();
            }
            return sum;
        }

        double energy_vha()
        {
            return potential_.energy_vha();
        }
        
        double energy_vxc()
        {
            return density_.rho()->inner(potential_.xc_potential());
        }
        
        double energy_exc()
        {
            double exc = density_.rho()->inner(potential_.xc_energy_density());
            if (!ctx_.full_potential()) {
                exc += density_.rho_pseudo_core()->inner(potential_.xc_energy_density());
            }
            return exc;
        }

        double energy_bxc()
        {
            double ebxc{0};
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                ebxc += density_.magnetization(j)->inner(potential_.effective_magnetic_field(j));
            }
            return ebxc;
        }

        double energy_veff()
        {
            return density_.rho()->inner(potential_.effective_potential());
        }

        double energy_vloc()
        {
            return density_.rho()->inner(&potential_.local_potential());
        }

        /// Full eigen-value sum (core + valence)
        double eval_sum()
        {
            return (core_eval_sum() + kset_.valence_eval_sum());
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
         *
         *  Total energy in PW-PP method has the following expression:
         *  \f[
         *    E_{tot} = \sum_{i} f_i \langle \psi_i | \Big( \hat T + \sum_{\xi \xi'} |\beta_{\xi} \rangle D_{\xi \xi'}^{ion} \langle \beta_{\xi'} |\Big) | \psi_i \rangle +
         *     \int V^{ion}({\bf r})\rho({\bf r})d{\bf r} + \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} + E^{XC}[\rho + \rho_{core}]
         *  \f]
         *  The following rearrangement is performed:
         *  \f[
         *     \int V^{ion}({\bf r})\rho({\bf r})d{\bf r} + \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} = \\
         *     \int V^{ion}({\bf r})\rho({\bf r})d{\bf r} + \int V^{H}({\bf r})\rho({\bf r})d{\bf r} + \int V^{XC}({\bf r})\rho({\bf r})d{\bf r} - 
         *     \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} - \int V^{XC}({\bf r})\rho({\bf r})d{\bf r}  = \\
         *      \int V^{eff}({\bf r})\rho({\bf r})d{\bf r} - \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} - \int V^{XC}({\bf r})\rho({\bf r})d{\bf r} = \\
         *      \int V^{eff}({\bf r})\rho^{ps}({\bf r})d{\bf r} + \int V^{eff}({\bf r})\rho^{aug}({\bf r})d{\bf r} - \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} - \int V^{XC}({\bf r})\rho({\bf r})d{\bf r}
         *  \f]
         *  Where
         *  \f[
         *    V^{eff}({\bf r}) = V^{H}({\bf r}) + V^{XC}({\bf r}) + V^{ion}({\bf r})
         *  \f]
         *  Next, we use the definition of \f$ \rho^{aug} \f$:
         *  \f[
         *    \int V^{eff}({\bf r})\rho^{aug}({\bf r})d{\bf r} = \int V^{eff}({\bf r}) \sum_{i}\sum_{\xi \xi'} f_i \langle \psi_i | \beta_{\xi} \rangle Q_{\xi \xi'}({\bf r}) \langle \beta_{\xi'} | \psi_i \rangle d{\bf r} = 
         *     \sum_{i}\sum_{\xi \xi'} f_i \langle \psi_i | \beta_{\xi} \rangle D_{\xi \xi'}^{aug} \langle \beta_{\xi'} | \psi_i \rangle
         *  \f]
         *  Now we can rewrite the total energy expression:
         *  \f[
         *    E_{tot} = \sum_{i} f_i \langle \psi_i | \Big( \hat T + \sum_{\xi \xi'} |\beta_{\xi} \rangle D_{\xi \xi'}^{ion} + D_{\xi \xi'}^{aug} \langle \beta_{\xi'} |\Big) | \psi_i \rangle +
         *      \int V^{eff}({\bf r})\rho^{ps}({\bf r})d{\bf r} - \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} - \int V^{XC}({\bf r})\rho({\bf r}) d{\bf r} + E^{XC}[\rho + \rho_{core}]
         *  \f]
         *  From the Kohn-Sham equations
         *  \f[
         *    \Big( \hat T + \sum_{\xi \xi'} |\beta_{\xi} \rangle D_{\xi \xi'}^{ion} + D_{\xi \xi'}^{aug} \langle \beta_{\xi'}| + \hat V^{eff} \Big) | \psi_i \rangle = \varepsilon_i \Big( 1+\hat S \Big) |\psi_i \rangle 
         *  \f]
         *  we can extract the sum
         *  \f[
         *    \sum_{i} f_i \langle \psi_i | \Big( \hat T + \sum_{\xi \xi'} |\beta_{\xi} \rangle D_{\xi \xi'}^{ion} + D_{\xi \xi'}^{aug} \langle \beta_{\xi'} |\Big) | \psi_i \rangle  = 
         *     \sum_{i} f_i \varepsilon_i - \int V^{eff}({\bf r}) \rho^{ps}({\bf r}) d{\bf r}
         *  \f]
         *  Combining all together we obtain the following expression for total energy:
         *  \f[
         *    E_{tot} = \sum_{i} f_i \varepsilon_i - \frac{1}{2} \int V^{H}({\bf r})\rho({\bf r})d{\bf r} - \int V^{XC}({\bf r})\rho({\bf r}) d{\bf r} + E^{XC}[\rho + \rho_{core}]
         *  \f]
         */
        double total_energy()
        {
            double tot_en{0};

            switch (ctx_.esm_type()) {
                case electronic_structure_method_t::full_potential_lapwlo: {
                    tot_en = (energy_kin() + energy_exc() + 0.5 * energy_vha() + energy_enuc());
                    break;
                }

                case electronic_structure_method_t::pseudopotential: {
                    //tot_en = (kset_.valence_eval_sum() - energy_veff() + energy_vloc() - potential_.PAW_one_elec_energy()) +
                    //         0.5 * energy_vha() + energy_exc() + potential_.PAW_total_energy() + ewald_energy_;
                    tot_en = (kset_.valence_eval_sum() - energy_vxc() - potential_.PAW_one_elec_energy()) -
                             0.5 * energy_vha() + energy_exc() + potential_.PAW_total_energy() + ewald_energy_;
                    break;
                }
            }

            return tot_en;
        }

        void symmetrize(Periodic_function<double>* f__,
                        Periodic_function<double>* gz__,
                        Periodic_function<double>* gx__,
                        Periodic_function<double>* gy__)
        {
            PROFILE("sirius::DFT_ground_state::symmetrize");

            auto& comm = ctx_.comm();

            /* symmetrize PW components */
            unit_cell_.symmetry().symmetrize_function(&f__->f_pw(0), ctx_.gvec(), comm);
            switch (ctx_.num_mag_dims()) {
                case 1: {
                    unit_cell_.symmetry().symmetrize_vector_function(&gz__->f_pw(0), ctx_.gvec(), comm);
                    break;
                }
                case 3: {
                    unit_cell_.symmetry().symmetrize_vector_function(&gx__->f_pw(0), &gy__->f_pw(0), &gz__->f_pw(0),
                                                                     ctx_.gvec(), comm);
                    break;
                }
            }

            if (ctx_.full_potential()) {
                /* symmetrize MT components */
                unit_cell_.symmetry().symmetrize_function(f__->f_mt(), comm);
                switch (ctx_.num_mag_dims()) {
                    case 1: {
                        unit_cell_.symmetry().symmetrize_vector_function(gz__->f_mt(), comm);
                        break;
                    }
                    case 3: {
                        unit_cell_.symmetry().symmetrize_vector_function(gx__->f_mt(), gy__->f_mt(), gz__->f_mt(), comm);
                        break;
                    }
                }
            }
        }

        Band const& band() const
        {
            return band_;
        }

        json serialize()
        {
            json dict;

            dict["mpi_grid"] = ctx_.mpi_grid_dims();

            std::vector<int> fftgrid(3);
            for (int i = 0; i < 3; i++) {
                fftgrid[i] = ctx_.fft().grid().size(i);
            }
            dict["fft_grid"] = fftgrid;
            if (!ctx_.full_potential()) {
                for (int i = 0; i < 3; i++) {
                    fftgrid[i] = ctx_.fft_coarse().grid().size(i);
                }
                dict["fft_coarse_grid"] = fftgrid;
            }
            dict["num_fv_states"] = ctx_.num_fv_states();
            dict["num_bands"] = ctx_.num_bands();
            dict["aw_cutoff"] = ctx_.aw_cutoff();
            dict["pw_cutoff"] = ctx_.pw_cutoff();
            dict["omega"] = ctx_.unit_cell().omega();
            dict["chemical_formula"] = ctx_.unit_cell().chemical_formula();
            dict["num_atoms"] = ctx_.unit_cell().num_atoms();
            dict["energy"] = json::object();
            dict["energy"]["total"] = total_energy();
            dict["energy"]["enuc"] = energy_enuc();
            dict["energy"]["core_eval_sum"] = core_eval_sum();
            dict["energy"]["vha"] = energy_vha();
            dict["energy"]["vxc"] = energy_vxc();
            dict["energy"]["exc"] = energy_exc();
            dict["energy"]["bxc"] = energy_bxc();
            dict["energy"]["veff"] = energy_veff();
            dict["energy"]["eval_sum"] = eval_sum();
            dict["energy"]["kin"] = energy_kin();
            dict["energy"]["ewald"] = energy_ewald();
            dict["efermi"] = kset_.energy_fermi();
            dict["band_gap"] = kset_.band_gap();
            dict["core_leakage"] = density_.core_leakage();

            return std::move(dict);
        }
};

inline double DFT_ground_state::ewald_energy()
{
    PROFILE("sirius::DFT_ground_state::ewald_energy");

    double alpha = 1.5;
    
    double ewald_g = 0;

    #pragma omp parallel
    {
        double ewald_g_pt = 0;

        #pragma omp for
        for (int igloc = 0; igloc < ctx_.gvec_count(); igloc++) {
            int ig = ctx_.gvec_offset() + igloc;
            if (!ig) {
                continue;
            }

            double g2 = std::pow(ctx_.gvec().gvec_len(ig), 2);

            double_complex rho(0, 0);

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                rho += ctx_.gvec_phase_factor(ig, ia) * static_cast<double>(unit_cell_.atom(ia).zn());
            }

            ewald_g_pt += std::pow(std::abs(rho), 2) * std::exp(-g2 / 4 / alpha) / g2;
        }

        #pragma omp critical
        ewald_g += ewald_g_pt;
    }
    ctx_.comm().allreduce(&ewald_g, 1);
    if (ctx_.gvec().reduced()) {
        ewald_g *= 2;
    }
    /* remaining G=0 contribution */
    ewald_g -= std::pow(unit_cell_.num_electrons(), 2) / alpha / 4; 
    ewald_g *= (twopi / unit_cell_.omega());

    /* remove self-interaction */
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        ewald_g -= std::sqrt(alpha / pi) * std::pow(unit_cell_.atom(ia).zn(), 2);
    }

    double ewald_r = 0;
    #pragma omp parallel
    {
        double ewald_r_pt = 0;

        #pragma omp for
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            for (int i = 1; i < unit_cell_.num_nearest_neighbours(ia); i++) {
                int ja = unit_cell_.nearest_neighbour(i, ia).atom_id;
                double d = unit_cell_.nearest_neighbour(i, ia).distance;
                ewald_r_pt += 0.5 * unit_cell_.atom(ia).zn() * unit_cell_.atom(ja).zn() *
                              gsl_sf_erfc(std::sqrt(alpha) * d) / d;
            }
        }

        #pragma omp critical
        ewald_r += ewald_r_pt;
    }

    return (ewald_g + ewald_r);
}

inline void DFT_ground_state::forces(mdarray<double, 2>& inout_forces)
{
    PROFILE("sirius::DFT_ground_state::forces");

    //forces_->calc_forces_contributions();

    //forces_->sum_forces(inout_forces);

    //if(ctx_.comm().rank() == 0)
    //{
    //    auto print_forces=[&](mdarray<double, 2> const& forces)
    //    {
    //        for(int ia=0; ia < unit_cell_.num_atoms(); ia++)
    //        {
    //            printf("Atom %4i    force = %15.7f  %15.7f  %15.7f \n",
    //                   unit_cell_.atom(ia).type_id(), forces(0,ia), forces(1,ia), forces(2,ia));
    //        }
    //    };

    //    std::cout<<"===== Total Forces in Ha/bohr =====" << std::endl;
    //    print_forces( inout_forces );

    //    std::cout<<"===== Forces: ultrasoft contribution from Qij =====" << std::endl;
    //    print_forces( forces_->ultrasoft_forces() );

    //    std::cout<<"===== Forces: non-local contribution from Beta-projectors =====" << std::endl;
    //    print_forces( forces_->nonlocal_forces() );

    //    std::cout<<"===== Forces: local contribution from local potential=====" << std::endl;
    //    print_forces( forces_->local_forces() );

    //    std::cout<<"===== Forces: nlcc contribution from core density=====" << std::endl;
    //    print_forces( forces_->nlcc_forces() );

    //    std::cout<<"===== Forces: Ewald forces from ions =====" << std::endl;
    //    print_forces( forces_->ewald_forces() );
    //}
}


inline mdarray<double,2 > DFT_ground_state::forces()
{
    PROFILE("sirius::DFT_ground_state::forces");

    mdarray<double,2 > tot_forces(3, unit_cell_.num_atoms());

    forces(tot_forces);

    return std::move(tot_forces);
}

inline int DFT_ground_state::find(double potential_tol, double energy_tol, int num_dft_iter, bool write_state)
{
    PROFILE("sirius::DFT_ground_state::scf_loop");
    
    double eold{0}, rms{0};

    if (ctx_.full_potential()) {
        potential_.mixer_init();
    } else {
        density_.mixer_init();
    }

    int result{-1};

//    tbb::task_scheduler_init tbb_init(omp_get_num_threads());

    for (int iter = 0; iter < num_dft_iter; iter++) {
        sddk::timer t1("sirius::DFT_ground_state::scf_loop|iteration");

        /* find new wave-functions */
        band_.solve_for_kset(kset_, potential_, true);
        /* find band occupancies */
        kset_.find_band_occupancies();
        /* generate new density from the occupied wave-functions */
        density_.generate(kset_);
        /* symmetrize density and magnetization */
        if (use_symmetry_) {
            symmetrize(density_.rho(), density_.magnetization(0), density_.magnetization(1),
                       density_.magnetization(2));
        }
        /* set new tolerance of iterative solver */
        if (!ctx_.full_potential()) {
            rms = density_.mix();
            double tol = std::max(1e-12, 0.1 * density_.dr2() / ctx_.unit_cell().num_valence_electrons());
            if (ctx_.comm().rank() == 0) {
                printf("dr2: %18.10f, tol: %18.10f\n",  density_.dr2(), tol);
            }
            ctx_.set_iterative_solver_tolerance(std::min(ctx_.iterative_solver_tolerance(), tol));
        }

        if (!ctx_.full_potential()) {
            density_.generate_paw_loc_density();
        }
        /* transform density to realspace after mixing and symmetrization */
        density_.fft_transform(1);
        /* check number of elctrons */
        density_.check_num_electrons();

        //== if (ctx_.num_mag_dims())
        //== {
        //==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        //==     {
        //==         vector3d<double> mag(0, 0, 0);

        //==         for (int j0 = 0; j0 < ctx_.fft().grid().size(0); j0++)
        //==         {
        //==             for (int j1 = 0; j1 < ctx_.fft().grid().size(1); j1++)
        //==             {
        //==                 for (int j2 = 0; j2 < ctx_.fft().local_size_z(); j2++)
        //==                 {
        //==                     /* get real space fractional coordinate */
        //==                     auto v0 = vector3d<double>(double(j0) / ctx_.fft().grid().size(0), 
        //==                                                double(j1) / ctx_.fft().grid().size(1), 
        //==                                                double(ctx_.fft().offset_z() + j2) / ctx_.fft().grid().size(2));
        //==                     /* index of real space point */
        //==                     int ir = ctx_.fft().grid().index_by_coord(j0, j1, j2);

        //==                     for (int t0 = -1; t0 <= 1; t0++)
        //==                     {
        //==                         for (int t1 = -1; t1 <= 1; t1++)
        //==                         {
        //==                             for (int t2 = -1; t2 <= 1; t2++)
        //==                             {
        //==                                 vector3d<double> v1 = v0 - (unit_cell_.atom(ia).position() + vector3d<double>(t0, t1, t2));
        //==                                 auto r = unit_cell_.get_cartesian_coordinates(v1);
        //==                                 auto a = r.length();

        //==                                 if (a <= 2.0)
        //==                                 {
        //==                                     mag[2] += density_.magnetization(0)->f_rg(ir);
        //==                                 }
        //==                             }
        //==                         }
        //==                     }
        //==                 }
        //==             }
        //==         }
        //==         for (int x: {0, 1, 2}) mag[x] *= (unit_cell_.omega() / ctx_.fft().size());
        //==         printf("atom: %i, mag: %f %f %f\n", ia, mag[0], mag[1], mag[2]);
        //==     }
        //== }

        /* compute new potential */
        potential_.generate(density_);

        /* symmetrize potential and effective magnetic field */
        if (use_symmetry_) {
            symmetrize(potential_.effective_potential(), potential_.effective_magnetic_field(0),
                       potential_.effective_magnetic_field(1), potential_.effective_magnetic_field(2));
        }
        /* transform potential to real space after symmetrization */
        potential_.fft_transform(1);

        /* compute new total energy for a new density */
        double etot = total_energy();
        
        if (ctx_.full_potential()) {
            rms = potential_.mix();
            double tol = std::max(1e-12, rms);
            if (ctx_.comm().rank() == 0) {
                printf("tol: %18.10f\n", tol);
            }
            ctx_.set_iterative_solver_tolerance(std::min(ctx_.iterative_solver_tolerance(), tol));
        }

        /* write some information */
        print_info();

        if (ctx_.comm().rank() == 0) {
            printf("iteration : %3i, RMS %18.12f, energy difference : %12.6f\n", iter, rms, etot - eold);
        }
        
        if (std::abs(eold - etot) < energy_tol && rms < potential_tol) {
            result = iter;
            break;
        }

        eold = etot;
    }
    
    if (write_state) {
        ctx_.create_storage_file();
        potential_.save();
        density_.save();
    }

//    tbb_init.terminate();

    return result;
}

inline void DFT_ground_state::print_info()
{
    double evalsum1 = kset_.valence_eval_sum();
    double evalsum2 = core_eval_sum();
    double ekin = energy_kin();
    double evxc = energy_vxc();
    double eexc = energy_exc();
    double ebxc = energy_bxc();
    double evha = energy_vha();
    double etot = total_energy();
    double gap = kset_.band_gap() * ha2ev;
    double ef = kset_.energy_fermi();
    double core_leak = density_.core_leakage();
    double enuc = energy_enuc();

    double one_elec_en = evalsum1 - (evxc + evha);

    if (ctx_.esm_type() == electronic_structure_method_t::pseudopotential) {
        one_elec_en -= potential_.PAW_one_elec_energy();
    }

    std::vector<double> mt_charge;
    double it_charge;
    double total_charge = density_.rho()->integrate(mt_charge, it_charge); 
    
    double total_mag[3];
    std::vector<double> mt_mag[3];
    double it_mag[3];
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        total_mag[j] = density_.magnetization(j)->integrate(mt_mag[j], it_mag[j]);
    }
    
    if (ctx_.comm().rank() == 0) {
        if (ctx_.full_potential()) {
            double total_core_leakage = 0.0;
            printf("\n");
            printf("Charges and magnetic moments\n");
            for (int i = 0; i < 80; i++) printf("-");
            printf("\n"); 
            printf("atom      charge    core leakage");
            if (ctx_.num_mag_dims()) printf("              moment                |moment|");
            printf("\n");
            for (int i = 0; i < 80; i++) printf("-");
            printf("\n"); 

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                double core_leakage = unit_cell_.atom(ia).symmetry_class().core_leakage();
                total_core_leakage += core_leakage;
                printf("%4i  %10.6f  %10.8e", ia, mt_charge[ia], core_leakage);
                if (ctx_.num_mag_dims()) {
                    vector3d<double> v;
                    v[2] = mt_mag[0][ia];
                    if (ctx_.num_mag_dims() == 3) {
                        v[0] = mt_mag[1][ia];
                        v[1] = mt_mag[2][ia];
                    }
                    printf("  [%8.4f, %8.4f, %8.4f]  %10.6f", v[0], v[1], v[2], v.length());
                }
                printf("\n");
            }
            
            printf("\n");
            printf("interstitial charge   : %10.6f\n", it_charge);
            if (ctx_.num_mag_dims()) {
                vector3d<double> v;
                v[2] = it_mag[0];
                if (ctx_.num_mag_dims() == 3) {
                    v[0] = it_mag[1];
                    v[1] = it_mag[2];
                }
                printf("interstitial moment   : [%8.4f, %8.4f, %8.4f], magnitude : %10.6f\n", 
                       v[0], v[1], v[2], v.length());
            }
            
            printf("\n");
            printf("total charge          : %10.6f\n", total_charge);
            printf("total core leakage    : %10.8e\n", total_core_leakage);
            if (ctx_.num_mag_dims()) {
                vector3d<double> v;
                v[2] = total_mag[0];
                if (ctx_.num_mag_dims() == 3) {
                    v[0] = total_mag[1];
                    v[1] = total_mag[2];
                }
                printf("total moment          : [%8.4f, %8.4f, %8.4f], magnitude : %10.6f\n", 
                       v[0], v[1], v[2], v.length());
            }
        }
        printf("\n");
        printf("Energy\n");
        for (int i = 0; i < 80; i++) printf("-");
        printf("\n"); 

        printf("valence_eval_sum          : %18.8f\n", evalsum1);
        if (ctx_.full_potential()) {
            printf("core_eval_sum             : %18.8f\n", evalsum2);
            printf("kinetic energy            : %18.8f\n", ekin);
            printf("enuc                      : %18.8f\n", enuc);
        }
        printf("<rho|V^{XC}>              : %18.8f\n", evxc);
        printf("<rho|E^{XC}>              : %18.8f\n", eexc);
        printf("<mag|B^{XC}>              : %18.8f\n", ebxc);
        printf("<rho|V^{H}>               : %18.8f\n", evha);
        if (!ctx_.full_potential()) {
            printf("one-electron contribution : %18.8f (Ha), %18.8f (Ry)\n", one_elec_en, one_elec_en * 2); // eband + deband in QE
            printf("hartree contribution      : %18.8f\n", 0.5 * evha);
            printf("xc contribution           : %18.8f\n", eexc);
            printf("ewald contribution        : %18.8f\n", ewald_energy_);
            printf("PAW contribution          : %18.8f\n", potential_.PAW_total_energy());
        }
        printf("Total energy              : %18.8f (Ha), %18.8f (Ry)\n", etot, etot * 2);

        printf("\n");
        printf("band gap (eV) : %18.8f\n", gap);
        printf("Efermi        : %18.8f\n", ef);
        printf("\n");
        if (ctx_.full_potential()) {
            printf("core leakage : %18.8f\n", core_leak);
        }
    }
}

} // namespace

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
 *         \sum_{j}^{occ} \psi_{j}^{\uparrow *} \psi_{j}^{\uparrow} + \psi_{j}^{\downarrow *} \psi_{j}^{\downarrow} 
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

