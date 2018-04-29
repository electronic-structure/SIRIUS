// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file free_atom.hpp
 *
 *  \brief Free atom full-potential solver.
 */

#ifndef __FREE_ATOM_HPP__
#define __FREE_ATOM_HPP__

#include "atom_type.h"

namespace sirius {

/// Full potential free atom solver.
class Free_atom : public Atom_type
{
  private:
    /// Densities of individual orbitals.
    mdarray<double, 2> free_atom_orbital_density_;
    /// Radial wave-functions.
    mdarray<double, 2> free_atom_wave_functions_;
    /// Atomic potential.
    Spline<double> free_atom_potential_;
    /// NIST total energy for LDA calculation.
    double NIST_LDA_Etot_{0};
    /// NIST total energy for scalar-relativistic LDA calculation.
    double NIST_ScRLDA_Etot_{0};
    /// Energies of atomic levels
    std::vector<double> enu_;

  public:

    Free_atom(Free_atom&& src) = default;
    
    /// Constructor.
    Free_atom(Simulation_parameters const& param__,
              std::string                  symbol__)
        : Atom_type(param__, symbol__, atomic_name[atomic_zn.at(symbol__) - 1], atomic_zn.at(symbol__), 0.0,
                    atomic_conf[atomic_zn.at(symbol__) - 1], radial_grid_t::lin_exp_grid)
        , NIST_LDA_Etot_(atomic_energy_NIST_LDA[atomic_zn.at(symbol__) - 1])
    {
        radial_grid_ = Radial_grid_exp<double>(2000 + 150 * zn(), 1e-7, 20.0 + 0.25 * zn());
    }

    /// Constructor.
    Free_atom(Simulation_parameters const& param__,
              int                          zn__)
        : Atom_type(param__, atomic_symb[zn__ - 1], atomic_name[zn__ - 1], zn__, 0.0,
                    atomic_conf[zn__ - 1], radial_grid_t::lin_exp_grid)
        , NIST_LDA_Etot_(atomic_energy_NIST_LDA[zn__ - 1])
    {
        radial_grid_ = Radial_grid_exp<double>(2000 + 150 * zn(), 1e-7, 20.0 + 0.25 * zn());
    }

    json ground_state(double energy_tol, double charge_tol, bool rel)
    {
        PROFILE("sirius::Free_atom::ground_state");

        int np = radial_grid().num_points();
        assert(np > 0);

        free_atom_orbital_density_ = mdarray<double, 2>(np, num_atomic_levels());
        free_atom_wave_functions_  = mdarray<double, 2>(np, num_atomic_levels());

        XC_functional Ex("XC_LDA_X", 1);
        XC_functional Ec("XC_LDA_C_VWN", 1);
        Ex.set_relativistic(rel);

        std::vector<double> veff(np);
        std::vector<double> vrho(np);
        std::vector<double> vnuc(np);
        for (int i = 0; i < np; i++) {
            vnuc[i] = -zn() * radial_grid().x_inv(i);
            veff[i] = vnuc[i];
            vrho[i] = 0;
        }

        Mixer<double>* mixer = new Broyden1<double>(0, np, 12, 0.8, mpi_comm_self());
        for (int i = 0; i < np; i++) {
            mixer->input_local(i, vrho[i]);
        }
        mixer->initialize();

        Spline<double> rho(radial_grid());

        Spline<double> f(radial_grid());

        std::vector<double> vh(np);
        std::vector<double> vxc(np);
        std::vector<double> exc(np);
        std::vector<double> vx(np);
        std::vector<double> vc(np);
        std::vector<double> ex(np);
        std::vector<double> ec(np);
        std::vector<double> g1;
        std::vector<double> g2;
        std::vector<double> rho_old;

        enu_.resize(num_atomic_levels());

        double energy_tot = 0.0;
        double energy_tot_old;
        double charge_rms;
        double energy_diff;
        double energy_enuc = 0;
        double energy_xc   = 0;
        double energy_kin  = 0;
        double energy_coul = 0;

        /* starting values for E_{nu} */
        for (int ist = 0; ist < num_atomic_levels(); ist++) {
            enu_[ist] = -1.0 * zn() / 2 / std::pow(double(atomic_level(ist).n), 2);
        }

        int num_iter{-1};

        for (int iter = 0; iter < 200; iter++) {
            rho_old = rho.values();

            std::memset(&rho(0), 0, rho.num_points() * sizeof(double));
            #pragma omp parallel default(shared)
            {
                std::vector<double> rho_t(rho.num_points());
                std::memset(&rho_t[0], 0, rho.num_points() * sizeof(double));

                #pragma omp for
                for (int ist = 0; ist < num_atomic_levels(); ist++) {
                    // relativity_t rt = (rel) ? relativity_t::koelling_harmon : relativity_t::none;
                    relativity_t rt = (rel) ? relativity_t::dirac : relativity_t::none;
                    Bound_state bound_state(rt, zn(), atomic_level(ist).n, atomic_level(ist).l, atomic_level(ist).k,
                                            radial_grid(), veff, enu_[ist]);
                    enu_[ist]    = bound_state.enu();
                    auto& bs_rho = bound_state.rho();
                    auto& bs_u   = bound_state.u();

                    /* assume a spherical symmetry */
                    for (int i = 0; i < np; i++) {
                        free_atom_orbital_density_(i, ist) = bs_rho(i);
                        free_atom_wave_functions_(i, ist)  = bs_u(i);
                        /* sum of squares of spherical harmonics for angular momentm l is (2l+1)/4pi */
                        rho_t[i] += atomic_level(ist).occupancy * free_atom_orbital_density_(i, ist) / fourpi;
                    }
                }

                #pragma omp critical
                for (int i = 0; i < rho.num_points(); i++) {
                    rho(i) += rho_t[i];
                }
            }

            charge_rms = 0.0;
            for (int i = 0; i < np; i++) {
                charge_rms += std::pow(rho(i) - rho_old[i], 2);
            }
            charge_rms = sqrt(charge_rms / np);

            rho.interpolate();

            /* compute Hartree potential */
            rho.integrate(g2, 2);
            double t1 = rho.integrate(g1, 1);

            for (int i = 0; i < np; i++) {
                vh[i] = fourpi * (g2[i] / radial_grid(i) + t1 - g1[i]);
            }

            /* compute XC potential and energy */
            Ex.get_lda(rho.num_points(), &rho(0), &vx[0], &ex[0]);
            Ec.get_lda(rho.num_points(), &rho(0), &vc[0], &ec[0]);
            for (int ir = 0; ir < rho.num_points(); ir++) {
                vxc[ir] = (vx[ir] + vc[ir]);
                exc[ir] = (ex[ir] + ec[ir]);
            }

            /* mix old and new effective potential */
            for (int i = 0; i < np; i++) {
                mixer->input_local(i, vh[i] + vxc[i]);
            }
            mixer->mix(1e-16);
            for (int i = 0; i < np; i++) {
                vrho[i] = mixer->output_local(i);
                veff[i] = vrho[i] + vnuc[i];
            }

            //= /* mix old and new effective potential */
            //= for (int i = 0; i < np; i++)
            //=     veff[i] = (1 - beta) * veff[i] + beta * (vnuc[i] + vh[i] + vxc[i]);

            /* sum of occupied eigen values */
            double eval_sum = 0.0;
            for (int ist = 0; ist < num_atomic_levels(); ist++) {
                eval_sum += atomic_level(ist).occupancy * enu_[ist];
            }

            for (int i = 0; i < np; i++) {
                f(i) = (veff[i] - vnuc[i]) * rho(i);
            }
            /* kinetic energy */
            energy_kin = eval_sum - fourpi * (f.interpolate().integrate(2) - zn() * rho.integrate(1));

            /* XC energy */
            for (int i = 0; i < np; i++) {
                f(i) = exc[i] * rho(i);
            }
            energy_xc = fourpi * f.interpolate().integrate(2);

            /* electron-nuclear energy: \int vnuc(r) * rho(r) r^2 dr */
            energy_enuc = -fourpi * zn() * rho.integrate(1);

            /* Coulomb energy */
            for (int i = 0; i < np; i++) {
                f(i) = vh[i] * rho(i);
            }
            energy_coul = 0.5 * fourpi * f.interpolate().integrate(2);

            energy_tot_old = energy_tot;

            energy_tot = energy_kin + energy_xc + energy_coul + energy_enuc;

            energy_diff = std::abs(energy_tot - energy_tot_old);

            if (energy_diff < energy_tol && charge_rms < charge_tol) {
                num_iter = iter;
                break;
            }
        }

        json dict;
        if (num_iter >= 0) {
            dict["converged"] = true;
            dict["num_scf_iterations"] = num_iter;
        } else {
            dict["converged"] = false;
        } 
        dict["energy_diff"] = energy_diff;
        dict["charge_rms"] = charge_rms;
        dict["energy_tot"] = energy_tot;

        free_atom_density_spline_ = Spline<double>(radial_grid_, rho.values());

        free_atom_potential_ = Spline<double>(radial_grid_, vrho);

        return std::move(dict);

        //double Eref = (rel) ? NIST_ScRLDA_Etot_ : NIST_LDA_Etot_;

        //printf("\n");
        //printf("Radial gird\n");
        //printf("-----------\n");
        //printf("type             : %s\n", radial_grid().name().c_str());
        //printf("number of points : %i\n", np);
        //printf("origin           : %20.12f\n", radial_grid(0));
        //printf("infinity         : %20.12f\n", radial_grid(np - 1));
        //printf("\n");
        //printf("Energy\n");
        //printf("------\n");
        //printf("Ekin  : %20.12f\n", energy_kin);
        //printf("Ecoul : %20.12f\n", energy_coul);
        //printf("Eenuc : %20.12f\n", energy_enuc);
        //printf("Eexc  : %20.12f\n", energy_xc);
        //printf("Total : %20.12f\n", energy_tot);
        //printf("NIST  : %20.12f\n", Eref);

        ///* difference between NIST and computed total energy. Comparison is valid only for VWN XC functional. */
        //double dE = (Utils::round(energy_tot, 6) - Eref);
        //std::cerr << zn() << " " << dE << " # " << symbol() << std::endl;

        //return energy_tot;
    }

    inline double free_atom_orbital_density(int ir, int ist)
    {
        return free_atom_orbital_density_(ir, ist);
    }

    inline double free_atom_wave_function(int ir, int ist)
    {
        return free_atom_wave_functions_(ir, ist);
    }

    inline double free_atom_potential(double x)
    {
        return free_atom_potential_.at_point(x);
    }
};
} // namespace sirius

#endif
