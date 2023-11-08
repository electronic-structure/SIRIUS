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

#include "mixer/anderson_mixer.hpp"
#include "atom_type_base.hpp"
#include "radial/radial_solver.hpp"
#include "potential/xc_functional.hpp"

namespace sirius {

// TODO: pass grid parameters or set a good default

/// Full potential free atom solver.
class Free_atom : public Atom_type_base
{
  private:
    /// Densities of individual orbitals.
    mdarray<double, 2> free_atom_orbital_density_;
    /// Radial wave-functions.
    mdarray<double, 2> free_atom_wave_functions_;
    /// Radial wave-functions multiplied by r.
    mdarray<double, 2> free_atom_wave_functions_x_;
    /// Derivatieve of radial wave-functions multiplied by r.
    mdarray<double, 2> free_atom_wave_functions_x_deriv_;
    /// Potential generated by the electronic ensity.
    /** Nuclear potential -z/r can always be treated analyticaly */
    Spline<double> free_atom_electronic_potential_;
    /// NIST total energy for LDA calculation.
    double NIST_LDA_Etot_{0};
    /// NIST total energy for scalar-relativistic LDA calculation.
    double NIST_ScRLDA_Etot_{0};
    /// Energies of atomic levels
    std::vector<double> enu_;

    void
    find_new_rho(std::vector<double> const& veff__, bool rel__)
    {
        int np = free_atom_radial_grid().num_points();
        std::memset(&free_atom_density_spline_(0), 0, np * sizeof(double));

        #pragma omp parallel for
        for (int ist = 0; ist < num_atomic_levels(); ist++) {
            relativity_t rt = (rel__) ? relativity_t::dirac : relativity_t::none;
            Bound_state bound_state(rt, zn(), atomic_level(ist).n, atomic_level(ist).l, atomic_level(ist).k,
                                    free_atom_radial_grid(), veff__, enu_[ist]);
            enu_[ist]     = bound_state.enu();
            auto& bs_rho  = bound_state.rho();
            auto& bs_u    = bound_state.u();
            auto& bs_p    = bound_state.p();
            auto& bs_dpdr = bound_state.dpdr();

            /* assume a spherical symmetry */
            for (int i = 0; i < np; i++) {
                free_atom_orbital_density_(i, ist)        = bs_rho(i);
                free_atom_wave_functions_(i, ist)         = bs_u(i);
                free_atom_wave_functions_x_(i, ist)       = bs_p(i);
                free_atom_wave_functions_x_deriv_(i, ist) = bs_dpdr[i];
            }
            #pragma omp critical
            for (int i = 0; i < np; i++) {
                /* sum of squares of spherical harmonics for angular momentm l is (2l+1)/4pi */
                free_atom_density_spline_(i) +=
                        atomic_level(ist).occupancy * free_atom_orbital_density_(i, ist) / fourpi;
            }
        }

        free_atom_density_spline_.interpolate();
    }

    void
    find_potential()
    {
    }

  public:
    Free_atom(Free_atom&& src) = default;

    /// Constructor
    Free_atom(int zn__)
        : Atom_type_base(zn__)
        , NIST_LDA_Etot_(atomic_energy_NIST_LDA[zn_ - 1])
    {
    }

    /// Constructor
    Free_atom(std::string symbol__)
        : Atom_type_base(symbol__)
        , NIST_LDA_Etot_(atomic_energy_NIST_LDA[zn_ - 1])
    {
    }

    json
    ground_state(double energy_tol, double charge_tol, bool rel)
    {
        PROFILE("sirius::Free_atom::ground_state");

        int np = free_atom_radial_grid().num_points();
        assert(np > 0);

        free_atom_orbital_density_        = mdarray<double, 2>(np, num_atomic_levels());
        free_atom_wave_functions_         = mdarray<double, 2>(np, num_atomic_levels());
        free_atom_wave_functions_x_       = mdarray<double, 2>(np, num_atomic_levels());
        free_atom_wave_functions_x_deriv_ = mdarray<double, 2>(np, num_atomic_levels());

        XC_functional_base* Ex = nullptr;
        XC_functional_base Ec("XC_LDA_C_VWN", 1);
        ;
        if (rel) {
            RTE_THROW("Fixme : the libxc staring with version 4 changed the way to set relativitic LDA exchange");
            Ex = new XC_functional_base("XC_LDA_REL_X", 1);
        } else {
            Ex = new XC_functional_base("XC_LDA_X", 1);
        }

        std::vector<double> veff(np);
        std::vector<double> vrho(np);
        std::vector<double> vnuc(np);
        for (int i = 0; i < np; i++) {
            vnuc[i] = -zn() * free_atom_radial_grid().x_inv(i);
            veff[i] = vnuc[i];
            vrho[i] = 0;
        }

        auto mixer = std::make_shared<sirius::mixer::Anderson<std::vector<double>>>(12,  // max history
                                                                                    0.8, // beta
                                                                                    0.1, // beta0
                                                                                    1.0  // beta scaling factor
        );

        // use simple inner product for mixing
        auto mixer_function_prop = sirius::mixer::FunctionProperties<std::vector<double>>(
                [](const std::vector<double>& x) -> std::size_t { return x.size(); },
                [](const std::vector<double>& x, const std::vector<double>& y) -> double {
                    double result = 0.0;
                    for (std::size_t i = 0; i < x.size(); ++i)
                        result += x[i] * y[i];
                    return result;
                },
                [](double alpha, std::vector<double>& x) -> void {
                    for (auto& val : x)
                        val *= alpha;
                },
                [](const std::vector<double>& x, std::vector<double>& y) -> void {
                    std::copy(x.begin(), x.end(), y.begin());
                },
                [](double alpha, const std::vector<double>& x, std::vector<double>& y) -> void {
                    for (std::size_t i = 0; i < x.size(); ++i)
                        y[i] += alpha * x[i];
                },
                [](double c, double s, std::vector<double>& x, std::vector<double>& y) -> void {
                    for (std::size_t i = 0; i < x.size(); ++i) {
                        auto xi = x[i];
                        auto yi = y[i];
                        x[i]    = xi * c + yi * s;
                        y[i]    = xi * -s + yi * c;
                    }
                });

        // initialize with value of vrho
        mixer->initialize_function<0>(mixer_function_prop, vrho, vrho.size());

        free_atom_density_spline_ = Spline<double>(free_atom_radial_grid());

        Spline<double> f(free_atom_radial_grid());

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
            rho_old = free_atom_density_spline_.values();

            find_new_rho(veff, rel);

            /* find RMS in charge density change */
            charge_rms = 0.0;
            for (int i = 0; i < np; i++) {
                charge_rms += std::pow(free_atom_density_spline_(i) - rho_old[i], 2);
            }
            charge_rms = std::sqrt(charge_rms / np);

            /* compute Hartree potential */
            free_atom_density_spline_.integrate(g2, 2);
            double t1 = free_atom_density_spline_.integrate(g1, 1);

            for (int i = 0; i < np; i++) {
                vh[i] = fourpi * (g2[i] / free_atom_radial_grid(i) + t1 - g1[i]);
            }

            /* compute XC potential and energy */
            Ex->get_lda(np, &free_atom_density_spline_(0), &vx[0], &ex[0]);
            Ec.get_lda(np, &free_atom_density_spline_(0), &vc[0], &ec[0]);
            for (int ir = 0; ir < np; ir++) {
                vxc[ir] = (vx[ir] + vc[ir]);
                exc[ir] = (ex[ir] + ec[ir]);
            }

            /* mix old and new effective potential */
            for (int i = 0; i < np; i++) {
                vrho[i] = vh[i] + vxc[i];
            }
            mixer->set_input<0>(vrho);
            mixer->mix(1e-16);
            mixer->get_output<0>(vrho);
            for (int i = 0; i < np; i++) {
                veff[i] = vrho[i] + vnuc[i];
            }

            /* sum of occupied eigen values */
            double eval_sum = 0.0;
            for (int ist = 0; ist < num_atomic_levels(); ist++) {
                eval_sum += atomic_level(ist).occupancy * enu_[ist];
            }

            for (int i = 0; i < np; i++) {
                f(i) = vrho[i] * free_atom_density_spline_(i);
            }
            /* kinetic energy */
            energy_kin =
                    eval_sum - fourpi * (f.interpolate().integrate(2) - zn() * free_atom_density_spline_.integrate(1));

            /* XC energy */
            for (int i = 0; i < np; i++) {
                f(i) = exc[i] * free_atom_density_spline_(i);
            }
            energy_xc = fourpi * f.interpolate().integrate(2);

            /* electron-nuclear energy: \int vnuc(r) * rho(r) r^2 dr */
            energy_enuc = -fourpi * zn() * free_atom_density_spline_.integrate(1);

            /* Coulomb energy */
            for (int i = 0; i < np; i++) {
                f(i) = vh[i] * free_atom_density_spline_(i);
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
            dict["converged"]          = true;
            dict["num_scf_iterations"] = num_iter;
        } else {
            dict["converged"] = false;
        }
        dict["energy_diff"] = energy_diff;
        dict["charge_rms"]  = charge_rms;
        dict["energy_tot"]  = energy_tot;

        free_atom_electronic_potential_ = Spline<double>(free_atom_radial_grid_, vrho);

        return dict;

        // double Eref = (rel) ? NIST_ScRLDA_Etot_ : NIST_LDA_Etot_;

        // printf("\n");
        // printf("Radial gird\n");
        // printf("-----------\n");
        // printf("type             : %s\n", radial_grid().name().c_str());
        // printf("number of points : %i\n", np);
        // printf("origin           : %20.12f\n", radial_grid(0));
        // printf("infinity         : %20.12f\n", radial_grid(np - 1));
        // printf("\n");
        // printf("Energy\n");
        // printf("------\n");
        // printf("Ekin  : %20.12f\n", energy_kin);
        // printf("Ecoul : %20.12f\n", energy_coul);
        // printf("Eenuc : %20.12f\n", energy_enuc);
        // printf("Eexc  : %20.12f\n", energy_xc);
        // printf("Total : %20.12f\n", energy_tot);
        // printf("NIST  : %20.12f\n", Eref);

        ///* difference between NIST and computed total energy. Comparison is valid only for VWN XC functional. */
        // double dE = (Utils::round(energy_tot, 6) - Eref);
        // std::cerr << zn() << " " << dE << " # " << symbol() << std::endl;

        // return energy_tot;
    }

    inline void
    generate_local_orbitals(std::string const& recipe__)
    {
        // json dict;
        // std::istringstream(recipe__) >> dict;

        // int idxlo{0};
        // for (auto& e: dict) {
        //     for (auto& lo_desc: e) {
        //         if (lo_desc.count("enu")) {
        //         }
        //         int n = lo_desc["n"];
        //         int o = lo_desc["o"];
        //
        //         //std::cout << r << "\n";

        //    }

        //}

        //    for (int n = 1; n <= 7; n++) {
        //        for (int l = 0; l < 4; l++) {
        //            if (nl_v[n][l]) {
        //                if (lo_type.find("lo1") != std::string::npos) {
        //                    a.add_lo_descriptor(idxlo, n, l, e_nl_v[n][l], 0, 1);
        //                    a.add_lo_descriptor(idxlo, n, l, e_nl_v[n][l], 1, 1);
        //                    idxlo++;
        //
    }

    inline double
    free_atom_orbital_density(int ir, int ist) const
    {
        return free_atom_orbital_density_(ir, ist);
    }

    inline double
    free_atom_wave_function(int ir, int ist) const
    {
        return free_atom_wave_functions_(ir, ist);
    }

    inline double
    free_atom_electronic_potential(double x) const
    {
        return free_atom_electronic_potential_.at_point(x);
    }

    std::vector<double>
    radial_grid_points() const
    {
        std::vector<double> v = free_atom_radial_grid().values();
        return v;
    }

    std::vector<double>
    free_atom_wave_function(int ist__) const
    {
        int np = free_atom_radial_grid().num_points();
        std::vector<double> v(np);
        for (int i = 0; i < np; i++) {
            v[i] = free_atom_wave_function(i, ist__);
        }
        return v;
    }

    std::vector<double>
    free_atom_wave_function_x(int ist__) const
    {
        int np = free_atom_radial_grid().num_points();
        std::vector<double> v(np);
        for (int i = 0; i < np; i++) {
            v[i] = free_atom_wave_functions_x_(i, ist__);
        }
        return v;
    }

    std::vector<double>
    free_atom_wave_function_x_deriv(int ist__) const
    {
        int np = free_atom_radial_grid().num_points();
        std::vector<double> v(np);
        for (int i = 0; i < np; i++) {
            v[i] = free_atom_wave_functions_x_deriv_(i, ist__);
        }
        return v;
    }

    std::vector<double>
    free_atom_electronic_potential() const
    {
        std::vector<double> v = free_atom_electronic_potential_.values();
        return v;
    }

    double
    atomic_level_energy(int ist__) const
    {
        return enu_[ist__];
    }

    /// Get residual.
    std::vector<double>
    free_atom_wave_function_residual(int ist__) const
    {
        int np = free_atom_radial_grid_.num_points();
        Spline<double> p(free_atom_radial_grid_);
        for (int i = 0; i < np; i++) {
            p(i) = free_atom_wave_functions_x_(i, ist__);
        }
        p.interpolate();
        std::vector<double> v(np);
        int l = atomic_level(ist__).l;
        for (int i = 0; i < np; i++) {
            double x = free_atom_radial_grid_[i];
            v[i]     = -0.5 * p.deriv(2, i) +
                   (free_atom_electronic_potential_(i) - zn() / x + l * (l + 1) / x / x / 2) * p(i) -
                   enu_[ist__] * p(i);
        }
        return v;
    }
};

} // namespace sirius

#endif
