// Copyright (c) 2013-2024 Anton Kozhevnikov, Thomas Schulthess
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

#include <algorithm>
#include <sirius.hpp>
#include "mixer/anderson_mixer.hpp"
#include "unit_cell/atomic_conf.hpp"
#include "potential/xc_functional.hpp"

using namespace sirius;

/// Helper class to compute free atom in spherical potential approximation.
class Free_atom : public sirius::Atom_type
{
  private:
    /// Eigen-values of each atomic level.
    std::vector<double> enu_;
    /// Charge density of each atomic level.
    mdarray<double, 2> orbital_density_;
    /// Wave-function of each atomic level.
    mdarray<double, 2> wave_functions_;
    /// Total spherical potential.
    Spline<double> potential_;
    /// Minimum value of the radial grid.
    double rmin_{1e-6};
    /// Type of relativity treatment.
    relativity_t rel_{relativity_t::none};

  public:
    /// NIST reference energy for LDA calculation.
    double NIST_LDA_Etot{0};
    /// NIST reference energy for scalar-relativistic calculation.
    double NIST_ScRLDA_Etot{0};

    Free_atom(Free_atom&& src) = default;

    /// Constructor.
    Free_atom(Simulation_parameters const& param__, std::string const symbol, std::string const name, int zn,
              double mass, std::vector<atomic_level_descriptor> const& levels_nl, int num_points, double rmax,
              relativity_t rel__)
        : Atom_type(param__, symbol, name, zn, mass, levels_nl)
        , rel_{rel__}
    {
        // radial_grid_ = sirius::Radial_grid_exp<double>(2000 + 150 * zn, rmin, 15.0 + 0.15 * zn, 1.0);
        // radial_grid_ = sirius::Radial_grid_pow<double>(6000 + 50 * zn, rmin, 15.0 + 0.15 * zn, 3.0);
        /* init free atom radial grid */
        radial_grid_ = Radial_grid_pow<double>(num_points, rmin_, rmax, 3.0);

        enu_.resize(num_atomic_levels());
        /* starting values for E_{nu} */
        for (int ist = 0; ist < num_atomic_levels(); ist++) {
            enu_[ist] = -1.0 * zn / 2 / std::pow(double(atomic_level(ist).n), 2);
        }

        orbital_density_ = mdarray<double, 2>({num_points, num_atomic_levels()});
        wave_functions_  = mdarray<double, 2>({num_points, num_atomic_levels()});
    }

    double
    enu(int ist) const
    {
        return enu_[ist];
    }

    double
    ground_state(double energy_tol, double charge_tol)
    {
        PROFILE("sirius::Free_atom::ground_state");

        int np = radial_grid().num_points();
        RTE_ASSERT(np > 0);

        std::unique_ptr<XC_functional_base> Ex;
        XC_functional_base Ec("XC_LDA_C_VWN", 1);

        if (rel_ == relativity_t::koelling_harmon) {
            Ex = std::make_unique<XC_functional_base>("XC_LDA_X_REL", 1);
        } else {
            Ex = std::make_unique<XC_functional_base>("XC_LDA_X", 1);
        }

        std::vector<double> veff(np);
        std::vector<double> vrho(np);
        std::vector<double> vnuc(np);
        for (int i = 0; i < np; i++) {
            vnuc[i] = -zn() * radial_grid().x_inv(i);
            veff[i] = vnuc[i];
            vrho[i] = 0;
        }

        auto mixer = std::make_shared<mixer::Anderson<std::vector<double>>>(12,  // max history
                                                                            0.8, // beta
                                                                            0.1, // beta0
                                                                            1.0  // beta scaling factor
        );

        auto mixer_function_prop = mixer::FunctionProperties<std::vector<double>>(
                [](const std::vector<double>& x) -> std::size_t { return x.size(); },
                /* use simple inner product for mixing */
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

        /* we will mix electronic part of effective potential */
        /* initialize with value of vrho */
        mixer->initialize_function<0>(mixer_function_prop, vrho, vrho.size());

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

        double energy_tot = 0.0;
        double energy_tot_old;
        double charge_rms;
        double energy_diff;
        double energy_enuc = 0;
        double energy_xc   = 0;
        double energy_kin  = 0;
        double energy_coul = 0;

        bool converged = false;

        for (int iter = 0; iter < 200; iter++) {
            rho_old = rho.values();

            rho = [](int i) { return 0; };

            // std::memset(&rho(0), 0, rho.num_points() * sizeof(double));
            #pragma omp parallel default(shared)
            {
                std::vector<double> rho_t(rho.num_points(), 0);

                #pragma omp for
                for (int ist = 0; ist < num_atomic_levels(); ist++) {
                    Bound_state bound_state(rel_, zn(), atomic_level(ist).n, atomic_level(ist).l, atomic_level(ist).k,
                                            radial_grid(), veff, enu_[ist]);
                    enu_[ist]    = bound_state.enu();
                    auto& bs_rho = bound_state.rho();
                    auto& bs_u   = bound_state.u();

                    /* assume spherical symmetry */
                    for (int i = 0; i < np; i++) {
                        orbital_density_(i, ist) = bs_rho(i);
                        wave_functions_(i, ist)  = bs_u(i);
                        /* sum of squares of spherical harmonics for angular momentum l is (2l+1)/4pi */
                        rho_t[i] += atomic_level(ist).occupancy * bs_rho(i) / fourpi;
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
            charge_rms = std::sqrt(charge_rms / np);

            rho.interpolate();

            /* compute Hartree potential */
            rho.integrate(g2, 2);
            double t1 = rho.integrate(g1, 1);

            for (int i = 0; i < np; i++) {
                vh[i] = fourpi * (g2[i] / radial_grid(i) + t1 - g1[i]);
            }

            /* compute XC potential and energy */
            Ex->get_lda(rho.num_points(), &rho(0), &vx[0], &ex[0]);
            Ec.get_lda(rho.num_points(), &rho(0), &vc[0], &ec[0]);
            for (int ir = 0; ir < rho.num_points(); ir++) {
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
            /* add nuclear potetial to the mixed electron potential */
            for (int i = 0; i < np; i++) {
                veff[i] = vrho[i] + vnuc[i];
            }

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

            /* new total energy */
            energy_tot = energy_kin + energy_xc + energy_coul + energy_enuc;

            energy_diff = std::abs(energy_tot - energy_tot_old);

            std::cout << "iteration : " << std::setw(3) << iter << ", energy diff : " << ffmt(16, 8) << energy_diff
                      << ", charge rms : " << ffmt(16, 8) << charge_rms << std::endl;
            if (energy_diff < energy_tol && charge_rms < charge_tol) {
                converged = true;
                std::cout << "Converged in " << iter << " iterations." << std::endl;
                break;
            }
        }

        if (!converged) {
            std::stringstream s;
            s << "atom " << symbol() << " is not converged" << std::endl
              << "  energy difference : " << energy_diff << std::endl
              << "  charge difference : " << charge_rms;
            RTE_THROW(s);
        }

        free_atom_density_spline_ = Spline<double>(radial_grid_, rho.values());

        potential_ = Spline<double>(radial_grid_, vrho);

        double Eref{0};
        if (rel_ == relativity_t::none) {
            Eref = NIST_LDA_Etot;
        } else if (rel_ == relativity_t::koelling_harmon) {
            Eref = NIST_ScRLDA_Etot;
        }

        std::cout << std::endl
                  << "Radial grid" << std::endl
                  << hbar(11, '-') << std::endl
                  << "type             : " << radial_grid().name() << std::endl
                  << "number of points : " << np << std::endl
                  << "origin           : " << ffmt(14, 8) << radial_grid(0) << std::endl
                  << "infinity         : " << ffmt(14, 8) << radial_grid(np - 1) << std::endl
                  << std::endl
                  << "Energy" << std::endl
                  << hbar(6, '-') << std::endl
                  << "Ekin  : " << ffmt(16, 8) << energy_kin << std::endl
                  << "Ecoul : " << ffmt(16, 8) << energy_coul << std::endl
                  << "Eenuc : " << ffmt(16, 8) << energy_enuc << std::endl
                  << "Eexc  : " << ffmt(16, 8) << energy_xc << std::endl
                  << "Total : " << ffmt(16, 8) << energy_tot << std::endl
                  << "NIST  : " << ffmt(16, 8) << Eref << std::endl;

        /* difference between NIST and computed total energy. Comparison is valid only for VWN XC functional. */
        double dE = (round(energy_tot, 6) - Eref);
        std::cerr << zn() << " " << dE << " # " << symbol() << std::endl;

        return energy_tot;
    }

    inline double
    orbital_density(int ir, int ist)
    {
        return orbital_density_(ir, ist);
    }

    inline double
    wave_function(int ir, int ist)
    {
        return wave_functions_(ir, ist);
    }

    inline double
    potential(double x)
    {
        return potential_.at_point(x);
    }

    inline void
    save_radial_functions() const
    {
        // FILE* fout = fopen("rho.dat", "w");
        // for (int ir = 0; ir < a.radial_grid().num_points(); ir++)
        //{
        //     double x = a.radial_grid(ir);
        //     fprintf(fout, "%12.6f %16.8f\n", x, rho[ir] * x * x);
        // }
        // fclose(fout);

        // mdarray<int, 2> nl_st(8, 4);
        // nl_st.zero();
        // FILE* fout = fopen((a.symbol() + "_wfs.dat").c_str(), "w");
        // for (int ist = 0; ist < a.num_atomic_levels(); ist++) {
        //     int n = a.atomic_level(ist).n;
        //     int l = a.atomic_level(ist).l;
        //     if (!nl_st(n, l) && !nl_c[n][l]) {
        //         for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
        //             double x = a.radial_grid(ir);
        //             fprintf(fout, "%12.6f %16.8f\n", x, a.free_atom_wave_function(ir, ist));
        //         }
        //         fprintf(fout, "\n");
        //         nl_st(n, l) = 1;
        //     }
        // }
        // fclose(fout);

        // FILE* fout = fopen((a.symbol() + "_wfs.dat").c_str(), "w");
        // for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
        //     double x = a.radial_grid(ir);
        //     fprintf(fout, "%12.6f ", x);

        //    for (int ist = 0; ist < a.num_atomic_levels(); ist++) {
        //        fprintf(fout, "%12.6f ", a.wave_function(ir, ist));
        //    }
        //    fprintf(fout, "\n");
        //}
        // fclose(fout);
    }
};

inline auto
init_atom_configuration(cmd_args const& args, Simulation_parameters const& param__)
{
    auto symbol     = args.value<std::string>("symbol");
    auto num_points = args.value<int>("num_points", 3000);
    auto rmax       = args.value<double>("rmax", 25);
    bool rel        = args.exist("rel");

    atomic_level_descriptor nlk;
    std::vector<atomic_level_descriptor> levels_nlk;

    for (size_t i = 0; i < atomic_conf_dictionary_[symbol]["levels"].size(); i++) {
        nlk.n         = atomic_conf_dictionary_[symbol]["levels"][i][0].get<int>();
        nlk.l         = atomic_conf_dictionary_[symbol]["levels"][i][1].get<int>();
        nlk.k         = atomic_conf_dictionary_[symbol]["levels"][i][2].get<int>();
        nlk.occupancy = atomic_conf_dictionary_[symbol]["levels"][i][3].get<double>();
        levels_nlk.push_back(nlk);
    }

    auto zn   = atomic_conf_dictionary_[symbol]["zn"].get<int>();
    auto mass = atomic_conf_dictionary_[symbol]["mass"].get<double>();
    auto name = atomic_conf_dictionary_[symbol]["name"].get<std::string>();

    Free_atom a(param__, symbol, name, zn, mass, levels_nlk, num_points, rmax,
                rel ? relativity_t::koelling_harmon : relativity_t::none);

    a.NIST_LDA_Etot    = atomic_conf_dictionary_[symbol].value("NIST_LDA_Etot", 0.0);
    a.NIST_ScRLDA_Etot = atomic_conf_dictionary_[symbol].value("NIST_ScRLDA_Etot", 0.0);
    return a;
}

inline void
generate_atom_file(cmd_args const& args, Free_atom& a)
{
    double core_cutoff = args.value<double>("core", -10.0);

    auto lo_type = args.value<std::string>("type", "lo1");

    int apw_order = args.value<int>("order", 2);

    double apw_enu = args.value<double>("apw_enu", 0.15);

    bool auto_apw_enu = args.exist("auto_apw_enu");

    // JSON_write jw(fname);
    json dict;
    dict["name"]   = a.name();
    dict["symbol"] = a.symbol();
    dict["number"] = a.zn();
    dict["mass"]   = a.mass();
    dict["rmin"]   = a.radial_grid(0);

    std::vector<atomic_level_descriptor> core;
    std::vector<atomic_level_descriptor> valence;
    std::string level_symb[] = {"s", "p", "d", "f"};

    /* number of core states for a given n,l (1 or 2) */
    mdarray<int, 2> nl_c({8, 4});
    /* number of valence states for a given n,l (1 or 2) */
    mdarray<int, 2> nl_v({8, 4});

    /* average energy of core n,l state */
    mdarray<double, 2> e_nl_c({8, 4});
    /* average energy of valence n,l state */
    mdarray<double, 2> e_nl_v({8, 4});

    nl_c.zero();
    nl_v.zero();
    e_nl_c.zero();
    e_nl_v.zero();

    std::cout << std::endl << "Core / valence partitioning" << std::endl << hbar(27, '-') << std::endl;
    if (core_cutoff <= 0) {
        std::cout << "core cutoff energy       : " << core_cutoff << std::endl;
    } else {
        std::cout << "core cutoff radius       : " << core_cutoff << std::endl;
    }
    Spline<double> rho_c(a.radial_grid());
    Spline<double> rho(a.radial_grid());
    Spline<double> s(a.radial_grid());
    int ncore{0};
    /* iterate over bound states */
    for (int ist = 0; ist < a.num_atomic_levels(); ist++) {
        int n = a.atomic_level(ist).n;
        int l = a.atomic_level(ist).l;

        std::cout << a.atomic_level(ist).n << level_symb[a.atomic_level(ist).l] << " | occ : " << ffmt(5, 2)
                  << a.atomic_level(ist).occupancy << ", energy : " << ffmt(14, 7) << a.enu(ist);

        /* total density */
        for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
            s(ir) = a.orbital_density(ir, ist);
            rho(ir) += a.atomic_level(ist).occupancy * s(ir) / fourpi;
        }

        std::vector<double> g;
        s.interpolate().integrate(g, 2);
        double rc{100};

        /* find distance at which most of the bound state's charge density is contained */
        for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
            if (1.0 - g[ir] < 1e-5) {
                rc = a.radial_grid(ir);
                break;
            }
        }

        /* assign this state to core */
        if ((core_cutoff <= 0 && a.enu(ist) < core_cutoff) || (core_cutoff > 0 && rc < core_cutoff)) {
            core.push_back(a.atomic_level(ist));
            std::cout << "  => core (rc = " << ffmt(9, 4) << rc << ")" << std::endl;

            for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
                rho_c(ir) += a.atomic_level(ist).occupancy * a.orbital_density(ir, ist) / fourpi;
            }

            nl_c(n, l)++;
            e_nl_c(n, l) += a.enu(ist);
            ncore += static_cast<int>(a.atomic_level(ist).occupancy + 1e-12);
        } else { /* assign this state to valence */
            valence.push_back(a.atomic_level(ist));
            std::cout << "  => valence (rc = " << ffmt(9, 4) << rc << ")" << std::endl;

            nl_v(n, l)++;
            e_nl_v(n, l) += a.enu(ist);
        }
    }
    std::cout << "number of core electrons : " << ncore << std::endl;

    /* average energies for {n,l} level */
    for (int n = 1; n <= 7; n++) {
        for (int l = 0; l < 4; l++) {
            e_nl_v(n, l) = nl_v(n, l) ? e_nl_v(n, l) / nl_v(n, l) : 0;
            e_nl_c(n, l) = nl_c(n, l) ? e_nl_c(n, l) / nl_c(n, l) : 0;
        }
    }

    /* get the lmax for valence states */
    int lmax{0};
    for (size_t i = 0; i < valence.size(); i++) {
        lmax = std::max(lmax, valence[i].l);
    }
    std::cout << "valence lmax : " << lmax << std::endl;

    /* valence principal quantum numbers for each l */
    std::array<std::vector<int>, 4> n_v;
    for (int n = 1; n <= 7; n++) {
        for (int l = 0; l < 4; l++) {
            if (nl_v(n, l)) {
                n_v[l].push_back(n);
            }
        }
    }

    std::cout << "valence n for each l" << std::endl;
    for (int l = 0; l < 4; l++) {
        if (n_v[l].size()) {
            std::cout << "l: " << l << ", n: ";
            for (int n : n_v[l]) {
                std::cout << n << " ";
            }
            std::cout << std::endl;
        }
    }

    /* estimate effective infinity */
    std::vector<double> g;
    rho.interpolate().integrate(g, 2);
    double rinf{0};
    for (int ir = a.radial_grid().num_points() - 1; ir >= 0; ir--) {
        if (g[ir] / g.back() < 0.99999999) {
            rinf = a.radial_grid(ir);
            break;
        }
    }
    std::cout << "effective infinity : " << ffmt(12, 6) << rinf << std::endl;

    /* estimate core radius */
    double core_radius = (core_cutoff > 0) ? core_cutoff : 0.65;
    if (ncore != 0) {
        std::vector<double> g;
        rho_c.interpolate().integrate(g, 2);

        for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
            if ((ncore - fourpi * g[ir]) / ncore < 1e-5) {
                core_radius = a.radial_grid(ir);
                break;
            }
        }
    }

    /* good number of MT points */
    int nrmt{500};

    std::cout << "minimum MT radius : " << ffmt(12, 6) << core_radius << std::endl;

    dict["rmt"]  = core_radius;
    dict["nrmt"] = nrmt;
    dict["rinf"] = rinf;

    /* compact representation of core states */
    std::string core_str;
    for (int n = 1; n <= 7; n++) {
        for (int l = 0; l < 4; l++) {
            if (nl_c(n, l)) {
                core_str += std::to_string(n) + level_symb[l];
            }
        }
    }
    std::cout << "core states : " << core_str << std::endl;
    dict["core"] = core_str;

    dict["valence"] = json::array();
    dict["valence"].push_back(json::object({{"basis", json::array({})}}));
    dict["valence"][0]["basis"].push_back({{"enu", apw_enu}, {"dme", 0}, {"auto", 0}});
    if (apw_order == 2) {
        dict["valence"][0]["basis"].push_back({{"enu", apw_enu}, {"dme", 1}, {"auto", 0}});
    }

    if (auto_apw_enu) {
        for (int l = 0; l <= lmax; l++) {
            int n{0};
            /* APW for s,p,d,f is constructed for the highest valence state */
            for (auto e : valence) {
                if (e.l == l) {
                    n = std::max(n, e.n);
                }
            }
            dict["valence"].push_back(json::object({{"n", n}, {"l", l}, {"basis", json::array({})}}));
            dict["valence"].back()["basis"].push_back({{"enu", apw_enu}, {"dme", 0}, {"auto", 1}});
            if (apw_order == 2) {
                dict["valence"].back()["basis"].push_back({{"enu", apw_enu}, {"dme", 1}, {"auto", 1}});
            }
        }
    }

    int idxlo{0};
    for (int n = 1; n <= 7; n++) {
        for (int l = 0; l < 4; l++) {
            /* for each valence state */
            if (nl_v(n, l)) {
                /* add 2nd order local orbital composed of u_l(r, E_l) and \dot u_l(r, E_l)
                   linearization energy E_l is searched automatically */
                if (lo_type.find("lo1") != std::string::npos) {
                    a.add_lo_descriptor(idxlo, n, l, e_nl_v(n, l), 0, 1);
                    a.add_lo_descriptor(idxlo, n, l, e_nl_v(n, l), 1, 1);
                    idxlo++;
                }

                /* add 2nd order local orbital composed of \dot u_l(r, E_l) and \ddot u_l(r, E_l)
                   linearization energy E_l is searched automatically */
                if (lo_type.find("lo2") != std::string::npos) {
                    a.add_lo_descriptor(idxlo, n, l, e_nl_v(n, l), 1, 1);
                    a.add_lo_descriptor(idxlo, n, l, e_nl_v(n, l), 2, 1);
                    idxlo++;
                }

                /* add 3rd order local orbital composed of u_l(r, E=0.15), \dot u_l(r, E=0.15) and u_l(r, E_l)
                   linearization energy E_l is searched automatically */
                if (lo_type.find("LO1") != std::string::npos) {
                    a.add_lo_descriptor(idxlo, 0, l, 0.15, 0, 0);
                    a.add_lo_descriptor(idxlo, 0, l, 0.15, 1, 0);
                    a.add_lo_descriptor(idxlo, n, l, e_nl_v(n, l), 0, 1);
                    idxlo++;
                }

                /* add high-energy 3rd order local orbital composed of u_l(r, E=1.15), \dot u_l(r, E=1.15) and u_l(r,
                   E_l) linearization energy E_l is searched automatically */
                if (lo_type.find("LO2") != std::string::npos) {
                    a.add_lo_descriptor(idxlo, 0, l, 1.15, 0, 0);
                    a.add_lo_descriptor(idxlo, 0, l, 1.15, 1, 0);
                    a.add_lo_descriptor(idxlo, n + 1, l, e_nl_v(n, l) + 1, 0, 1);
                    idxlo++;
                }
            }
        }
    }

    /* add high angular momentum 2nd order local orbitals with fixed linearisation energies */
    if (lo_type.find("lo3") != std::string::npos) {
        for (int l = lmax + 1; l < lmax + 4; l++) {
            a.add_lo_descriptor(idxlo, 0, l, 0.15, 0, 0);
            a.add_lo_descriptor(idxlo, 0, l, 0.15, 1, 0);
            idxlo++;

            a.add_lo_descriptor(idxlo, 0, l, 0.15, 1, 0);
            a.add_lo_descriptor(idxlo, 0, l, 0.15, 2, 0);
            idxlo++;
        }
    }

    // if (lo_type.find("lo4") != std::string::npos) {
    //     for (int n = 1; n <= 6; n++) {
    //         for (int l = 0; l < n; l++) {
    //             a.add_lo_descriptor(idxlo, n, l, 0.15, 0, 1);
    //             a.add_lo_descriptor(idxlo, n, l, 0.15, 1, 1);
    //             idxlo++;
    //         }
    //     }
    // }

    std::vector<double> fa_rho(a.radial_grid().num_points());

    for (int i = 0; i < a.radial_grid().num_points(); i++) {
        fa_rho[i] = a.free_atom_density(i);
    }

    dict["free_atom"]                = json::object();
    dict["free_atom"]["density"]     = fa_rho;
    dict["free_atom"]["radial_grid"] = a.radial_grid().values();

    Radial_grid_pow<double> rg(nrmt, a.radial_grid(0), 1.8, 3);
    auto x = rg.values();
    std::vector<double> veff;
    for (int ir = 0; ir < rg.num_points(); ir++) {
        veff.push_back(a.potential(rg[ir]) - a.zn() * rg.x_inv(ir));
    }
    a.set_radial_grid(nrmt, x.data());

    std::cout << "=== initializing atom ===" << std::endl;
    a.init();
    Atom_symmetry_class a1(0, a);
    a1.set_spherical_potential(veff);
    a1.generate_radial_functions(relativity_t::none);
    mpi::pstdout pout(mpi::Communicator::self());
    a1.write_enu(pout);
    std::cout << pout.flush(0);

    auto lo_to_str = [](sirius::local_orbital_descriptor lod) {
        std::stringstream s;
        s << "[";
        for (size_t o = 0; o < lod.rsd_set.size(); o++) {
            s << "{"
              << "\"n\" : " << lod.rsd_set[o].n << ", \"enu\" : " << lod.rsd_set[o].enu
              << ", \"dme\" : " << lod.rsd_set[o].dme << ", \"auto\" : " << lod.rsd_set[o].auto_enu << "}";
            if (o != lod.rsd_set.size() - 1) {
                s << ", ";
            }
        }
        s << "]";
        return s.str();
    };

    auto inc   = a1.check_lo_linear_independence(0.0001);
    dict["lo"] = json::array();

    for (int j = 0; j < a1.num_lo_descriptors(); j++) {
        auto s = lo_to_str(a1.lo_descriptor(j));
        if (!inc[j]) {
            std::cout << "X ";
        } else {
            std::cout << "  ";
            dict["lo"].push_back({{"l", a1.lo_descriptor(j).am.l()}, {"basis", json::parse(s)}});
        }
        std::cout << "l: " << a1.lo_descriptor(j).am.l() << ", basis: " << s << std::endl;
    }

    std::ofstream ofs(a.symbol() + std::string(".json"), std::ofstream::out | std::ofstream::trunc);
    ofs << dict.dump(4);
    ofs.close();

    // if (write_to_xml)
    //{
    //     std::string fname = a.symbol() + std::string(".xml");
    //     FILE* fout = fopen(fname.c_str(), "w");
    //     fprintf(fout, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    //     fprintf(fout, "<spdb>\n");
    //     fprintf(fout, "  <sp chemicalSymbol=\"%s\" name=\"%s\" z=\"%f\" mass=\"%f\">\n", a.symbol().c_str(),
    //     a.name().c_str(), -1.0 * a.zn(), a.mass()); fprintf(fout, "    <muffinTin rmin=\"%e\" radius=\"%f\"
    //     rinf=\"%f\" radialmeshPoints=\"%i\"/>\n", 1e-6, 2.0, rinf, 1000);

    //    for (int ist = 0; ist < a.num_atomic_levels(); ist++)
    //    {
    //        std::string str_core = (enu[ist] < core_cutoff_energy) ? "true" : "false";

    //        fprintf(fout, "      <atomicState n=\"%i\" l=\"%i\" kappa=\"%i\" occ=\"%f\" core=\"%s\"/>\n",
    //                a.atomic_level(ist).n,
    //                a.atomic_level(ist).l,
    //                a.atomic_level(ist).k,
    //                a.atomic_level(ist).occupancy,
    //                str_core.c_str());
    //    }
    //    fprintf(fout, "      <basis>\n");
    //    fprintf(fout, "        <default type=\"lapw\" trialEnergy=\"0.15\" searchE=\"false\"/>\n");
    //    for (int l = 0; l < 4; l++)
    //    {
    //        if (n_v[l].size() == 1)
    //        {
    //            int n = n_v[l][0];

    //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //            fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //            e_nl_v(n, l)); fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\"
    //            searchE=\"false\"/>\n", e_nl_v(n, l)); fprintf(fout, "        </lo>\n");

    //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //            fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //            e_nl_v(n, l)); fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\"
    //            searchE=\"false\"/>\n", e_nl_v(n, l)); fprintf(fout, "        </lo>\n");
    //
    //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //            fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //            e_nl_v(n, l)); fprintf(fout, "          <wf matchingOrder=\"3\" trialEnergy=\"%f\"
    //            searchE=\"false\"/>\n", e_nl_v(n, l)); fprintf(fout, "        </lo>\n");
    //        }
    //        if (n_v[l].size() > 1)
    //        {
    //            for (size_t i = 0; i < n_v[l].size() - 1; i++)
    //            {
    //                int n = n_v[l][i];
    //                int n1 = n_v[l][i + 1];

    //                fprintf(fout, "        <lo l=\"%i\">\n", l);
    //                fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //                e_nl_v(n, l)); fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\"
    //                searchE=\"false\"/>\n", e_nl_v(n, l)); fprintf(fout, "        </lo>\n");

    //                fprintf(fout, "        <lo l=\"%i\">\n", l);
    //                fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //                e_nl_v(n, l)); fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\"
    //                searchE=\"false\"/>\n", e_nl_v(n, l)); fprintf(fout, "        </lo>\n");
    //
    //                fprintf(fout, "        <lo l=\"%i\">\n", l);
    //                fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //                e_nl_v(n, l)); fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\"
    //                searchE=\"false\"/>\n", e_nl_v[n1][l]); fprintf(fout, "        </lo>\n");
    //            }
    //            int n = n_v[l].back();
    //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //            fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //            e_nl_v(n, l)); fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\"
    //            searchE=\"false\"/>\n", e_nl_v(n, l)); fprintf(fout, "        </lo>\n");
    //        }

    //        //if (n_v[l].size())
    //        //    fprintf(fout, "        <custom l=\"%i\" type=\"lapw\" trialEnergy=\"%f\" searchE=\"false\"/>\n", l,
    //        e_nl_v[n_v[l].front()][l]);

    //        //for (int n = 1; n <= 7; n++)
    //        //{
    //        //    if (nl_v[n][l])
    //        //    {
    //        //        fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        //        fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //        e_nl_v(n, l));
    //        //        fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //        e_nl_v(n, l));
    //        //        fprintf(fout, "        </lo>\n");

    //        //        //if (e_nl_v(n, l) > -5.0)
    //        //        //{
    //        //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        //            fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //        e_nl_v(n, l));
    //        //            fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //        e_nl_v(n, l));
    //        //            fprintf(fout, "        </lo>\n");
    //        //        //}
    //        //
    //        //        //if (e_nl_v(n, l) > -1.0)
    //        //        //{
    //        //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        //            fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //        e_nl_v(n, l));
    //        //            fprintf(fout, "          <wf matchingOrder=\"3\" trialEnergy=\"%f\" searchE=\"false\"/>\n",
    //        e_nl_v(n, l));
    //        //            fprintf(fout, "        </lo>\n");
    //        //        //}

    //        //        //fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        //        //fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"true\"/>\n",
    //        e_nl_v(n, l));
    //        //        //fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"true\"/>\n",
    //        e_nl_v(n, l));
    //        //        //fprintf(fout, "        </lo>\n");
    //        //        //
    //        //        //if (e_nl_v(n, l) < -1.0)
    //        //        //{
    //        //        //    fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        //        //    fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"0.15\"
    //        searchE=\"false\"/>\n");
    //        //        //    fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"0.15\"
    //        searchE=\"false\"/>\n");
    //        //        //    fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"true\"/>\n",
    //        e_nl_v(n, l));
    //        //        //    fprintf(fout, "        </lo>\n");
    //        //        //}
    //        //    }
    //        //}
    //    }
    //    for (int l = lmax + 1; l <= lmax + 3; l++)
    //    {
    //        fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"0.15\" searchE=\"false\"/>\n");
    //        fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"0.15\" searchE=\"false\"/>\n");
    //        fprintf(fout, "        </lo>\n");

    //        fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"0.15\" searchE=\"false\"/>\n");
    //        fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"0.15\" searchE=\"false\"/>\n");
    //        fprintf(fout, "        </lo>\n");
    //    }
    //    fprintf(fout, "      </basis>\n");
    //    fprintf(fout, "  </sp>\n");
    //    fprintf(fout, "</spdb>\n");

    //    fclose(fout);
    //}
}

int
main(int argn, char** argv)
{
    /* handle command line arguments */
    cmd_args args;
    args.register_key("--symbol=", "{string} symbol of a chemical element");
    args.register_key("--type=", "{lo1, lo2, lo3, LO1, LO2} type of local orbital basis");
    args.register_key("--core=", "{double} cutoff for core states: energy (in Ha, if <0), radius (in a.u. if >0)");
    args.register_key("--order=", "{int} order of augmentation; 1: APW, 2: LAPW");
    args.register_key("--apw_enu=", "{double} default value for APW linearization energies");
    args.register_key("--auto_apw_enu", "allow search of APW linearization energies of valence states");
    args.register_key("--rel", "use scalar-relativistic solver");
    args.register_key("--num_points=", "{int} number of radial grid points");
    args.register_key("--rmax=", "{double} maximum value of radial grid");
    args.parse_args(argn, argv);

    if (argn == 1 || args.exist("help")) {
        std::cout << std::endl;
        std::cout << "Atom (L)APW+lo basis generation." << std::endl;
        std::cout << std::endl;
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
        args.print_help();
        std::cout << std::endl;
        std::cout << "Definition of the local orbital types:" << std::endl;
        std::cout << "  lo1 : 2nd order valence local orbitals composed of {u_l(r, E_l), \\dot u_l(e, E_l)}"
                  << std::endl;
        std::cout << "  lo2 : 2nd order valence local orbitals composed of {\\dot u_l(r, E_l), \\ddot u_l(e, E_l)}"
                  << std::endl;
        std::cout << "  lo3 : two 2nd order high angular momentum local orbitals composed of {u_l(r, E=0.15), \\dot "
                     "u_l(e, E=0.15)}"
                  << std::endl;
        std::cout << "        and {\\dot u_l(r, E=0.15), \\ddot u_l(e, E=0.15)}" << std::endl;
        std::cout << "  LO1 : 3rd order valence local orbital composed of {u_l(r, E=0.15), \\dot u_l(r, E=0.15), "
                     "u_l(r, E_l)}"
                  << std::endl;
        std::cout << "  LO2 : 3rd order high-energy local orbital composed of {u_l(r, E=1.15), \\dot u_l(r, E=1.15), "
                     "u_l(r, E_l)}"
                  << std::endl;
        std::cout << std::endl;
        std::cout << "You can combine different local-orbital types, for example lo1+lo2+LO1" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << std::endl;
        std::cout << "  generate default basis for lithium:" << std::endl;
        std::cout << "    ./atom --symbol=Li" << std::endl;
        std::cout << std::endl;
        std::cout << "  generate high precision basis for titanium:" << std::endl;
        std::cout << "    ./atom --type=lo+LO --symbol=Ti" << std::endl;
        std::cout << std::endl;
        std::cout << "  make all states of iron to be valence:" << std::endl;
        std::cout << "    ./atom --core=-1000 --symbol=Fe" << std::endl;
        std::cout << std::endl;
        return 0;
    }

    sirius::initialize(true);

    sirius::Simulation_parameters param;
    param.lmax_apw(-1);

    auto symbol = args.value<std::string>("symbol");
    auto zn     = atomic_conf_dictionary_[symbol]["zn"].get<int>();

    std::cout << hbar(78, '-') << std::endl;
    std::cout << "atom : " << symbol << ", Z : " << zn << std::endl;
    std::cout << hbar(78, '-') << std::endl;

    /* use standard ground state atomic configuration */
    auto a = init_atom_configuration(args, param);
    /* solve free atom */
    a.ground_state(1e-8, 1e-8);
    /* save species file */
    generate_atom_file(args, a);

    sirius::finalize();

    auto timing_result = global_rtgraph_timer.process();
    std::cout << timing_result.print({rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
                                      rt_graph::Stat::SelfPercentage, rt_graph::Stat::Median, rt_graph::Stat::Min,
                                      rt_graph::Stat::Max});
}
