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

#include <algorithm>
#include <sirius.h>
#include "Mixer/broyden1_mixer.hpp"

double const rmin{1e-5};

class Free_atom : public sirius::Atom_type
{
  private:
    mdarray<double, 2>     free_atom_orbital_density_;
    mdarray<double, 2>     free_atom_wave_functions_;
    sirius::Spline<double> free_atom_potential_;

  public:
    double NIST_LDA_Etot{0};
    double NIST_ScRLDA_Etot{0};

    Free_atom(Free_atom&& src) = default;

    Free_atom(sirius::Simulation_parameters const&        param__,
              const std::string                           symbol,
              const std::string                           name,
              int                                         zn,
              double                                      mass,
              std::vector<atomic_level_descriptor> const& levels_nl)
        : Atom_type(param__, symbol, name, zn, mass, levels_nl)
    {
        radial_grid_ = sirius::Radial_grid_exp<double>(2000 + 150 * zn, rmin, 20.0 + 0.25 * zn, 1.0);
    }

    double ground_state(double solver_tol, double energy_tol, double charge_tol, std::vector<double>& enu, bool rel)
    {
        PROFILE("sirius::Free_atom::ground_state");

        int np = radial_grid().num_points();
        assert(np > 0);

        free_atom_orbital_density_ = mdarray<double, 2>(np, num_atomic_levels());
        free_atom_wave_functions_  = mdarray<double, 2>(np, num_atomic_levels());

        sirius::XC_functional_base *Ex;
        sirius::XC_functional_base Ec("XC_LDA_C_VWN", 1);

        if (rel) {
            Ex = new sirius::XC_functional_base("XC_LDA_X_REL", 1);
        } else {
            Ex = new sirius::XC_functional_base("XC_LDA_X", 1);
        }

        std::vector<double> veff(np);
        std::vector<double> vrho(np);
        std::vector<double> vnuc(np);
        for (int i = 0; i < np; i++) {
            vnuc[i] = -zn() * radial_grid().x_inv(i);
            veff[i] = vnuc[i];
            vrho[i] = 0;
        }

        auto mixer = std::make_shared<sirius::mixer::Broyden1<std::vector<double>>>(12,  // max history
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
            });

        // initialize with value of vrho
        mixer->initialize_function<0>(mixer_function_prop, vrho, vrho.size());

        sirius::Spline<double> rho(radial_grid());

        sirius::Spline<double> f(radial_grid());

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

        enu.resize(num_atomic_levels());

        double energy_tot = 0.0;
        double energy_tot_old;
        double charge_rms;
        double energy_diff;
        double energy_enuc = 0;
        double energy_xc   = 0;
        double energy_kin  = 0;
        double energy_coul = 0;

        bool converged = false;

        /* starting values for E_{nu} */
        for (int ist = 0; ist < num_atomic_levels(); ist++) {
            enu[ist] = -1.0 * zn() / 2 / std::pow(double(atomic_level(ist).n), 2);
        }

        for (int iter = 0; iter < 200; iter++) {
            rho_old = rho.values();

            std::memset(&rho(0), 0, rho.num_points() * sizeof(double));
            #pragma omp parallel default(shared)
            {
                std::vector<double> rho_t(rho.num_points());
                std::memset(&rho_t[0], 0, rho.num_points() * sizeof(double));

                #pragma omp for
                for (int ist = 0; ist < num_atomic_levels(); ist++) {
                    //relativity_t rt = (rel) ? relativity_t::koelling_harmon : relativity_t::none;
                    relativity_t        rt = (rel) ? relativity_t::dirac : relativity_t::none;
                    sirius::Bound_state bound_state(rt, zn(), atomic_level(ist).n, atomic_level(ist).l,
                                                    atomic_level(ist).k, radial_grid(), veff, enu[ist]);
                    enu[ist]     = bound_state.enu();
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
            for (int i = 0; i < np; i++) {
                veff[i] = vrho[i] + vnuc[i];
            }

            //= /* mix old and new effective potential */
            //= for (int i = 0; i < np; i++)
            //=     veff[i] = (1 - beta) * veff[i] + beta * (vnuc[i] + vh[i] + vxc[i]);

            /* sum of occupied eigen values */
            double eval_sum = 0.0;
            for (int ist = 0; ist < num_atomic_levels(); ist++) {
                eval_sum += atomic_level(ist).occupancy * enu[ist];
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
                converged = true;
                std::printf("Converged in %i iterations.\n", iter);
                break;
            }
        }

        if (!converged) {
            std::printf("energy_diff : %18.10f   charge_rms : %18.10f\n", energy_diff, charge_rms);
            std::stringstream s;
            s << "atom " << symbol() << " is not converged" << std::endl
              << "  energy difference : " << energy_diff << std::endl
              << "  charge difference : " << charge_rms;
            TERMINATE(s);
        }

        free_atom_density_spline_ = sirius::Spline<double>(radial_grid_, rho.values());

        free_atom_potential_ = sirius::Spline<double>(radial_grid_, vrho);

        double Eref = (rel) ? NIST_ScRLDA_Etot : NIST_LDA_Etot;

        std::printf("\n");
        std::printf("Radial gird\n");
        std::printf("-----------\n");
        std::printf("type             : %s\n", radial_grid().name().c_str());
        std::printf("number of points : %i\n", np);
        std::printf("origin           : %20.12f\n", radial_grid(0));
        std::printf("infinity         : %20.12f\n", radial_grid(np - 1));
        std::printf("\n");
        std::printf("Energy\n");
        std::printf("------\n");
        std::printf("Ekin  : %20.12f\n", energy_kin);
        std::printf("Ecoul : %20.12f\n", energy_coul);
        std::printf("Eenuc : %20.12f\n", energy_enuc);
        std::printf("Eexc  : %20.12f\n", energy_xc);
        std::printf("Total : %20.12f\n", energy_tot);
        std::printf("NIST  : %20.12f\n", Eref);

        /* difference between NIST and computed total energy. Comparison is valid only for VWN XC functional. */
        double dE = (utils::round(energy_tot, 6) - Eref);
        std::cerr << zn() << " " << dE << " # " << symbol() << std::endl;

        return energy_tot;
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

Free_atom init_atom_configuration(const std::string& label, sirius::Simulation_parameters param__)
{
    json jin;
    std::ifstream("atoms.json") >> jin;

    atomic_level_descriptor              nlk;
    std::vector<atomic_level_descriptor> levels_nlk;

    for (size_t i = 0; i < jin[label]["levels"].size(); i++) {
        nlk.n         = jin[label]["levels"][i][0];
        nlk.l         = jin[label]["levels"][i][1];
        nlk.k         = jin[label]["levels"][i][2];
        nlk.occupancy = jin[label]["levels"][i][3];
        levels_nlk.push_back(nlk);
    }

    int zn;
    zn = jin[label]["zn"];
    double mass;
    mass = jin[label]["mass"];
    std::string name;
    name                    = jin[label]["name"];
    double NIST_LDA_Etot    = 0.0;
    NIST_LDA_Etot           = jin[label].value("NIST_LDA_Etot", NIST_LDA_Etot);
    double NIST_ScRLDA_Etot = 0.0;
    NIST_ScRLDA_Etot        = jin[label].value("NIST_ScRLDA_Etot", NIST_ScRLDA_Etot);

    Free_atom a(param__, label, name, zn, mass, levels_nlk);
    a.NIST_LDA_Etot    = NIST_LDA_Etot;
    a.NIST_ScRLDA_Etot = NIST_ScRLDA_Etot;
    return a;
}

void generate_atom_file(Free_atom&         a,
                        double             core_cutoff,
                        const std::string& lo_type,
                        int                apw_order,
                        double             apw_enu,
                        bool               auto_enu,
                        bool               write_to_xml,
                        bool               rel)
{
    std::vector<double> enu;

    std::printf("\n");
    std::printf("atom : %s, Z = %i\n", a.symbol().c_str(), a.zn());
    std::printf("----------------------------------\n");

    /* solve a free atom */
    a.ground_state(1e-10, 1e-8, 1e-8, enu, rel);

    //JSON_write jw(fname);
    json dict;
    dict["name"]   = a.name();
    dict["symbol"] = a.symbol();
    dict["number"] = a.zn();
    dict["mass"]   = a.mass();
    dict["rmin"]   = a.radial_grid(0);

    std::vector<atomic_level_descriptor> core;
    std::vector<atomic_level_descriptor> valence;
    std::string                          level_symb[] = {"s", "p", "d", "f"};

    int    nl_c[8][4];
    int    nl_v[8][4];
    double e_nl_c[8][4];
    double e_nl_v[8][4];

    std::memset(&nl_c[0][0], 0, 32 * sizeof(int));
    std::memset(&nl_v[0][0], 0, 32 * sizeof(int));
    std::memset(&e_nl_c[0][0], 0, 32 * sizeof(double));
    std::memset(&e_nl_v[0][0], 0, 32 * sizeof(double));

    std::printf("\n");
    std::printf("Core / valence partitioning\n");
    std::printf("---------------------------\n");
    if (core_cutoff <= 0) {
        std::printf("core cutoff energy       : %f\n", core_cutoff);
    } else {
        std::printf("core cutoff radius       : %f\n", core_cutoff);
    }
    sirius::Spline<double> rho_c(a.radial_grid());
    sirius::Spline<double> rho(a.radial_grid());
    sirius::Spline<double> s(a.radial_grid());
    int                    ncore{0};
    for (int ist = 0; ist < a.num_atomic_levels(); ist++) {
        int n = a.atomic_level(ist).n;
        int l = a.atomic_level(ist).l;

        std::printf("%i%s  occ : %8.4f  energy : %12.6f", a.atomic_level(ist).n, level_symb[a.atomic_level(ist).l].c_str(),
               a.atomic_level(ist).occupancy, enu[ist]);

        /* total density */
        for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
            s(ir) = a.free_atom_orbital_density(ir, ist);
            rho(ir) += a.atomic_level(ist).occupancy * s(ir) / fourpi;
        }

        std::vector<double> g;
        s.interpolate().integrate(g, 2);
        double rc{100};

        for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
            if (1.0 - g[ir] < 1e-5) {
                rc = a.radial_grid(ir);
                break;
            }
        }

        /* assign this state to core */
        if ((core_cutoff <= 0 && enu[ist] < core_cutoff) ||
            (core_cutoff > 0 && rc < core_cutoff)) {
            core.push_back(a.atomic_level(ist));
            std::printf("  => core (rc = %f)\n", rc);

            for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
                rho_c(ir) += a.atomic_level(ist).occupancy * a.free_atom_orbital_density(ir, ist) / fourpi;
            }

            nl_c[n][l]++;
            e_nl_c[n][l] += enu[ist];
            ncore += static_cast<int>(a.atomic_level(ist).occupancy + 1e-12);
        } else { /* assign this state to valence */
            valence.push_back(a.atomic_level(ist));
            std::printf("  => valence (rc = %f)\n", rc);

            nl_v[n][l]++;
            e_nl_v[n][l] += enu[ist];
        }
    }
    std::printf("number of core electrons : %i\n", ncore);

    /* average energies for {n,l} level */
    for (int n = 1; n <= 7; n++) {
        for (int l = 0; l < 4; l++) {
            if (nl_v[n][l])
                e_nl_v[n][l] /= nl_v[n][l];
            if (nl_c[n][l])
                e_nl_c[n][l] /= nl_c[n][l];
        }
    }

    /* get the lmax for valence states */
    int lmax{0};
    for (size_t i = 0; i < valence.size(); i++) {
        lmax = std::max(lmax, valence[i].l);
    }
    std::printf("lmax: %i\n", lmax);

    /* valence principal quantum numbers for each l */
    std::array<std::vector<int>, 4> n_v;
    for (int n = 1; n <= 7; n++) {
        for (int l = 0; l < 4; l++) {
            if (nl_v[n][l]) {
                n_v[l].push_back(n);
            }
        }
    }

    std::printf("valence n for each l:\n");
    for (int l = 0; l < 4; l++) {
        std::printf("l: %i, n: ", l);
        for (int n : n_v[l]) {
            std::printf("%i ", n);
        }
        std::printf("\n");
    }

    //FILE* fout = fopen("rho.dat", "w");
    //for (int ir = 0; ir < a.radial_grid().num_points(); ir++)
    //{
    //    double x = a.radial_grid(ir);
    //    fprintf(fout, "%12.6f %16.8f\n", x, rho[ir] * x * x);
    //}
    //fclose(fout);

    //mdarray<int, 2> nl_st(8, 4);
    //nl_st.zero();
    //FILE* fout = fopen((a.symbol() + "_wfs.dat").c_str(), "w");
    //for (int ist = 0; ist < a.num_atomic_levels(); ist++) {
    //    int n = a.atomic_level(ist).n;
    //    int l = a.atomic_level(ist).l;
    //    if (!nl_st(n, l) && !nl_c[n][l]) {
    //        for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
    //            double x = a.radial_grid(ir);
    //            fprintf(fout, "%12.6f %16.8f\n", x, a.free_atom_wave_function(ir, ist));
    //        }
    //        fprintf(fout, "\n");
    //        nl_st(n, l) = 1;
    //    }
    //}
    //fclose(fout);

    FILE* fout = fopen((a.symbol() + "_wfs.dat").c_str(), "w");
    for (int ir = 0; ir < a.radial_grid().num_points(); ir++) {
        double x = a.radial_grid(ir);
        fprintf(fout, "%12.6f ", x);

        for (int ist = 0; ist < a.num_atomic_levels(); ist++) {
            fprintf(fout, "%12.6f ", a.free_atom_wave_function(ir, ist));
        }
        fprintf(fout, "\n");
    }
    fclose(fout);

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
    std::printf("Effective infinity : %f\n", rinf);

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
    int nrmt{1000};

    std::printf("minimum MT radius : %f\n", core_radius);
    //printf("approximate number of MT points : %i\n", a.radial_grid().index_of(2.0));
    dict["rmt"]  = core_radius;
    dict["nrmt"] = nrmt;
    dict["rinf"] = rinf;

    /* compact representation of core states */
    std::string core_str;
    int         nl_core[8][4];
    std::memset(&nl_core[0][0], 0, 32 * sizeof(int));
    for (size_t i = 0; i < core.size(); i++) {
        std::stringstream ss;
        if (!nl_core[core[i].n][core[i].l]) {
            ss << core[i].n;
            core_str += (ss.str() + level_symb[core[i].l]);
            nl_core[core[i].n][core[i].l] = 1;
        }
    }
    dict["core"] = core_str;

    dict["valence"] = json::array();
    dict["valence"].push_back(json::object({{"basis", json::array({})}}));
    dict["valence"][0]["basis"].push_back({{"enu", apw_enu}, {"dme", 0}, {"auto", 0}});
    if (apw_order == 2) {
        dict["valence"][0]["basis"].push_back({{"enu", apw_enu}, {"dme", 1}, {"auto", 0}});
    }

    if (auto_enu) {
        for (int l = 0; l <= lmax; l++) {
            /* default value for n */
            int n = l + 1;
            /* next n above the core */
            for (size_t i = 0; i < core.size(); i++) {
                if (core[i].l == l) {
                    n = std::max(core[i].n + 1, n);
                }
            }
            /* APW for s,p,d,f is constructed for the highest valence state */
            for (size_t i = 0; i < valence.size(); i++) {
                if (valence[i].l == l) {
                    n = std::max(n, valence[i].n);
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
            if (nl_v[n][l]) {
                if (lo_type.find("lo1") != std::string::npos) {
                    a.add_lo_descriptor(idxlo, n, l, e_nl_v[n][l], 0, 1);
                    a.add_lo_descriptor(idxlo, n, l, e_nl_v[n][l], 1, 1);
                    idxlo++;
                }

                if (lo_type.find("lo2") != std::string::npos) {
                    a.add_lo_descriptor(idxlo, n, l, e_nl_v[n][l], 1, 1);
                    a.add_lo_descriptor(idxlo, n, l, e_nl_v[n][l], 2, 1);
                    idxlo++;
                }

                if (lo_type.find("LO1") != std::string::npos) {
                    a.add_lo_descriptor(idxlo, 0, l, 0.15, 0, 0);
                    a.add_lo_descriptor(idxlo, 0, l, 0.15, 1, 0);
                    a.add_lo_descriptor(idxlo, n, l, e_nl_v[n][l], 0, 1);
                    idxlo++;
                }

                if (lo_type.find("LO2") != std::string::npos) {
                    a.add_lo_descriptor(idxlo, 0, l, 1.15, 0, 0);
                    a.add_lo_descriptor(idxlo, 0, l, 1.15, 1, 0);
                    a.add_lo_descriptor(idxlo, n + 1, l, e_nl_v[n][l] + 1, 0, 1);
                    idxlo++;
                }
            }
        }
    }

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

    if (lo_type.find("lo4") != std::string::npos) {
        for (int n = 1; n <= 6; n++) {
            for (int l = 0; l < n; l++) {
                a.add_lo_descriptor(idxlo, n, l, 0.15, 0, 1);
                a.add_lo_descriptor(idxlo, n, l, 0.15, 1, 1);
                idxlo++;
            }
        }
    }

    std::vector<double> fa_rho(a.radial_grid().num_points());
    std::vector<double> fa_r(a.radial_grid().num_points());

    for (int i = 0; i < a.radial_grid().num_points(); i++) {
        fa_rho[i] = a.free_atom_density(i);
        fa_r[i]   = a.radial_grid(i);
    }

    sirius::Radial_grid_lin_exp<double> rg(1500, rmin, 2.0);
    std::vector<double>                 x;
    std::vector<double>                 veff;
    for (int ir = 0; ir < rg.num_points(); ir++) {
        x.push_back(rg[ir]);
        veff.push_back(a.free_atom_potential(rg[ir]) - a.zn() * rg.x_inv(ir));
    }
    a.set_radial_grid(1500, &x[0]);

    std::printf("=== initializing atom ===\n");
    a.init(0);
    sirius::Atom_symmetry_class atom_class(0, a);
    atom_class.set_spherical_potential(veff);
    atom_class.generate_radial_functions(relativity_t::none);
    pstdout pout(Communicator::self());
    atom_class.write_enu(pout);
    pout.flush();

    auto lo_to_str = [](local_orbital_descriptor lod) {
        std::stringstream s;
        s << "[";
        for (size_t o = 0; o < lod.rsd_set.size(); o++) {
            s << "{"
              << "\"n\" : " << lod.rsd_set[o].n
              << ", \"enu\" : " << lod.rsd_set[o].enu
              << ", \"dme\" : " << lod.rsd_set[o].dme
              << ", \"auto\" : " << lod.rsd_set[o].auto_enu << "}";
            if (o != lod.rsd_set.size() - 1) {
                s << ", ";
            }
        }
        s << "]";
        return s.str();
    };

    //== std::vector<int> inc(a.num_lo_descriptors(), 1);

    //== mdarray<double, 2> rad_func(rg.num_points(), a.num_lo_descriptors());

    //== for (int i = 0; i < a.num_lo_descriptors(); i++) {
    //==     int idxrf = a.indexr().index_by_idxlo(i);
    //==     int l = a.lo_descriptor(i).l;

    //==     sirius::Spline<double> f(rg);
    //==     for (int ir = 0; ir < a.num_mt_points(); ir++) {
    //==         f[ir] = atom_class.radial_function(ir, idxrf);
    //==     }

    //==     sirius::Spline<double> s(rg);

    //==     for (int j = 0; j < i; j++) {
    //==         int l1 = a.lo_descriptor(j).l;

    //==         if (l1 == l && inc[j]) {
    //==             for (int ir = 0; ir < a.num_mt_points(); ir++) {
    //==                 s[ir] = f[ir] * rad_func(ir, j);
    //==             }
    //==             double ovlp = s.interpolate().integrate(2);
    //==             for (int ir = 0; ir < a.num_mt_points(); ir++) {
    //==                 f[ir] -= ovlp * rad_func(ir, j);
    //==             }
    //==         }
    //==     }
    //==     for (int ir = 0; ir < a.num_mt_points(); ir++) {
    //==         s[ir] = std::pow(f[ir], 2);
    //==     }
    //==     double norm = std::sqrt(s.interpolate().integrate(2));
    //==     std::printf("orbital: %i, norm: %f\n", i, norm);
    //==     if (norm < 0.05) {
    //==         auto s = lo_to_str(a.lo_descriptor(i));
    //==         std::printf("local orbital %i is linearly dependent\n", i);
    //==         std::printf("  l: %i, basis: %s\n", l, s.str().c_str());
    //==         inc[i] = 0;
    //==     } else {
    //==         norm = 1 / norm;
    //==         for (int ir = 0; ir < a.num_mt_points(); ir++) {
    //==             rad_func(ir, i) = f[ir] * norm;
    //==             //atom_class.radial_function(ir, idxrf) = rad_func(ir, i);
    //==         }
    //==     }
    //== }

    auto inc   = atom_class.check_lo_linear_independence(0.0001);
    dict["lo"] = json::array();

    for (int j = 0; j < atom_class.num_lo_descriptors(); j++) {
        auto s = lo_to_str(atom_class.lo_descriptor(j));
        if (!inc[j]) {
            std::printf("X ");
        } else {
            std::printf("  ");
            dict["lo"].push_back({{"l", atom_class.lo_descriptor(j).l}, {"basis", json::parse(s)}});
        }
        std::printf("l: %i, basis: %s\n", atom_class.lo_descriptor(j).l, s.c_str());
    }

    dict["free_atom"]                = json::object();
    dict["free_atom"]["density"]     = fa_rho;
    dict["free_atom"]["radial_grid"] = fa_r;

    std::ofstream ofs(a.symbol() + std::string(".json"), std::ofstream::out | std::ofstream::trunc);
    ofs << dict.dump(4);
    ofs.close();

    //if (write_to_xml)
    //{
    //    std::string fname = a.symbol() + std::string(".xml");
    //    FILE* fout = fopen(fname.c_str(), "w");
    //    fprintf(fout, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    //    fprintf(fout, "<spdb>\n");
    //    fprintf(fout, "  <sp chemicalSymbol=\"%s\" name=\"%s\" z=\"%f\" mass=\"%f\">\n", a.symbol().c_str(), a.name().c_str(), -1.0 * a.zn(), a.mass());
    //    fprintf(fout, "    <muffinTin rmin=\"%e\" radius=\"%f\" rinf=\"%f\" radialmeshPoints=\"%i\"/>\n", 1e-6, 2.0, rinf, 1000);

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
    //            fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //            fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //            fprintf(fout, "        </lo>\n");

    //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //            fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //            fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //            fprintf(fout, "        </lo>\n");
    //
    //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //            fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //            fprintf(fout, "          <wf matchingOrder=\"3\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //            fprintf(fout, "        </lo>\n");
    //        }
    //        if (n_v[l].size() > 1)
    //        {
    //            for (size_t i = 0; i < n_v[l].size() - 1; i++)
    //            {
    //                int n = n_v[l][i];
    //                int n1 = n_v[l][i + 1];

    //                fprintf(fout, "        <lo l=\"%i\">\n", l);
    //                fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //                fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //                fprintf(fout, "        </lo>\n");

    //                fprintf(fout, "        <lo l=\"%i\">\n", l);
    //                fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //                fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //                fprintf(fout, "        </lo>\n");
    //
    //                fprintf(fout, "        <lo l=\"%i\">\n", l);
    //                fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //                fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n1][l]);
    //                fprintf(fout, "        </lo>\n");
    //            }
    //            int n = n_v[l].back();
    //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //            fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //            fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //            fprintf(fout, "        </lo>\n");
    //        }

    //        //if (n_v[l].size())
    //        //    fprintf(fout, "        <custom l=\"%i\" type=\"lapw\" trialEnergy=\"%f\" searchE=\"false\"/>\n", l, e_nl_v[n_v[l].front()][l]);

    //        //for (int n = 1; n <= 7; n++)
    //        //{
    //        //    if (nl_v[n][l])
    //        //    {
    //        //        fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        //        fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //        //        fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //        //        fprintf(fout, "        </lo>\n");

    //        //        //if (e_nl_v[n][l] > -5.0)
    //        //        //{
    //        //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        //            fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //        //            fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //        //            fprintf(fout, "        </lo>\n");
    //        //        //}
    //        //
    //        //        //if (e_nl_v[n][l] > -1.0)
    //        //        //{
    //        //            fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        //            fprintf(fout, "          <wf matchingOrder=\"2\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //        //            fprintf(fout, "          <wf matchingOrder=\"3\" trialEnergy=\"%f\" searchE=\"false\"/>\n", e_nl_v[n][l]);
    //        //            fprintf(fout, "        </lo>\n");
    //        //        //}

    //        //        //fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        //        //fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"true\"/>\n", e_nl_v[n][l]);
    //        //        //fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"%f\" searchE=\"true\"/>\n", e_nl_v[n][l]);
    //        //        //fprintf(fout, "        </lo>\n");
    //        //        //
    //        //        //if (e_nl_v[n][l] < -1.0)
    //        //        //{
    //        //        //    fprintf(fout, "        <lo l=\"%i\">\n", l);
    //        //        //    fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"0.15\" searchE=\"false\"/>\n");
    //        //        //    fprintf(fout, "          <wf matchingOrder=\"1\" trialEnergy=\"0.15\" searchE=\"false\"/>\n");
    //        //        //    fprintf(fout, "          <wf matchingOrder=\"0\" trialEnergy=\"%f\" searchE=\"true\"/>\n", e_nl_v[n][l]);
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

int main(int argn, char** argv)
{
    sirius::initialize(true);

    /* handle command line arguments */
    cmd_args args;
    args.register_key("--symbol=", "{string} symbol of a chemical element");
    args.register_key("--type=", "{lo1, lo2, lo3, LO1, LO2} type of local orbital basis");
    args.register_key("--core=", "{double} cutoff for core states: energy (in Ha, if <0), radius (in a.u. if >0)");
    args.register_key("--order=", "{int} order of augmentation");
    args.register_key("--apw_enu=", "{double} default value for APW linearization energies");
    args.register_key("--auto_enu", "allow search of APW linearization energies");
    args.register_key("--xml", "xml output for Exciting code");
    args.register_key("--rel", "use scalar-relativistic solver");
    args.parse_args(argn, argv);

    if (argn == 1 || args.exist("help")) {
        std::printf("\n");
        std::printf("Atom (L)APW+lo basis generation.\n");
        std::printf("\n");
        std::printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        std::printf("\n");
        std::printf("Definition of the local orbital types:\n");
        std::printf("  lo  : 2nd order local orbitals composed of u(E) and udot(E),\n");
        std::printf("        where E is the energy of the bound-state level {n,l}\n");
        std::printf("  LO  : 3rd order local orbitals composed of u(E), udot(E) and u(E1),\n");
        std::printf("        where E and E1 are the energies of the bound-state levels {n,l} and {n+1,l}\n");
        std::printf("\n");
        std::printf("Examples:\n");
        std::printf("\n");
        std::printf("  generate default basis for lithium:\n");
        std::printf("    ./atom --symbol=Li\n");
        std::printf("\n");
        std::printf("  generate high precision basis for titanium:\n");
        std::printf("    ./atom --type=lo+LO --symbol=Ti\n");
        std::printf("\n");
        std::printf("  make all states of iron to be valence:\n");
        std::printf("    ./atom --core=-1000 --symbol=Fe\n");
        std::printf("\n");
        return 0;
    }

    auto symbol = args.value<std::string>("symbol");

    double core_cutoff = args.value<double>("core", -10.0);

    std::string lo_type = args.value<std::string>("type", "lo1");

    int apw_order = args.value<int>("order", 2);

    double apw_enu = args.value<double>("apw_enu", 0.15);

    bool auto_enu = args.exist("auto_enu");

    bool write_to_xml = args.exist("xml");

    bool rel = args.exist("rel");

    sirius::Simulation_parameters param;
    param.set_lmax_apw(-1);
    Free_atom a = init_atom_configuration(symbol, param);

    generate_atom_file(a, core_cutoff, lo_type, apw_order, apw_enu, auto_enu, write_to_xml, rel);

    sirius::finalize();
}
