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

#include <sirius.h>

class Free_atom : public sirius::Atom_type
{
    private:

        mdarray<double, 2> free_atom_radial_functions_;

    public:
    
        double NIST_LDA_Etot;
    
        Free_atom(const char* symbol, 
                  const char* name, 
                  int zn, 
                  double mass, 
                  std::vector<atomic_level_descriptor>& levels_nl) 
            : Atom_type(symbol, name, zn, mass, levels_nl, scaled_pow_grid), 
              NIST_LDA_Etot(0.0)
        {
            radial_grid_ = sirius::Radial_grid(scaled_pow_grid, 2000 + 150 * zn, 1e-7, 20.0 + 0.25 * zn); 
        }

        double ground_state(double solver_tol, double energy_tol, double charge_tol, std::vector<double>& enu)
        {
            sirius::Timer t("sirius::Free_atom::ground_state");
        
            int np = radial_grid().num_points();
            assert(np > 0);

            free_atom_radial_functions_ = mdarray<double, 2>(np, num_atomic_levels()); 
            
            sirius::Radial_solver solver(false, -1.0 * zn(), radial_grid());
        
            sirius::XC_functional Ex("XC_LDA_X", 1);
            sirius::XC_functional Ec("XC_LDA_C_VWN", 1);
        
            solver.set_tolerance(solver_tol);
            
            std::vector<double> veff(np);
            std::vector<double> vnuc(np);
            for (int i = 0; i < np; i++)
            {
                vnuc[i] = -1.0 * zn() / radial_grid(i);
                veff[i] = vnuc[i];
            }
        
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
            double energy_xc = 0;
            double energy_kin = 0;
            double energy_coul = 0;
        
            double beta = 0.9;
            
            bool converged = false;
            
            /* starting values for E_{nu} */
            for (int ist = 0; ist < num_atomic_levels(); ist++)
                enu[ist] = -1.0 * zn() / 2 / pow(double(atomic_level(ist).n), 2);
            
            for (int iter = 0; iter < 200; iter++)
            {
                rho_old = rho.values();
                
                memset(&rho[0], 0, rho.num_points() * sizeof(double));
                #pragma omp parallel default(shared)
                {
                    std::vector<double> p(rho.num_points());
                    std::vector<double> rho_t(rho.num_points());
                    memset(&rho_t[0], 0, rho.num_points() * sizeof(double));
                
                    #pragma omp for
                    for (int ist = 0; ist < num_atomic_levels(); ist++)
                    {
                        enu[ist] = solver.bound_state(atomic_level(ist).n, atomic_level(ist).l, enu[ist], veff, p);
                    
                        for (int i = 0; i < np; i++)
                        {
                            free_atom_radial_functions_(i, ist) = p[i] * radial_grid().x_inv(i);
                            rho_t[i] += atomic_level(ist).occupancy * pow(y00 * free_atom_radial_functions_(i, ist), 2);
                        }
                    }
        
                    #pragma omp critical
                    for (int i = 0; i < rho.num_points(); i++) rho[i] += rho_t[i];
                } 
                
                charge_rms = 0.0;
                for (int i = 0; i < np; i++) charge_rms += pow(rho[i] - rho_old[i], 2);
                charge_rms = sqrt(charge_rms / np);
                
                rho.interpolate();
        
                /* compute Hartree potential */
                rho.integrate(g2, 2);
                double t1 = rho.integrate(g1, 1);
        
                for (int i = 0; i < np; i++) vh[i] = fourpi * (g2[i] / radial_grid(i) + t1 - g1[i]);
                
                /* compute XC potential and energy */
                Ex.get_lda(rho.num_points(), &rho[0], &vx[0], &ex[0]);
                Ec.get_lda(rho.num_points(), &rho[0], &vc[0], &ec[0]);
                for (int ir = 0; ir < rho.num_points(); ir++)
                {
                   vxc[ir] = (vx[ir] + vc[ir]);
                   exc[ir] = (ex[ir] + ec[ir]);
                }
               
                /* mix old and new effective potential */
                for (int i = 0; i < np; i++)
                    veff[i] = (1 - beta) * veff[i] + beta * (vnuc[i] + vh[i] + vxc[i]);
                
                /* sum of occupied eigen values */
                double eval_sum = 0.0;
                for (int ist = 0; ist < num_atomic_levels(); ist++)
                    eval_sum += atomic_level(ist).occupancy * enu[ist];
        
                for (int i = 0; i < np; i++) f[i] = (veff[i] - vnuc[i]) * rho[i];
                /* kinetic energy */
                energy_kin = eval_sum - fourpi * (f.interpolate().integrate(2) - zn() * rho.integrate(1));
                
                /* XC energy */
                for (int i = 0; i < np; i++) f[i] = exc[i] * rho[i];
                energy_xc = fourpi * f.interpolate().integrate(2); 
                
                /* electron-nuclear energy: \int vnuc(r) * rho(r) r^2 dr */
                energy_enuc = -fourpi * zn() * rho.integrate(1); 
        
                /* Coulomb energy */
                for (int i = 0; i < np; i++) f[i] = vh[i] * rho[i];
                energy_coul = 0.5 * fourpi * f.interpolate().integrate(2);
                
                energy_tot_old = energy_tot;
        
                energy_tot = energy_kin + energy_xc + energy_coul + energy_enuc; 
                
                energy_diff = fabs(energy_tot - energy_tot_old);
                
                if (energy_diff < energy_tol && charge_rms < charge_tol) 
                { 
                    converged = true;
                    printf("Converged in %i iterations.\n", iter);
                    break;
                }
                
                beta = std::max(beta * 0.95, 0.005);
            }
        
            if (!converged)
            {
                printf("energy_diff : %18.10f   charge_rms : %18.10f   beta : %18.10f\n", energy_diff, charge_rms, beta);
                std::stringstream s;
                s << "atom " << symbol() << " is not converged" << std::endl
                  << "  energy difference : " << energy_diff << std::endl
                  << "  charge difference : " << charge_rms;
                error_local(__FILE__, __LINE__, s);
            }
            
            free_atom_density_ = rho.values();
            
            free_atom_potential_ = veff;

            printf("\n");
            printf("Radial gird\n");
            printf("-----------\n");
            printf("type             : %s\n", radial_grid().grid_type_name().c_str());
            printf("number of points : %i\n", np);
            printf("origin           : %20.12f\n", radial_grid(0));
            printf("infinity         : %20.12f\n", radial_grid(np - 1));
            printf("\n");
            printf("Energy\n");
            printf("------\n");
            printf("Ekin  : %20.12f\n", energy_kin);
            printf("Ecoul : %20.12f\n", energy_coul);
            printf("Eenuc : %20.12f\n", energy_enuc);
            printf("Eexc  : %20.12f\n", energy_xc);
            printf("Total : %20.12f\n", energy_tot);
            printf("NIST  : %20.12f\n", NIST_LDA_Etot);

            /* difference between NIST and computed total energy. Comparison is valid only for VWN XC functional. */
            double dE = double(int64_t(fabs(energy_tot - NIST_LDA_Etot) * 1e8)) / 1e8;
            if (dE < 5e-7) dE = 0;
            std::cerr << zn() << " " << dE << " # " << symbol() << std::endl;
            
            return energy_tot;
        }

        inline double free_atom_radial_function(int ir, int ist)
        {
            return free_atom_radial_functions_(ir, ist);
        }
};

Free_atom* init_atom_configuration(const std::string& label)
{
    JSON_tree jin("atoms.json");
    
    int nl_occ[7][4];
    atomic_level_descriptor nlk;
    atomic_level_descriptor nl;
    std::vector<atomic_level_descriptor> levels_nl;
    std::vector<atomic_level_descriptor> levels_nlk;
    
    Free_atom* a;

    memset(&nl_occ[0][0], 0, 28 * sizeof(int));

    for (int i = 0; i < jin[label]["levels"].size(); i++)
    {
        jin[label]["levels"][i][0] >> nlk.n;
        jin[label]["levels"][i][1] >> nlk.l;
        jin[label]["levels"][i][2] >> nlk.k;
        jin[label]["levels"][i][3] >> nlk.occupancy;
        nl_occ[nlk.n - 1][nlk.l] += int(nlk.occupancy + 1e-12);
        levels_nlk.push_back(nlk);
    }

    for (int n = 0; n < 7; n++) 
    { 
        for (int l = 0; l < 4; l++) 
        { 
            if (nl_occ[n][l]) 
            { 
                nl.n = n + 1; 
                nl.l = l; 
                nl.occupancy = nl_occ[n][l]; 
                levels_nl.push_back(nl);
            } 
        } 
    }
    int zn;
    jin[label]["zn"] >> zn;
    double mass;
    jin[label]["mass"] >> mass;
    std::string name;
    jin[label]["name"] >> name;
    double NIST_LDA_Etot = 0.0;
    NIST_LDA_Etot = jin[label]["NIST_LDA_Etot"].get(NIST_LDA_Etot);
    
    a = new Free_atom(label.c_str(), name.c_str(), zn, mass, levels_nl);
    a->NIST_LDA_Etot = NIST_LDA_Etot;
    return a;
}

void generate_atom_file(Free_atom* a, double core_cutoff_energy, const std::string& lo_type, int apw_order)
{
    std::vector<double> enu;
    
    printf("\n");
    printf("atom : %s, Z = %i\n", a->symbol().c_str(), a->zn());
    printf("----------------------------------\n");
   
    /* solve a free atom */
    a->ground_state(1e-10, 1e-8, 1e-7, enu);
   
    /* find number of core states */
    int ncore = 0;
    for (int ist = 0; ist < (int)a->num_atomic_levels(); ist++)
    {
        if (enu[ist] < core_cutoff_energy) ncore += int(a->atomic_level(ist).occupancy + 1e-12);
    }

    std::string fname = a->symbol() + std::string(".json");
    JSON_write jw(fname);
    jw.single("name", a->name());
    jw.single("symbol", a->symbol());
    jw.single("number", a->zn());
    jw.single("mass", a->mass());
    jw.single("rmin", a->radial_grid(0));

    std::vector<atomic_level_descriptor> core;
    std::vector<atomic_level_descriptor> valence;
    std::string level_symb[] = {"s", "p", "d", "f"};
    
    printf("\n");
    printf("Core / valence partitioning\n");
    printf("---------------------------\n");
    printf("core cutoff energy       : %f\n", core_cutoff_energy);
    printf("number of core electrons : %i\n", ncore);
    sirius::Spline <double> rho_c(a->radial_grid());
    sirius::Spline <double> rho(a->radial_grid());
    for (int ist = 0; ist < a->num_atomic_levels(); ist++)
    {
        printf("%i%s  occ : %8.4f  energy : %12.6f", a->atomic_level(ist).n, level_symb[a->atomic_level(ist).l].c_str(), 
                                                     a->atomic_level(ist).occupancy, enu[ist]);
        
        /* total density */
        for (int ir = 0; ir < a->radial_grid().num_points(); ir++) 
            rho[ir] += a->atomic_level(ist).occupancy * pow(y00 * a->free_atom_radial_function(ir, ist), 2);

        if (enu[ist] < core_cutoff_energy)
        {
            core.push_back(a->atomic_level(ist));
            printf("  => core \n");

            for (int ir = 0; ir < a->radial_grid().num_points(); ir++) 
                rho_c[ir] += a->atomic_level(ist).occupancy * pow(y00 * a->free_atom_radial_function(ir, ist), 2);
        }
        else
        {
            valence.push_back(a->atomic_level(ist));
            printf("  => valence\n");
        }
    }

    //** FILE* fout = fopen("rho.dat", "w");
    //** for (int ir = 0; ir < a->radial_grid().size(); ir++) 
    //** {
    //**     double x = a->radial_grid(ir);
    //**     fprintf(fout, "%12.6f %16.8f\n", x, rho[ir] * x * x);
    //** }
    //** fclose(fout);

    /* estimate effective infinity */
    double rinf = 0.0;
    for (int ir = 0; ir < a->radial_grid().num_points(); ir++)
    {
        rinf = a->radial_grid(ir);
        if (rinf > 5.0 && (rho[ir] * rinf * rinf) < 1e-7) break;
    }
    printf("Effective infinity : %f\n", rinf);

    std::vector<double> g;
    rho_c.interpolate().integrate(g, 2);

    double core_radius = 2.0;
    int nrmt = 1500;
    if (ncore != 0)
    {
        for (int ir = a->radial_grid().num_points() - 1; ir >= 0; ir--)
        {
            //if (fourpi * fabs(g[ir] - g[a->radial_grid().size() - 1]) > 1e-5) 
            if (fabs(g[ir] - g[a->radial_grid().num_points() - 1]) / fabs(g[a->radial_grid().num_points() - 1]) > 1e-5) 
            {
                core_radius = a->radial_grid(ir);
                nrmt = ir;
                break;
            }
        }
    }

    printf("minimum MT radius : %f\n", core_radius);
    jw.single("rmt", core_radius);
    jw.single("nrmt", nrmt);
    
    std::string core_str;
    for (int i = 0; i < (int)core.size(); i++)
    {
        std::stringstream ss;
        ss << core[i].n;
        core_str += (ss.str() + level_symb[core[i].l]);
    }
    jw.single("core", core_str);
    jw.begin_array("valence");
    jw.begin_set();
    if (apw_order == 1)
    {
        jw.string("basis", "[{\"enu\" : 0.15, \"dme\" : 0, \"auto\" : 0}]");
    }
    if (apw_order == 2)
    {
        jw.string("basis", "[{\"enu\" : 0.15, \"dme\" : 0, \"auto\" : 0}, {\"enu\" : 0.15, \"dme\" : 1, \"auto\" : 0}]");
    }
    jw.end_set();
    
    int lmax = 0;
    for (int i = 0; i < (int)valence.size(); i++) lmax = std::max(lmax, valence[i].l); 
    lmax = std::min(lmax + 1, 3);
    //lmax = 8;
    int nmax[9];
    for (int l = 0; l <= lmax; l++)
    {
        int n = l + 1;
        
        for (int i = 0; i < (int)core.size(); i++) 
        {
            if (core[i].l == l) n = core[i].n + 1;
        }
        
        for (int i = 0; i < (int)valence.size(); i++)
        {
            if (valence[i].l == l) n = valence[i].n;
        }
        nmax[l] = n;
               
        jw.begin_set();
        jw.single("l", l);
        jw.single("n", n);
        if (apw_order == 1)
        {
            jw.string("basis", "[{\"enu\" : 0.15, \"dme\" : 0, \"auto\" : 1}]");
        }
        if (apw_order == 2)
        {
            jw.string("basis", "[{\"enu\" : 0.15, \"dme\" : 0, \"auto\" : 1}, {\"enu\" : 0.15, \"dme\" : 1, \"auto\" : 1}]");
        }
        jw.end_set();
    }
    jw.end_array();
    jw.begin_array("lo");
    for (int i = 0; i < (int)valence.size(); i++)
    {
        jw.begin_set();
        std::stringstream s;
        s << "[{" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : 1}," 
          << " {" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 1, \"auto\" : 1}]";
        jw.single("l", valence[i].l);
        jw.string("basis", s.str());
        jw.end_set();
    }

    if (lo_type == "lo+SLO")
    {
        for (int l = 0; l <= lmax; l++)
        {
            for (int nn = 0; nn < 10; nn++)
            {
                jw.begin_set();
                std::stringstream s;
                s << "[{" << "\"n\" : " << nmax[l] + nn + 1 << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : 0}," 
                  << " {" << "\"n\" : " << nmax[l] + nn + 1 << ", \"enu\" : 0.15, \"dme\" : 1, \"auto\" : 0},"
                  << " {" << "\"n\" : " << nmax[l] + nn + 1 << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : 1}]";
                jw.single("l", l);
                jw.string("basis", s.str());
                jw.end_set();
            }
        }
    }
    if (lo_type == "lo+LO")
    {
        for (int i = 0; i < (int)valence.size(); i++)
        {
            jw.begin_set();
            std::stringstream s;
            s << "[{" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : 1}," 
              << " {" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 1, \"auto\" : 1}," 
              << " {" << "\"n\" : " << valence[i].n + 1 << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : 1}]";
            jw.single("l", valence[i].l);
            jw.string("basis", s.str());
            jw.end_set();
        }
    }
    if (lo_type == "lo+cp")
    {
        for (int l = 0; l <= lmax; l++)
        {
            jw.begin_set();
            jw.single("l", l);
            std::stringstream s;
            s << "{ \"p1\": [" << l << "], \"p2\" : [1,2,3,4,5]}";
            jw.string("polynom", s.str());
            jw.end_set();
        }
    }
    jw.end_array();

    std::vector<double> fa_rho(a->radial_grid().num_points());
    std::vector<double> fa_v(a->radial_grid().num_points());
    std::vector<double> fa_r(a->radial_grid().num_points());

    for (int i = 0; i < a->radial_grid().num_points(); i++)
    {
        fa_rho[i] = a->free_atom_density(i);
        fa_v[i] = a->free_atom_potential(i);
        //fa_v[i] = a->free_atom_potential(i) + a->zn() / a->radial_grid(i);
        fa_r[i] = a->radial_grid(i);
    }

    jw.begin_set("free_atom");
    jw.single("density", fa_rho);
    jw.single("potential", fa_v);
    jw.single("radial_grid", fa_r);
    jw.end_set();
}

int main(int argn, char **argv)
{
    Platform::initialize(true);

    /* handle command line arguments */
    cmd_args args;
    args.register_key("--symbol=", "{string} symbol of a chemical element");
    args.register_key("--type=", "{lo, lo+LO, lo+SLO, lo+cp} type of local orbital basis");
    args.register_key("--core=", "{double} cutoff energy (in Ha) for the core states");
    args.register_key("--order=", "{int} order of augmentation");
    args.parse_args(argn, argv);
    
    if (argn == 1)
    {
        printf("\n");
        printf("Atom (L)APW+lo basis generation.\n");
        printf("\n");
        printf("Usage: ./plot [options] \n");
        args.print_help();
        printf("\n");
        printf("Definition of the local orbital types:\n");
        printf("  lo  : 2nd order local orbitals composed of u(E) and udot(E),\n");
        printf("        where E is the energy of the bound-state level {n,l}\n");
        printf("  LO  : 3rd order local orbitals composed of u(E), udot(E) and u(E1),\n");
        printf("        where E and E1 are the energies of the bound-state levels {n,l} and {n+1,l}\n");
        printf("  SLO : sequence of 3rd order local orbitals composed of u(E), udot(E) and u(En),\n");
        printf("        where E is fixed and En is chosen in such a way that u(En) has n nodes inside the muffin-tin\n");
        printf("  cp  : confined polynomial of the form r^{l}*(1-r/R)^{p}\n");
        printf("\n");
        printf("Examples:\n");
        printf("\n");
        printf("  generate default basis for lithium:\n");
        printf("    ./atom --symbol=Li\n"); 
        printf("\n");
        printf("  generate high precision basis for titanium:\n");
        printf("    ./atom --type=lo+SLO --symbol=Ti\n"); 
        printf("\n");
        printf("  make all states of iron to be valence:\n");
        printf("    ./atom --core=-1000 --symbol=Fe\n"); 
        printf("\n");
        exit(0);
    }

    auto symbol = args.value<std::string>("symbol");

    double core_cutoff_energy = -10.0;
    if (args.exist("core")) core_cutoff_energy = args.value<double>("core");

    std::string lo_type = "lo";
    if (args.exist("type")) lo_type = args.value<std::string>("type");

    int apw_order = 1;
    if (args.exist("order")) apw_order = args.value<int>("order");
   
    Free_atom* a = init_atom_configuration(symbol);
    
    generate_atom_file(a, core_cutoff_energy, lo_type, apw_order);

    delete a;
}
