// This file is part of SIRIUS
//
// Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
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

class atom : public sirius::Atom_type
{
    public:
    
        double NIST_LDA_Etot;
    
        atom(const char* symbol, const char* name, int zn, double mass, std::vector<sirius::atomic_level_descriptor>& levels_nl) : 
            Atom_type(symbol, name, zn, mass, levels_nl), NIST_LDA_Etot(0.0)
        {
        }
};

atom* init_atom_configuration(const std::string& label)
{
    JSON_tree jin("atoms.json");
    
    int nl_occ[7][4];
    sirius::atomic_level_descriptor nlk;
    sirius::atomic_level_descriptor nl;
    std::vector<sirius::atomic_level_descriptor> levels_nl;
    std::vector<sirius::atomic_level_descriptor> levels_nlk;
    
    atom* a;

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
    
    a = new atom(label.c_str(), name.c_str(), zn, mass, levels_nl);
    a->NIST_LDA_Etot = NIST_LDA_Etot;
    return a;
}

void solve_atom(atom* a, double core_cutoff_energy, const std::string& lo_type)
{
    std::vector<double> enu;
    
    double energy_tot = a->solve_free_atom(1e-10, 1e-8, 1e-7, enu);
    
    int ncore = 0;
    for (int ist = 0; ist < (int)a->num_atomic_levels(); ist++)
    {
        if (enu[ist] < core_cutoff_energy) ncore += int(a->atomic_level(ist).occupancy + 1e-12);
    }

    std::cout << " atom : " << a->symbol() << "    Z : " << a->zn() << std::endl;
    std::cout << " =================== " << std::endl;
    printf(" total energy : %12.6f, NIST value : %12.6f\n", energy_tot, a->NIST_LDA_Etot);
    std::cout << " number of core electrons : " <<  ncore << std::endl;
    std::cout << std::endl;
  
    double dE = double(int64_t(fabs(energy_tot - a->NIST_LDA_Etot) * 1e8)) / 1e8;
    std::cerr << a->zn() << " " << dE << std::endl;
    
    std::string fname = a->symbol() + std::string(".json");
    JSON_write jw(fname);
    jw.single("name", a->name());
    jw.single("symbol", a->symbol());
    jw.single("number", a->zn());
    jw.single("mass", a->mass());
    jw.single("rmin", a->radial_grid(0));
    jw.single("rmax", a->radial_grid(a->radial_grid().size() - 1));
    jw.single("nrmt", a->num_mt_points());

    std::vector<sirius::atomic_level_descriptor> core;
    std::vector<sirius::atomic_level_descriptor> valence;
    std::string level_symb[] = {"s", "p", "d", "f"};
    
    printf("Core / valence partitioning\n");
    printf("core cutoff energy : %f\n", core_cutoff_energy);
    sirius::Spline <double> rho_c(a->radial_grid().size(), a->radial_grid());
    sirius::Spline <double> rho(a->radial_grid().size(), a->radial_grid());
    for (int ist = 0; ist < (int)a->num_atomic_levels(); ist++)
    {
        printf("%i%s  occ : %8.4f  energy : %12.6f", a->atomic_level(ist).n, level_symb[a->atomic_level(ist).l].c_str(), 
                                                     a->atomic_level(ist).occupancy, enu[ist]);
        
        // total density
        for (int ir = 0; ir < a->radial_grid().size(); ir++) 
            rho[ir] += a->atomic_level(ist).occupancy * pow(y00 * a->free_atom_radial_function(ir, ist), 2);

        if (enu[ist] < core_cutoff_energy)
        {
            core.push_back(a->atomic_level(ist));
            printf("  => core \n");

            for (int ir = 0; ir < a->radial_grid().size(); ir++) 
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

    // estimate effective infinity
    double rinf = 0.0;
    for (int ir = 0; ir < a->radial_grid().size(); ir++)
    {
        rinf = a->radial_grid(ir);
        if (rinf > 5.0 && (rho[ir] * rinf * rinf) < 1e-7) break;
    }
    printf("Effective infinity : %f\n", rinf);

    std::vector<double> g;
    rho_c.interpolate();
    rho_c.integrate(g, 2);

    double core_radius = 2.0;
    if (ncore != 0)
    {
        for (int ir = a->radial_grid().size() - 1; ir >= 0; ir--)
        {
            //if (fourpi * fabs(g[ir] - g[a->radial_grid().size() - 1]) > 1e-5) 
            if (fabs(g[ir] - g[a->radial_grid().size() - 1]) / fabs(g[a->radial_grid().size() - 1]) > 1e-5) 
            {
                core_radius = a->radial_grid(ir);
                break;
            }
        }
    }

    printf("suggested MT radius : %f\n", core_radius);
    jw.single("rmt", core_radius);
    
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
    jw.string("basis", "[{\"enu\" : 0.15, \"dme\" : 0, \"auto\" : 0}, {\"enu\" : 0.15, \"dme\" : 1, \"auto\" : 0}]");
    jw.end_set();
    
    int lmax = 0;
    for (int i = 0; i < (int)valence.size(); i++) lmax = std::max(lmax, valence[i].l); 
    lmax = std::min(lmax + 1, 3);
    int nmax[4];
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
        jw.string("basis", "[{\"enu\" : 0.15, \"dme\" : 0, \"auto\" : 1}, {\"enu\" : 0.15, \"dme\" : 1, \"auto\" : 1}]");

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
                  << " {" << "\"n\" : " << nmax[l] + nn + 1 << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : 2}]";
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
}

int main(int argn, char **argv)
{
    Platform::initialize(true);

    if (argn == 1)
    {
        printf("\n");
        printf("Atom (L)APW+lo basis generation\n");
        printf("\n");
        printf("Usage: ./atom [OPTIONS] symbol\n");
        printf("  [OPTIONS]\n");
        printf("    -type lo_type   set type of local orbital basis\n");
        printf("\n");
        printf("                    the following types are allowed: 'lo', 'lo+LO', 'lo+SLO', 'lo+cp'\n");
        printf("\n");
        printf("                    Definition:\n");
        printf("                      'lo'  : 2nd order local orbitals composed of u(E) and udot(E),\n");
        printf("                              where E is the energy of the bound-state level {n,l}\n");
        printf("                      'LO'  : 3rd order local orbitals composed of u(E), udot(E) and u(E1),\n");
        printf("                              where E and E1 are the energies of the bound-state levels {n,l} and {n+1,l}\n");
        printf("                      'SLO' : sequence of 3rd order local orbitals composed of u(E), udot(E) and u(En),\n");
        printf("                              where E is fixed and En is chosen in such a way that u(En) has n nodes inside the muffin-tin\n");
        printf("                      'cp'  : confined polynomial of the form r^{l}*(1-r/R)^{p}\n");
        printf("\n");
        printf("                    default is 'lo'\n");
        printf("\n");
        printf("    -core energy    set the cutoff energy for the core states\n");
        printf("\n");
        printf("                    default is -10.0 Ha\n");
        printf("\n");
        printf("Examples:\n");
        printf("\n");
        printf("  generate default basis for lithium:\n");
        printf("    ./atom Li\n"); 
        printf("\n");
        printf("  generate high precision basis for oxygen:\n");
        printf("    ./atom -type lo+SLO O\n"); 
        printf("\n");
        printf("  make all states of iron to be valence:\n");
        printf("    ./atom -core -1000 Fe\n"); 
        return -1;
    }

    double core_cutoff_energy = -10.0;
    std::string lo_type = "lo";
   
    std::string label = "";
    int i = 1;
    while (i < argn)
    {
        std::string s(argv[i]);
        if (s == "-type")
        {
            lo_type = std::string(argv[i + 1]);
            if (!(lo_type == "lo" || lo_type == "lo+LO" || lo_type == "lo+SLO" || lo_type == "lo+cp"))
                error_local(__FILE__, __LINE__, "wrong type of local orbital basis");
            i += 2;
        }
        else if (s == "-core")
        {
            std::istringstream iss((std::string(argv[i + 1])));
            iss >> core_cutoff_energy;
            i += 2;
        }
        else 
        {
            label = s;
            i++;
        }
    }
    if (label == "") error_local(__FILE__, __LINE__, "atom symbol was not specified");
    
    atom* a = init_atom_configuration(label);
    
    solve_atom(a, core_cutoff_energy, lo_type);

    delete a;
}
