
#include <sirius.h>

class atom : public sirius::AtomType
{
    public:
    
        double NIST_LDA_Etot;
    
        atom(const char* _symbol, 
             const char* _name, 
             int _zn, 
             double _mass, 
             std::vector<sirius::atomic_level_descriptor>& _levels_nl) : AtomType(_symbol, _name, _zn, _mass, _levels_nl),
                                                                         NIST_LDA_Etot(0.0)
        {
        }
};

atom* init_atom_configuration(const std::string& label)
{
    JsonTree jin("atoms.json");
    
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
        nl_occ[nlk.n - 1][nlk.l] += nlk.occupancy;
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
    a = new atom(label.c_str(), jin[label]["name"].get<std::string>().c_str(), jin[label]["zn"].get<int>(), 
                 jin[label]["mass"].get<double>(), levels_nl);
    a->NIST_LDA_Etot = jin[label]["NIST_LDA_Etot"].get<double>(); 
    return a;
}

void solve_atom(atom* a, double core_cutoff_energy, int lo_type)
{
    std::vector<double> enu;
    
    double energy_tot = a->solve_free_atom(1e-10, 1e-7, 1e-6, enu);
    
    int ncore = 0;
    for (int ist = 0; ist < (int)a->num_atomic_levels(); ist++)
    {
        if (enu[ist] < core_cutoff_energy) ncore += a->atomic_level(ist).occupancy;
    }

    std::cout << " atom : " << a->symbol() << "    Z : " << a->zn() << std::endl;
    std::cout << " =================== " << std::endl;
    std::cout << " total energy : " << energy_tot << ", NIST value : " <<  a->NIST_LDA_Etot 
              << ", difference : " << fabs(energy_tot - a->NIST_LDA_Etot) << std::endl;
    std::cout << " number of core electrons : " <<  ncore << std::endl;
    std::cout << std::endl;
    
    std::cerr << a->zn() << " " << fabs(energy_tot - a->NIST_LDA_Etot) << std::endl;
    
    std::string fname = a->symbol() + std::string(".json");
    json_write jw(fname);
    jw.single("name", a->name());
    jw.single("symbol", a->symbol());
    jw.single("number", a->zn());
    jw.single("mass", a->mass());
    jw.single("rmin", a->radial_grid()[0]);
    jw.single("rmax", a->radial_grid()[a->radial_grid().size() - 1]);
    //jw.single("rmt", a->mt_radius());
    jw.single("nrmt", a->num_mt_points());

    std::vector<sirius::atomic_level_descriptor> core;
    std::vector<sirius::atomic_level_descriptor> valence;
    std::string level_symb[] = {"s", "p", "d", "f"};
    
    printf("Core / valence partitioning\n");
    printf("core cutoff energy : %f\n", core_cutoff_energy);
    sirius::Spline <double> rho_c(a->radial_grid().size(), a->radial_grid());
    for (int ist = 0; ist < (int)a->num_atomic_levels(); ist++)
    {
        printf("%i%s  occ : %2i  energy : %12.6f", a->atomic_level(ist).n, level_symb[a->atomic_level(ist).l].c_str(), 
                                                   a->atomic_level(ist).occupancy, enu[ist]);
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
    jw.string("basis", "[{\"enu\" : 0.15, \"dme\" : 0, \"auto\" : false}]");
    jw.end_set();
    
    int lmax = 0;
    for (int i = 0; i < (int)valence.size(); i++) lmax = std::max(lmax, valence[i].l); 
    lmax = std::min(lmax + 1, 3);
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
               
        jw.begin_set();
        jw.single("l", l);
        jw.single("n", n);
        jw.string("basis", "[{\"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}]");

        jw.end_set();
    }
    jw.end_array();
    jw.begin_array("lo");
    for (int i = 0; i < (int)valence.size(); i++)
    {
        jw.begin_set();
        std::stringstream s;
        s << "[{" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}," 
          << " {" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 1, \"auto\" : true}]";
        jw.single("l", valence[i].l);
        jw.string("basis", s.str());
        jw.end_set();
    }
    if (lo_type == 1)
    {
        printf("LO1 is composed of u_{E1} and u_{E2}, where E1 and E2 are energies of levels n and n+1 respectively\n");
        for (int i = 0; i < (int)valence.size(); i++)
        {
            jw.begin_set();
            std::stringstream s;
            s << "[{" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}," 
              << " {" << "\"n\" : " << valence[i].n + 1 << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}]";
            jw.single("l", valence[i].l);
            jw.string("basis", s.str());
            jw.end_set();
        }
    }
    if (lo_type == 2)
    {
        printf("LO2 is composed of u_{E2}, udot_{E2} and u_{E1}, where E2 and E1 are energies of levels n+1 and n respectively\n");
        for (int i = 0; i < (int)valence.size(); i++)
        {
            jw.begin_set();
            std::stringstream s;
            s << "[{" << "\"n\" : " << valence[i].n + 1 << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}," 
              << " {" << "\"n\" : " << valence[i].n + 1 << ", \"enu\" : 0.15, \"dme\" : 1, \"auto\" : true}," 
              << " {" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}]";
            jw.single("l", valence[i].l);
            jw.string("basis", s.str());
            jw.end_set();
        }
    }
    if (lo_type == 3)
    {
        for (int i = 0; i < (int)valence.size(); i++)
        {
            jw.begin_set();
            std::stringstream s;
            s << "[{" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}," 
              << " {" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 1, \"auto\" : true}," 
              << " {" << "\"n\" : " << valence[i].n + 1 << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}]";
            jw.single("l", valence[i].l);
            jw.string("basis", s.str());
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
        printf("usage: atoms [OPTIONS] symbol\n");
        printf("  [OPTIONS]\n");
        printf("    -l T     generate local orbitals of type T (0: lo, 1: lo+LO1, 2: lo+LO2, 3: lo+LO3);\n");
        printf("             default is T = 0\n");
        printf("    -c E     set the cutoff energy for the core states; default is -10.0 Ha\n");
        printf("\n");
        printf("Example:\n");
        printf("atoms -l 1 -c -5 Eu\n"); 
        error(__FILE__, __LINE__, "stop");
    }

    double core_cutoff_energy = -10.0;
    int lo_type = 0;
    
    for (int i = 1; i < argn; i++)
    {
        std::string s(argv[i]);
        if (s == "-l")
        {
            std::string s(argv[i + 1]);
            std::istringstream iss(s);
            iss >> lo_type;
            i++;
        }
        if (s == "-c")
        {
            std::string s(argv[i + 1]);
            std::istringstream iss(s);
            iss >> core_cutoff_energy;
            i++;
        }
    }

    
    atom* a = init_atom_configuration(argv[argn - 1]);
    
    solve_atom(a, core_cutoff_energy, lo_type);

    delete a;
}
