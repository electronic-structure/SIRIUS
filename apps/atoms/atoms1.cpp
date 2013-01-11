
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

void solve_atom(atom* a, char type)
{
    std::vector<double> enu;
    
    double energy_tot = a->solve_free_atom(1e-10, 1e-7, 1e-6, enu);
    
    double ecore_cutoff = -10.0;
    
    int ncore = 0;
    for (int ist = 0; ist < (int)a->num_atomic_levels(); ist++)
        if (enu[ist] < ecore_cutoff)
            ncore += a->atomic_level(ist).occupancy;

    std::cout << " atom : " << a->symbol() << "    Z : " << a->zn() << std::endl;
    std::cout << " =================== " << std::endl;
    std::cout << " total energy : " << energy_tot << ", NIST value : " <<  a->NIST_LDA_Etot 
              << ", difference : " << fabs(energy_tot - a->NIST_LDA_Etot) << std::endl;
    std::cout << " number of core electrons : " <<  ncore << std::endl;
    std::cout << std::endl;
    
    std::string fname = a->symbol() + std::string(".json");
    std::ofstream fout(fname.c_str());
    fout << "{" << std::endl;
    fout << "  \"name\"    : \"" << a->name() << "\"," << std::endl;
    fout << "  \"symbol\"  : \"" << a->symbol() << "\"," << std::endl;
    fout << "  \"number\"  : " << a->zn() << "," << std::endl;
    fout << "  \"mass\"    : " << a->mass() << "," << std::endl;
    fout << "  \"rmin\"    : " << a->radial_grid()[0] << "," << std::endl;
    fout << "  \"rmax\"    : " << a->radial_grid()[a->radial_grid().size() - 1] << "," << std::endl;
    fout << "  \"rmt\"     : " << a->mt_radius() << "," << std::endl;
    fout << "  \"nrmt\"    : " << a->num_mt_points() << "," << std::endl;
    
    std::vector<sirius::atomic_level_descriptor> core;
    std::vector<sirius::atomic_level_descriptor> valence;
    for (int ist = 0; ist < (int)a->num_atomic_levels(); ist++)
    {
        if (enu[ist] < ecore_cutoff)
            core.push_back(a->atomic_level(ist));
        else
            valence.push_back(a->atomic_level(ist));
    }

    printf("Core levels : \n");
    for (int i = 0; i < (int)core.size(); i++) printf("%i %i %i\n", core[i].n, core[i].l, core[i].occupancy);

    printf("Valence levels : \n");
    for (int i = 0; i < (int)valence.size(); i++) printf("%i %i %i\n", valence[i].n, valence[i].l, valence[i].occupancy);

    std::string symb[] = {"s", "p", "d", "f"};
    std::string core_str;
    for (int i = 0; i < (int)core.size(); i++)
    {
        std::stringstream ss;
        ss << core[i].n;
        core_str += (ss.str() + symb[core[i].l]);
    }
    fout << "  \"core\"    : \"" << core_str << "\", " << std::endl;
    
    fout << "  \"valence\" : [" << std::endl;
    fout << "    {\"basis\" : [{\"enu\" : 0.15, \"dme\" : 0, \"auto\" : false}]}";
    
    int lmax = 0;
    for (int i = 0; i < (int)valence.size(); i++) lmax = std::max(lmax, valence[i].l); 
    lmax = std::min(lmax + 1, 4);
    for (int l = 0; l <= lmax; l++)
    {
        int n = l + 1;
        
        for (int i = 0; i < (int)core.size(); i++) 
            if (core[i].l == l) 
                n = core[i].n + 1;
        
        for (int i = 0; i < (int)valence.size(); i++)
            if (valence[i].l == l) 
                n = valence[i].n;
               
        fout << "," << std::endl 
             << "    {\"l\" : " << l << ", \"n\" : " << n << ", \"basis\" : [{\"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}]}";
    }
    fout << "]," << std::endl;
    
    fout << "  \"lo\"      : [";
    for (int i = 0; i < (int)valence.size(); i++)
    {
        if (i) fout << ",";
        fout << std::endl;
        fout << "    {\"l\" : " << valence[i].l
             << ", \"basis\" : [{" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}," 
             << " {" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 1, \"auto\" : true}]}";
    }
    if (type == '1')
    {
        printf("LO1 is composed of u_{E1} and u_{E2}, where E1 and E2 are energies of levels n and n+1 respectively\n");
        for (int i = 0; i < (int)valence.size(); i++)
        {
            fout << "," << std::endl;
            fout << "    {\"l\" : " << valence[i].l
                 << ", \"basis\" : [{" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}," 
                 << " {" << "\"n\" : " << valence[i].n + 1 << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}]}";
        }
    }
    if (type == '2')
    {
        printf("LO2 is composed of u_{E2}, udot_{E2} and u_{E1}, where E2 and E1 are energies of levels n+1 and n respectively\n");
        for (int i = 0; i < (int)valence.size(); i++)
        {
            fout << "," << std::endl;
            fout << "    {\"l\" : " << valence[i].l
                 << ", \"basis\" : [{" << "\"n\" : " << valence[i].n + 1 << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}," 
                 << " {" << "\"n\" : " << valence[i].n + 1 << ", \"enu\" : 0.15, \"dme\" : 1, \"auto\" : true}," 
                 << " {" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}]}";
        }
    }
    if (type == '3')
    {
        for (int i = 0; i < (int)valence.size(); i++)
        {
            fout << "," << std::endl;
            fout << "    {\"l\" : " << valence[i].l
                 << ", \"basis\" : [{" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}," 
                 << " {" << "\"n\" : " << valence[i].n << ", \"enu\" : 0.15, \"dme\" : 1, \"auto\" : true}," 
                 << " {" << "\"n\" : " << valence[i].n + 1 << ", \"enu\" : 0.15, \"dme\" : 0, \"auto\" : true}]}";
        }
    }

    fout << "]" << std::endl;
    fout<< "}" << std::endl;
    fout.close();
    
}

int main(int argn, char **argv)
{
    Platform::initialize(true);

    if (argn != 3)
    {
        printf("usage: ./atoms lo label\n");
        printf("where 'lo' is a type of local orbital basis (0: lo, 1: lo+LO1, 2: lo+LO2, 3: lo+LO3)\n");
        printf("and 'label' is an atom label (H, He, Li, etc.)\n");
        error(__FILE__, __LINE__, "stop");
    }
    
    atom* a = init_atom_configuration(argv[2]);
    
    solve_atom(a, argv[1][0]);

    delete a;
}
