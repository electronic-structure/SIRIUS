s1 = '''
#include "../../lib/sirius.h"

class atom : public sirius::AtomType
{
    public:
    
        std::vector<sirius::atomic_level_descriptor> levels_nlk_;
        
        double NIST_LDA_Etot;
    
        atom(const char* _symbol, 
             const char* _name, 
             int _zn, 
             double _mass, 
             std::vector<sirius::atomic_level_descriptor>& _levels_nl,
             std::vector<sirius::atomic_level_descriptor>& _levels_nlk) : AtomType(_symbol, _name, _zn, _mass, _levels_nl),
                                                                          levels_nlk_(_levels_nlk),
                                                                          NIST_LDA_Etot(0.0)
        {
        }
        
        inline int num_levels_nlk()
        {
            return levels_nlk_.size();
        }    
        
        inline sirius::atomic_level_descriptor& level_nlk(int idx)
        {
            return levels_nlk_[idx];
        }
};

std::vector<atom*> atoms; 

void init_atom_configuration();

void solve_atom(atom* a)
{
    std::vector<double> enu;
    
    double energy_tot = a->solve_free_atom(1e-10, 1e-7, 1e-6, enu);
    
    double ecore_cutoff = -4.0;
    
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
    fout << "  \\"name\\"    : \\"" << a->name() << "\\"," << std::endl;
    fout << "  \\"symbol\\"  : \\"" << a->symbol() << "\\"," << std::endl;
    fout << "  \\"number\\"  : " << a->zn() << "," << std::endl;
    fout << "  \\"mass\\"    : " << a->mass() << "," << std::endl;
    fout << "  \\"rmin\\"    : " << a->radial_grid()[0] << "," << std::endl;
    fout << "  \\"rmax\\"    : " << a->radial_grid()[a->radial_grid().size() - 1] << "," << std::endl;
    fout << "  \\"rmt\\"     : " << a->mt_radius() << "," << std::endl;
    fout << "  \\"nrmt\\"    : " << a->num_mt_points() << "," << std::endl;
    
    std::vector<sirius::atomic_level_descriptor> core;
    std::vector<sirius::atomic_level_descriptor> valence;
    for (int ist = 0; ist < (int)a->num_atomic_levels(); ist++)
    {
        if (enu[ist] < ecore_cutoff)
            core.push_back(a->atomic_level(ist));
        else
            valence.push_back(a->atomic_level(ist));
        
    }
    
    std::string symb[] = {"s", "p", "d", "f"};
    std::string core_str;
    for (int i = 0; i < (int)core.size(); i++)
    {
        std::stringstream ss;
        ss << core[i].n;
        core_str += (ss.str() + symb[core[i].l]);
    }
    fout << "  \\"core\\"    : \\"" << core_str << "\\", " << std::endl;
    
    fout << "  \\"valence\\" : [" << std::endl;
    fout << "    {\\"l\\" : -1, \\"basis\\" : [{\\"enu\\" : 0.15, \\"dme\\" : 0, \\"auto\\" : false}]}";
    
    int lmax = 0;
    for (int i = 0; i < (int)valence.size(); i++)
        lmax = std::max(lmax, valence[i].l); 
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
             << "    {\\"l\\" : " << l << ", \\"n\\" : " << n << ", \\"basis\\" : [{\\"enu\\" : 0.15, \\"dme\\" : 0, \\"auto\\" : true}]}";
    }
    fout << "]," << std::endl;
    
    fout << "  \\"lo\\"      : [";
    for (int i = 0; i < (int)valence.size(); i++)
    {
        if (i) fout << ",";
        fout << std::endl;
        fout << "    {\\"l\\" : " << valence[i].l << ", \\"n\\" : " << valence[i].n
             << ", \\"basis\\" : [{\\"enu\\" : 0.15, \\"dme\\" : 0, \\"auto\\" : true}, {\\"enu\\" : 0.15, \\"dme\\" : 1, \\"auto\\" : true}]}";
    }
    fout << "]" << std::endl;
    fout<< "}" << std::endl;
    fout.close();
    
}

int main(int argn, char **argv)
{
    init_atom_configuration();
    
    for (int i = 0; i < (int)atoms.size(); i++)
        solve_atom(atoms[i]); 
    
    /*std::ofstream fout("atomic_conf.h");
    fout << "const int atomic_conf[104][28][4] = " << std::endl;
    fout << "{" << std::endl;
    for (int iat = 0; iat < (int)atoms.size(); iat++)
    {
        if (iat) fout << ", " << std::endl;
        fout << "   {";
        for (int ist = 0; ist < (int)atoms[iat]->levels_nlk().size(); ist++)
        {
            if (ist) fout << ", ";
            fout << "{" << atoms[iat]->levels_nlk()[ist].n << ", " 
                        << atoms[iat]->levels_nlk()[ist].l << ", "
                        << atoms[iat]->levels_nlk()[ist].k << ", "
                        << atoms[iat]->levels_nlk()[ist].occupancy << "}";
        }
        for (int ist = atoms[iat]->levels_nlk().size(); ist < 28; ist++)
            fout << ", {-1, -1, -1, -1}";
        fout << "}";
    }
    fout << std::endl << "};" << std::endl;
    fout.close();
    
    fout.open("atomic_symb.h");
    fout << "const std::string atomic_symb[104] = {";
    for (int iat = 0; iat < (int)atoms.size(); iat++)
    {
        if (iat) fout << ", ";
        fout << "\\"" << atoms[iat]->symbol() << "\\"";
    }
    fout << "};" << std::endl;
    fout.close();*/
    
    sirius::Timer::print();
}

void init_atom_configuration() 
{
    int nl_occ[7][4];
    sirius::atomic_level_descriptor nlk;
    sirius::atomic_level_descriptor nl;
    std::vector<sirius::atomic_level_descriptor> levels_nl;
    std::vector<sirius::atomic_level_descriptor> levels_nlk;
    
    atom* a;
'''
fout = open("atoms.cpp", "w")
fout.write(s1)

fin = open("atoms.in", "r")
while 1:
    line = fin.readline()
    if not line: break
    if line.find("atom") == 0:
        fout.write("\n");
        
        line = fin.readline()
        s1 = line.split()
        zn = s1[0]
        line = fin.readline() # symbol and name
        line = line.replace("'", " ")
        s1 = line.split()
        symbol = s1[0]
        name = s1[1]
        line = fin.readline()
        s1 = line.split()
        mass = s1[0] # mass
        
        line = fin.readline() # Rmt
        line = fin.readline() # number of states
        s1 = line.split()
        nst = int(s1[0])
        
        fout.write("    levels_nl.clear(); \n");
        fout.write("    levels_nlk.clear(); \n");
        fout.write("    memset(&nl_occ[0][0], 0, 28 * sizeof(int)); \n")

        for i in range(nst):
            line = fin.readline() # n l k occ
            s1 = line.split()
            n = int(s1[0]);
            l = int(s1[1]);
            k = int(s1[2]);
            occ = int(s1[3]);
            fout.write("    nlk.n = " + str(n) + ";\n")
            fout.write("    nlk.l = " + str(l) + ";\n")
            fout.write("    nlk.k = " + str(k) + ";\n")
            fout.write("    nlk.occupancy = " + str(occ) + ";\n")
            fout.write("    levels_nlk.push_back(nlk);\n")
            fout.write("    nl_occ[" + str(n - 1) + "][" + str(l) + "] += " + str(occ) + "; \n")
            
        fout.write("    for (int n = 0; n < 7; n++) \n")
        fout.write("    { \n")
        fout.write("        for (int l = 0; l < 4; l++) \n")
        fout.write("        { \n")
        fout.write("            if (nl_occ[n][l]) \n")
        fout.write("            { \n")
        fout.write("                nl.n = n + 1; \n")
        fout.write("                nl.l = l; \n")
        fout.write("                nl.occupancy = nl_occ[n][l]; \n")
        fout.write("                levels_nl.push_back(nl);\n")
        fout.write("            } \n")    
        fout.write("        } \n")
        fout.write("    } \n")
        
        fout.write("    a = new atom(\"" + symbol + "\", \"" + name + "\", " + zn + ", " + mass + ", " + "levels_nl, levels_nlk);\n")
        line = fin.readline() # NIST LDA Etot
        line = line.strip()
        if  (line != ""):
            s1 = line.split()
            fout.write("    a->NIST_LDA_Etot = " + s1[0] + "; \n")
        
        fout.write("    atoms.push_back(a);\n")
        


fout.write("}\n")    
    
  
fin.close()
fout.close()


