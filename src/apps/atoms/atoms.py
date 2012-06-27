s1 = '''
#include <vector>
#include <string>
#include <string.h>

struct atomic_level_nlk 
{
    int n;
    int l;
    int k;
    int occupancy;
};

struct atomic_level_nl
{
    int n;
    int l;
    int occupancy;
};

struct atom
{
    std::string symbol;
    std::string name;
    int zn;
    double mass;
    std::vector<atomic_level_nlk> nlk_list;
    std::vector<atomic_level_nl> nl_list;
};

std::vector<atom> atoms(104); 

void atom_configuration() 
{
    int nl_occ[7][4];
    atomic_level_nlk nlk;
    atomic_level_nl nl;
'''
fout = open("atoms.cpp", "w")
fout.write(s1)

fin = open("atoms.in", "r")
while 1:
    line = fin.readline()
    if not line: break
    if line.find("atom") == 0:
        fout.write("\n");
        line = fin.readline() # atomic number
        s1 = line.split()
        zn = int(s1[0])
        line = fin.readline() # symbol and name
        line = line.replace("'", " ")
        s1 = line.split()
        fout.write("    atoms[" + str(zn - 1) + "].symbol = \"" + s1[0] + "\"; \n")
        fout.write("    atoms[" + str(zn - 1) + "].name = \"" + s1[1] + "\"; \n")
        line = fin.readline() # mass
        s1 = line.split()
        fout.write("    atoms[" + str(zn - 1) + "].mass = " + s1[0] + "; \n")
        line = fin.readline() # Rmt
        line = fin.readline() # number of states
        s1 = line.split()
        nst = int(s1[0])
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
            fout.write("    atoms[" + str(zn - 1) + "].nlk_list.push_back(nlk);\n")
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
        fout.write("                atoms[" + str(zn - 1) + "].nl_list.push_back(nl);\n")
        fout.write("            } \n")    
        fout.write("        } \n")
        fout.write("    } \n")


fout.write("}\n")    
    
  
fin.close()
fout.close()


