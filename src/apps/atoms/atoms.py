s1 = '''
#include <vector>
#include <string>
#include <string.h>
#include <iostream>
#include "../../lib/sirius.h"
#include <xc.h>

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

void atom_configuration();

void potxc(std::vector<double>& rho, std::vector<double>& vxc, std::vector<double>& exc)
{
    vxc.resize(rho.size());
    memset(&vxc[0], 0, vxc.size() * sizeof(double)); 
    exc.resize(rho.size());
    memset(&exc[0], 0, exc.size() * sizeof(double));

    std::vector<double> tmp(rho.size());

    int xc_id[2] = {1, 7};
    xc_func_type func;
    
    for (int i = 0; i < 2; i++)
    {
        if(xc_func_init(&func, xc_id[i], XC_UNPOLARIZED) != 0)
            stop(std::cout << "Functional is not found");
       
        xc_lda_vxc(&func, rho.size(), &rho[0], &tmp[0]);

        for (int j = 0; j < (int)rho.size(); j++)
            vxc[j] += tmp[j];

        xc_lda_exc(&func, rho.size(), &rho[0], &tmp[0]);

        for (int j = 0; j < (int)rho.size(); j++)
            exc[j] += tmp[j];
      
        xc_func_end(&func);
    }
}


void solve_atom(atom& a)
{
    sirius::radial_grid r(sirius::exponential_grid, 30000, 1e-8, 100.0);
    
    sirius::radial_solver solver(false, -1.0 * a.zn, r);

    std::vector<double> v(r.size(), 0.0);
    for (int i = 0; i < r.size(); i++)
        v[i] = -1.0 * a.zn / r[i];
    
    std::vector<double> p;

    std::vector<double> rho(r.size());

    memset(&rho[0], 0, rho.size() * sizeof(double));

    for (int ist = 0; ist < a.nl_list.size(); ist++)
    {
        double enu = -0.5;
        solver.bound_state(a.nl_list[ist].n, a.nl_list[ist].l, enu, v, p);

        for (int i = 0; i < r.size(); i++)
            rho[i] += a.nl_list[ist].occupancy * pow(y00 * p[i] / r[i], 2);
    }

    sirius::spline rho_spline(r.size(), r, rho);
    
    std::cout << "number of electrons : " << fourpi * rho_spline.integrate(2) << std::endl;
}

int main(int argn, char **argv)
{
    atom_configuration();
    
    for (int i = 0; i < atoms.size(); i++)
    {
        std::cout << " i = " << i << " atom : " << atoms[i].symbol << std::endl;
        solve_atom(atoms[i]);
    }
    


}

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


