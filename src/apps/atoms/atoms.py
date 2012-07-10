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
    double NIST_LDA_Etot;
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
    sirius::radial_grid r(sirius::exponential_grid, 4000, 1e-8, 30.0);
    
    sirius::radial_solver solver(false, -1.0 * a.zn, r);

    std::vector<double> veff(r.size());
    std::vector<double> vnuc(r.size());
    for (int i = 0; i < r.size(); i++)
    {
        vnuc[i] = -1.0 * a.zn / r[i];
        veff[i] = vnuc[i];
    }
    
    sirius::spline rho(r.size(), r);
    
    sirius::spline f(r.size(), r);
    
    std::vector<double> vh(r.size());
    std::vector<double> vxc;
    std::vector<double> exc;
    std::vector<double> g1;
    std::vector<double> g2;
    std::vector<double> p;
    std::vector<double> rho_old;
    
    std::vector<double> enu(a.nl_list.size());
    
    double energy_tot = 1e100;
    double energy_tot_old;
    double charge_rms;
    double energy_diff;
    
    double beta = 0.5;

    for (int iter = 0; iter < 100; iter++)
    {
        rho_old = rho.data_points();
        
        memset(&rho[0], 0, rho.size() * sizeof(double));

        for (int ist = 0; ist < a.nl_list.size(); ist++)
        {
            enu[ist] = -1.0 * a.zn / 2 / pow(a.nl_list[ist].n, 2);

            solver.bound_state(a.nl_list[ist].n, a.nl_list[ist].l, enu[ist], veff, p);
            
            for (int i = 0; i < r.size(); i++)
                rho[i] += a.nl_list[ist].occupancy * pow(y00 * p[i] / r[i], 2);
        }
        
        charge_rms = 0.0;
        for (int i = 0; i < r.size(); i++)
            charge_rms += pow(rho[i] - rho_old[i], 2);
        charge_rms = sqrt(charge_rms / r.size());
        
        rho.interpolate();
        
        //std::cout << "number of electrons : " << fourpi * rho.integrate(2) << std::endl;

        rho.integrate(g2, 2);
        double t1 = rho.integrate(g1, 1);

        for (int i = 0; i < r.size(); i++)
            vh[i] = fourpi * (g2[i] / r[i] + t1 - g1[i]);

        potxc(rho.data_points(), vxc, exc);

        for (int i = 0; i < r.size(); i++)
            veff[i] = (1 - beta) * veff[i] + beta * (vnuc[i] + vh[i] + vxc[i]);
        
        // kinetic energy
        for (int i = 0; i < r.size(); i++)
            f[i] = veff[i] * rho[i];
        f.interpolate();
        
        double eval_sum = 0.0;
        for (int ist = 0; ist < a.nl_list.size(); ist++)
            eval_sum += a.nl_list[ist].occupancy * enu[ist];

        double energy_kin = eval_sum - fourpi * f.integrate(2);

        // xc energy
        for (int i = 0; i < r.size(); i++)
            f[i] = exc[i] * rho[i];
        f.interpolate();
        double energy_xc = fourpi * f.integrate(2); 
        
        // electron-nuclear energy
        for (int i = 0; i < r.size(); i++)
            f[i] = vnuc[i] * rho[i];
        f.interpolate();
        double energy_enuc = fourpi * f.integrate(2); 

        // Coulomb energy
        for (int i = 0; i < r.size(); i++)
            f[i] = vh[i] * rho[i];
        f.interpolate();
        double energy_coul = 0.5 * fourpi * f.integrate(2);
        
        energy_tot_old = energy_tot;

        energy_tot = energy_kin + energy_xc + energy_coul + energy_enuc; 
        
        energy_diff = fabs(energy_tot - energy_tot_old);
        
        if (energy_diff < 1e-7 && charge_rms < 1e-7) break;
    }
    
    
    std::cout << " atom : " << a.symbol << "    Z : " << a.zn << std::endl;
    std::cout << " =================== " << std::endl;
    std::cout << "total energy : " << energy_tot 
              << ",  convergence (charge, energy) : " << charge_rms << " " << energy_diff 
              << ",  difference with NIST : " << fabs(energy_tot - a.NIST_LDA_Etot) << std::endl;
    std::cout << std::endl;
}

int main(int argn, char **argv)
{
    atom_configuration();
    
    for (int i = 0; i < atoms.size(); i++)
    {
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
        fout.write("    atoms[" + str(zn - 1) + "].zn = " + str(zn) + "; \n")
        
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
        
        line = fin.readline() # NIST LDA Etot
        line = line.strip()
        if  (line != ""):
            s1 = line.split()
            fout.write("    atoms[" + str(zn - 1) + "].NIST_LDA_Etot = " + s1[0] + "; \n")
        else:
            fout.write("    atoms[" + str(zn - 1) + "].NIST_LDA_Etot = 0.0; \n")


fout.write("}\n")    
    
  
fin.close()
fout.close()


