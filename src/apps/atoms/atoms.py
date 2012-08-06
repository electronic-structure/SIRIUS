s1 = '''
#include <vector>
#include <string>
#include <string.h>
#include <iostream>
#include "../../lib/sirius.h"
#include <xc.h>

struct atom
{
    std::string symbol;
    std::string name;
    int zn;
    double mass;
    std::vector<sirius::atomic_level_nlk> nlk_list;
    std::vector<sirius::atomic_level_nl> nl_list;
    double NIST_LDA_Etot;
};

std::vector<atom> atoms(104); 

void init_atom_configuration();

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
            error(__FILE__, __LINE__, "Functional is not found");
       
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
    sirius::RadialGrid r(sirius::exponential_grid, 2000, 1e-6 / a.zn, 20.0);
    
    sirius::RadialSolver solver(false, -1.0 * a.zn, r);

    std::vector<double> veff(r.size());
    std::vector<double> vnuc(r.size());
    for (int i = 0; i < r.size(); i++)
    {
        vnuc[i] = -1.0 * a.zn / r[i];
        veff[i] = vnuc[i];
    }
    
    sirius::Spline rho(r.size(), r);
    sirius::Spline rho_core(r.size(), r);
    
    sirius::Spline f(r.size(), r);
    
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
    
    double ecore_cutoff = -3.0;

    for (int iter = 0; iter < 100; iter++)
    {
        rho_old = rho.data_points();
        
        memset(&rho[0], 0, rho.size() * sizeof(double));
        
        memset(&rho_core[0], 0, rho_core.size() * sizeof(double));

        for (int ist = 0; ist < (int)a.nl_list.size(); ist++)
        {
            enu[ist] = -1.0 * a.zn / 2 / pow(a.nl_list[ist].n, 2);

            solver.bound_state(a.nl_list[ist].n, a.nl_list[ist].l, veff, enu[ist], p);
            
            for (int i = 0; i < r.size(); i++)
                rho[i] += a.nl_list[ist].occupancy * pow(y00 * p[i] / r[i], 2);
            
            if (enu[ist] < ecore_cutoff) 
                for (int i = 0; i < r.size(); i++)
                    rho_core[i] += a.nl_list[ist].occupancy * pow(y00 * p[i] / r[i], 2);
                  
            
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
        for (int ist = 0; ist < (int)a.nl_list.size(); ist++)
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
    
    rho_core.interpolate();
    
    double ncore = fourpi * rho_core.integrate(g2, 2);
    
    double core_radius = 1.0;
    if (ncore > 1.0)
    {
        for (int i = 0; i < r.size(); i++)
            if (fabs(ncore - fourpi * g2[i]) < 0.001 * ncore) 
            {
                core_radius = r[i];
                break;
            }
    }
    
    std::cout << " atom : " << a.symbol << "    Z : " << a.zn << std::endl;
    std::cout << " =================== " << std::endl;
    std::cout << " convergence (charge, energy) : " << charge_rms << " " << energy_diff << std::endl;
    std::cout << " total energy : " << energy_tot << ", NIST value : " <<  a.NIST_LDA_Etot 
              << ", difference : " << fabs(energy_tot - a.NIST_LDA_Etot) << std::endl;
    std::cout << " number of core electrons : " <<  ncore << std::endl;
    std::cout << " muffin-tin radius : " <<  core_radius << std::endl;
    std::cout << std::endl;
    
    std::string fname = a.symbol + std::string(".json");
    std::ofstream fout(fname.c_str());
    fout << "{" << std::endl;
    fout << "  \\"name\\"    : \\"" << a.name << "\\"," << std::endl;
    fout << "  \\"symbol\\"  : \\"" << a.symbol << "\\"," << std::endl;
    fout << "  \\"number\\"  : " << a.zn << "," << std::endl;
    fout << "  \\"mass\\"    : " << a.mass << "," << std::endl;
    fout << "  \\"rmin\\"    : " << r[0] << "," << std::endl;
    fout << "  \\"rmax\\"    : " << r[r.size() - 1] << "," << std::endl;
    fout << "  \\"rmt\\"     : " << core_radius << "," << std::endl;
    fout << "  \\"nrmt\\"    : " <<  800 << "," << std::endl;
    
    std::vector<sirius::atomic_level_nl> core;
    std::vector<sirius::atomic_level_nl> valence;
    for (int ist = 0; ist < (int)a.nl_list.size(); ist++)
    {
        if (enu[ist] < ecore_cutoff)
        {
            /*for (int jst = 0; jst < (int)a.nlk_list.size(); jst++)
            {
                if ((a.nlk_list[jst].n == a.nl_list[ist].n) && (a.nlk_list[jst].l == a.nl_list[ist].l)) 
                    core.push_back(a.nlk_list[jst]);
            }*/
            core.push_back(a.nl_list[ist]);
        }
        else
            valence.push_back(a.nl_list[ist]);
        
    }
    
    /*fout << "  \\"core\\"    : [";
    for (int i = 0; i < (int)core.size(); i++)
    {
        if (i) fout << ",";
        fout << std::endl << "    {\\"n\\" : " << core[i].n << ", \\"l\\" : " << core[i].l << ", \\"k\\" : " << core[i].k 
             << ", \\"occupancy\\" : " << core[i].occupancy << "}";
    }
    fout << "]," << std::endl; */
    
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
    {
        for (int ist = 0; ist < (int)atoms[i].nlk_list.size(); ist++)
        {
            if ((atomic_conf[i][ist][0] != atoms[i].nlk_list[ist].n) ||
                (atomic_conf[i][ist][1] != atoms[i].nlk_list[ist].l) ||
                (atomic_conf[i][ist][2] != atoms[i].nlk_list[ist].k) ||
                (atomic_conf[i][ist][3] != atoms[i].nlk_list[ist].occupancy)) 
                error(__FILE__, __LINE__, "wrong atomic_conf array");
        }
        solve_atom(atoms[i]); 
    }
    
    std::ofstream fout("atomic_conf.h");
    fout << "const int atomic_conf[104][28][4] = " << std::endl;
    fout << "{" << std::endl;
    for (int iat = 0; iat < (int)atoms.size(); iat++)
    {
        if (iat) fout << ", " << std::endl;
        fout << "   {";
        for (int ist = 0; ist < (int)atoms[iat].nlk_list.size(); ist++)
        {
            if (ist) fout << ", ";
            fout << "{" << atoms[iat].nlk_list[ist].n << ", " 
                        << atoms[iat].nlk_list[ist].l << ", "
                        << atoms[iat].nlk_list[ist].k << ", "
                        << atoms[iat].nlk_list[ist].occupancy << "}";
        }
        for (int ist = atoms[iat].nlk_list.size(); ist < 28; ist++)
        {
            fout << ", {-1, -1, -1, -1}";
        }
        fout << "}";
    }
    fout << std::endl << "};" << std::endl;
    fout.close();
    
    fout.open("atomic_symb.h");
    fout << "const std::string atomic_symb[104] = {";
    for (int iat = 0; iat < (int)atoms.size(); iat++)
    {
        if (iat) fout << ", ";
        fout << "\\"" << atoms[iat].symbol << "\\"";
    }
    fout << "};" << std::endl;
    fout.close();
}

void init_atom_configuration() 
{
    int nl_occ[7][4];
    sirius::atomic_level_nlk nlk;
    sirius::atomic_level_nl nl;
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


