#include "atom_type.h"

namespace sirius {

void Atom_type::read_input_core(JSON_tree& parser)
{
    std::string core_str;
    parser["core"] >> core_str;
    if (int size = (int)core_str.size())
    {
        if (size % 2)
        {
            std::string s = std::string("wrong core configuration string : ") + core_str;
            TERMINATE(s);
        }
        int j = 0;
        while (j < size)
        {
            char c1 = core_str[j++];
            char c2 = core_str[j++];
            
            int n = -1;
            int l = -1;
            
            std::istringstream iss(std::string(1, c1));
            iss >> n;
            
            if (n <= 0 || iss.fail())
            {
                std::string s = std::string("wrong principal quantum number : " ) + std::string(1, c1);
                TERMINATE(s);
            }
            
            switch (c2)
            {
                case 's':
                {
                    l = 0;
                    break;
                }
                case 'p':
                {
                    l = 1;
                    break;
                }
                case 'd':
                {
                    l = 2;
                    break;
                }
                case 'f':
                {
                    l = 3;
                    break;
                }
                default:
                {
                    std::string s = std::string("wrong angular momentum label : " ) + std::string(1, c2);
                    TERMINATE(s);
                }
            }

            atomic_level_descriptor level;
            level.n = n;
            level.l = l;
            level.core = true;
            for (int ist = 0; ist < 28; ist++)
            {
                if ((level.n == atomic_conf[zn_ - 1][ist][0]) && (level.l == atomic_conf[zn_ - 1][ist][1]))
                {
                    level.k = atomic_conf[zn_ - 1][ist][2]; 
                    level.occupancy = double(atomic_conf[zn_ - 1][ist][3]);
                    atomic_levels_.push_back(level);
                }
            }
        }
    }
}

void Atom_type::read_input_aw(JSON_tree& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;
    
    // default augmented wave basis
    rsd.n = -1;
    rsd.l = -1;
    for (int order = 0; order < parser["valence"][0]["basis"].size(); order++)
    {
        parser["valence"][0]["basis"][order]["enu"] >> rsd.enu;
        parser["valence"][0]["basis"][order]["dme"] >> rsd.dme;
        parser["valence"][0]["basis"][order]["auto"] >> rsd.auto_enu;
        aw_default_l_.push_back(rsd);
    }
    
    for (int j = 1; j < parser["valence"].size(); j++)
    {
        parser["valence"][j]["l"] >> rsd.l;
        parser["valence"][j]["n"] >> rsd.n;
        rsd_set.clear();
        for (int order = 0; order < parser["valence"][j]["basis"].size(); order++)
        {
            parser["valence"][j]["basis"][order]["enu"] >> rsd.enu;
            parser["valence"][j]["basis"][order]["dme"] >> rsd.dme;
            parser["valence"][j]["basis"][order]["auto"] >> rsd.auto_enu;
            rsd_set.push_back(rsd);
        }
        aw_specific_l_.push_back(rsd_set);
    }
}
    
void Atom_type::read_input_lo(JSON_tree& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;
    
    int l;
    for (int j = 0; j < parser["lo"].size(); j++)
    {
        parser["lo"][j]["l"] >> l;

        if (parser["lo"][j].exist("basis"))
        {
            local_orbital_descriptor lod;
            lod.l = l;
            rsd.l = l;
            rsd_set.clear();
            for (int order = 0; order < parser["lo"][j]["basis"].size(); order++)
            {
                parser["lo"][j]["basis"][order]["n"] >> rsd.n;
                parser["lo"][j]["basis"][order]["enu"] >> rsd.enu;
                parser["lo"][j]["basis"][order]["dme"] >> rsd.dme;
                parser["lo"][j]["basis"][order]["auto"] >> rsd.auto_enu;
                rsd_set.push_back(rsd);
            }
            lod.rsd_set = rsd_set;
            lo_descriptors_.push_back(lod);
        }
    }
}
    
void Atom_type::read_input(const std::string& fname)
{
    JSON_tree parser(fname);

    if (!parameters_.full_potential())
    {
        parser["uspp"]["header"]["element"] >> symbol_;

        double zp;
        parser["uspp"]["header"]["zp"] >> zp;
        zn_ = int(zp + 1e-10);

        int nmesh;
        parser["uspp"]["header"]["nmesh"] >> nmesh;

        parser["uspp"]["radial_grid"] >> uspp_.r;

        parser["uspp"]["vloc"] >> uspp_.vloc;

        uspp_.core_charge_density = parser["uspp"]["core_charge_density"].get(std::vector<double>(nmesh, 0));

        parser["uspp"]["total_charge_density"] >> uspp_.total_charge_density;

        if ((int)uspp_.r.size() != nmesh)
        {
            TERMINATE("wrong mesh size");
        }
        if ((int)uspp_.vloc.size() != nmesh || 
            (int)uspp_.core_charge_density.size() != nmesh || 
            (int)uspp_.total_charge_density.size() != nmesh)
        {
            std::cout << uspp_.vloc.size()  << " " << uspp_.core_charge_density.size() << " " << uspp_.total_charge_density.size() << std::endl;
            TERMINATE("wrong array size");
        }

        num_mt_points_ = nmesh;
        mt_radius_ = uspp_.r[nmesh - 1];
        
        set_radial_grid(nmesh, &uspp_.r[0]);

        parser["uspp"]["header"]["lmax"] >> uspp_.lmax;
        parser["uspp"]["header"]["nbeta"] >> uspp_.num_beta_radial_functions;

        if (parser["uspp"]["non_local"].exist("Q"))
        {
            parser["uspp"]["non_local"]["Q"]["num_q_coefs"] >> uspp_.num_q_coefs;

            parser["uspp"]["non_local"]["Q"]["q_functions_inner_radii"] >> uspp_.q_functions_inner_radii;

            uspp_.q_coefs = mdarray<double, 4>(uspp_.num_q_coefs, 2 * uspp_.lmax + 1, 
                                               uspp_.num_beta_radial_functions,  uspp_.num_beta_radial_functions); 

            uspp_.q_radial_functions_l = mdarray<double, 3>(num_mt_points_, uspp_.num_beta_radial_functions * (uspp_.num_beta_radial_functions + 1) / 2, 2 * uspp_.lmax + 1);

            for (int j = 0; j < uspp_.num_beta_radial_functions; j++)
            {
                for (int i = 0; i <= j; i++)
                {
                    int idx = j * (j + 1) / 2 + i;

                    std::vector<int> ij;
                    parser["uspp"]["non_local"]["Q"]["qij"][idx]["ij"] >> ij;
                    if (ij[0] != i || ij[1] != j) 
                    {
                        std::stringstream s;
                        s << "wrong ij indices" << std::endl
                          << "i = " << i << " j = " << j << " idx = " << idx << std::endl
                          << "ij = " << ij[0] << " " << ij[1];
                        TERMINATE(s);
                    }

                    std::vector<double> qfcoef;
                    parser["uspp"]["non_local"]["Q"]["qij"][idx]["q_coefs"] >> qfcoef;

                    int k = 0;
                    for (int l = 0; l <= 2 * uspp_.lmax; l++)
                    {
                        for (int n = 0; n < uspp_.num_q_coefs; n++) 
                        {
                            if (k >= (int)qfcoef.size()) TERMINATE("wrong size of qfcoef");
                            uspp_.q_coefs(n, l, i, j) = uspp_.q_coefs(n, l, j, i) = qfcoef[k++];
                        }
                    }

                    std::vector<double> qfunc;
                    parser["uspp"]["non_local"]["Q"]["qij"][idx]["q_radial_function"] >> qfunc;
                    if ((int)qfunc.size() != num_mt_points_) TERMINATE("wrong size of qfunc");
                    
                    for (int l = 0; l <= 2 * uspp_.lmax; l++)
                        memcpy(&uspp_.q_radial_functions_l(0, idx, l), &qfunc[0], num_mt_points_ * sizeof(double)); 
                }
            }
        }

        uspp_.beta_radial_functions = mdarray<double, 2>(num_mt_points_, uspp_.num_beta_radial_functions);
        uspp_.beta_radial_functions.zero();

        uspp_.num_beta_radial_points.resize(uspp_.num_beta_radial_functions);
        uspp_.beta_l.resize(uspp_.num_beta_radial_functions);

        local_orbital_descriptor lod;
        for (int i = 0; i < uspp_.num_beta_radial_functions; i++)
        {
            parser["uspp"]["non_local"]["beta"][i]["kbeta"] >> uspp_.num_beta_radial_points[i];
            std::vector<double> beta;
            parser["uspp"]["non_local"]["beta"][i]["beta"] >> beta;
            if ((int)beta.size() != uspp_.num_beta_radial_points[i]) TERMINATE("wrong size of beta function");
            memcpy(&uspp_.beta_radial_functions(0, i), &beta[0], beta.size() * sizeof(double)); 
 
            parser["uspp"]["non_local"]["beta"][i]["lll"] >> uspp_.beta_l[i];
        }

        uspp_.d_mtrx_ion = mdarray<double, 2>(uspp_.num_beta_radial_functions, uspp_.num_beta_radial_functions);
        uspp_.d_mtrx_ion.zero();

        for (int k = 0; k < parser["uspp"]["non_local"]["D"].size(); k++)
        {
            double d;
            std::vector<int> ij;
            parser["uspp"]["non_local"]["D"][k]["ij"] >> ij;
            parser["uspp"]["non_local"]["D"][k]["d_ion"] >> d;
            uspp_.d_mtrx_ion(ij[0], ij[1]) = d;
            uspp_.d_mtrx_ion(ij[1], ij[0]) = d;
        }
        
    }

    if (parameters_.full_potential())
    {
        parser["name"] >> name_;
        parser["symbol"] >> symbol_;
        parser["mass"] >> mass_;
        parser["number"] >> zn_;
        parser["rmin"] >> radial_grid_origin_;
        parser["rmt"] >> mt_radius_;
        parser["nrmt"] >> num_mt_points_;

        read_input_core(parser);

        read_input_aw(parser);

        read_input_lo(parser);
    }
}

}
