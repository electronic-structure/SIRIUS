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
        parser["pseudo_potential"]["header"]["element"] >> symbol_;

        double zp;
        parser["pseudo_potential"]["header"]["z_valence"] >> zp;
        zn_ = int(zp + 1e-10);

        int nmesh;
        parser["pseudo_potential"]["header"]["mesh_size"] >> nmesh;

        parser["pseudo_potential"]["radial_grid"] >> uspp_.r;

        parser["pseudo_potential"]["local_potential"] >> uspp_.vloc;

        uspp_.core_charge_density = parser["pseudo_potential"]["core_charge_density"].get(std::vector<double>(nmesh, 0));

        parser["pseudo_potential"]["total_charge_density"] >> uspp_.total_charge_density;

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

        parser["pseudo_potential"]["header"]["number_of_proj"] >> uspp_.num_beta_radial_functions;

        if (parser["pseudo_potential"].exist("augmentation"))
        {
            uspp_.q_radial_functions_l = mdarray<double, 3>(num_mt_points_, uspp_.num_beta_radial_functions * (uspp_.num_beta_radial_functions + 1) / 2, 2 * uspp_.lmax_beta_ + 1);
            uspp_.q_radial_functions_l.zero();

            for (int k = 0; k < parser["pseudo_potential"]["augmentation"].size(); k++)
            {
                int i, j, l;
                parser["pseudo_potential"]["augmentation"][k]["i"] >> i;
                parser["pseudo_potential"]["augmentation"][k]["j"] >> j;
                int idx = j * (j + 1) / 2 + i;
                parser["pseudo_potential"]["augmentation"][k]["angular_momentum"] >> l;
                std::vector<double> qij;
                parser["pseudo_potential"]["augmentation"][k]["radial_function"] >> qij;
                if ((int)qij.size() != num_mt_points_) TERMINATE("wrong size of qij");
                    
                std::memcpy(&uspp_.q_radial_functions_l(0, idx, l), &qij[0], num_mt_points_ * sizeof(double)); 
            }
        }

        uspp_.beta_radial_functions = mdarray<double, 2>(num_mt_points_, uspp_.num_beta_radial_functions);
        uspp_.beta_radial_functions.zero();

        uspp_.num_beta_radial_points.resize(uspp_.num_beta_radial_functions);
        uspp_.beta_l.resize(uspp_.num_beta_radial_functions);

        local_orbital_descriptor lod;
        for (int i = 0; i < uspp_.num_beta_radial_functions; i++)
        {
            parser["pseudo_potential"]["beta_projectors"][i]["cutoff_radius_index"] >> uspp_.num_beta_radial_points[i];
            std::vector<double> beta;
            parser["pseudo_potential"]["beta_projectors"][i]["radial_function"] >> beta;
            if ((int)beta.size() != uspp_.num_beta_radial_points[i]) TERMINATE("wrong size of beta function");
            std::memcpy(&uspp_.beta_radial_functions(0, i), &beta[0], beta.size() * sizeof(double)); 
 
            parser["pseudo_potential"]["beta_projectors"][i]["angular_momentum"] >> uspp_.beta_l[i];
        }

        uspp_.d_mtrx_ion = mdarray<double, 2>(uspp_.num_beta_radial_functions, uspp_.num_beta_radial_functions);
        uspp_.d_mtrx_ion.zero();
        std::vector<double> dion;
        parser["pseudo_potential"]["D_ion"] >> dion;

        for (int i = 0; i < uspp_.num_beta_radial_functions; i++)
        {
            for (int j = 0; j < uspp_.num_beta_radial_functions; j++)
                uspp_.d_mtrx_ion(i, j) = dion[j * uspp_.num_beta_radial_functions + i];
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
