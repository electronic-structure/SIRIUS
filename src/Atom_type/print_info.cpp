#include "atom_type.h"

namespace sirius {

void Atom_type::print_info() const
{
    printf("\n");
    printf("symbol         : %s\n", symbol_.c_str());
    printf("name           : %s\n", name_.c_str());
    printf("zn             : %i\n", zn_);
    printf("mass           : %f\n", mass_);
    printf("mt_radius      : %f\n", mt_radius_);
    printf("num_mt_points  : %i\n", num_mt_points_);
    printf("grid_origin    : %f\n", radial_grid_[0]);
    printf("grid_name      : %s\n", radial_grid_.grid_type_name().c_str());
    printf("\n");
    printf("number of core electrons    : %f\n", num_core_electrons_);
    printf("number of valence electrons : %f\n", num_valence_electrons_);

    if (parameters_.full_potential())
    {
        printf("\n");
        printf("atomic levels (n, l, k, occupancy, core)\n");
        for (int i = 0; i < (int)atomic_levels_.size(); i++)
        {
            printf("%i  %i  %i  %8.4f %i\n", atomic_levels_[i].n, atomic_levels_[i].l, atomic_levels_[i].k,
                                              atomic_levels_[i].occupancy, atomic_levels_[i].core);
        }
        printf("\n");
        printf("local orbitals\n");
        for (int j = 0; j < (int)lo_descriptors_.size(); j++)
        {
            printf("[");
            for (int order = 0; order < (int)lo_descriptors_[j].rsd_set.size(); order++)
            {
                if (order) printf(", ");
                printf("{l : %2i, n : %2i, enu : %f, dme : %i, auto : %i}", lo_descriptors_[j].rsd_set[order].l,
                                                                            lo_descriptors_[j].rsd_set[order].n,
                                                                            lo_descriptors_[j].rsd_set[order].enu,
                                                                            lo_descriptors_[j].rsd_set[order].dme,
                                                                            lo_descriptors_[j].rsd_set[order].auto_enu);
            }
            printf("]\n");
        }
    }

    if (parameters_.esm_type() == full_potential_lapwlo)
    {
        printf("\n");
        printf("augmented wave basis\n");
        for (int j = 0; j < (int)aw_descriptors_.size(); j++)
        {
            printf("[");
            for (int order = 0; order < (int)aw_descriptors_[j].size(); order++)
            { 
                if (order) printf(", ");
                printf("{l : %2i, n : %2i, enu : %f, dme : %i, auto : %i}", aw_descriptors_[j][order].l,
                                                                            aw_descriptors_[j][order].n,
                                                                            aw_descriptors_[j][order].enu,
                                                                            aw_descriptors_[j][order].dme,
                                                                            aw_descriptors_[j][order].auto_enu);
            }
            printf("]\n");
        }
        printf("maximum order of aw : %i\n", max_aw_order_);
    }

    printf("\n");
    printf("total number of radial functions : %i\n", indexr().size());
    printf("maximum number of radial functions per orbital quantum number: %i\n", indexr().max_num_rf());
    printf("total number of basis functions : %i\n", indexb().size());
    printf("number of aw basis functions : %i\n", indexb().size_aw());
    printf("number of lo basis functions : %i\n", indexb().size_lo());
}
        
}
