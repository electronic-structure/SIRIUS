#include "density.h"

namespace sirius {

void Density::generate_core_charge_density()
{
    PROFILE_WITH_TIMER("sirius::Density::generate_core_charge_density");

    for (int icloc = 0; icloc < unit_cell_.spl_num_atom_symmetry_classes().local_size(); icloc++)
    {
        int ic = unit_cell_.spl_num_atom_symmetry_classes(icloc);
        unit_cell_.atom_symmetry_class(ic).generate_core_charge_density();
    }

    for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++)
    {
        int rank = unit_cell_.spl_num_atom_symmetry_classes().local_rank(ic);
        unit_cell_.atom_symmetry_class(ic).sync_core_charge_density(ctx_.comm(), rank);
    }
}

}
