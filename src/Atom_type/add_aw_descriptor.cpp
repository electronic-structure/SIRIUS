#include "atom_type.h"

namespace sirius {

void Atom_type::add_aw_descriptor(int n, int l, double enu, int dme, int auto_enu)
{
    if ((int)aw_descriptors_.size() < (l + 1)) aw_descriptors_.resize(l + 1, radial_solution_descriptor_set());
    
    radial_solution_descriptor rsd;
    
    rsd.n = n;
    if (n == -1)
    {
        /* default principal quantum number value for any l */
        rsd.n = l + 1;
        for (int ist = 0; ist < num_atomic_levels(); ist++)
        {
            /* take next level after the core */
            if (atomic_level(ist).core && atomic_level(ist).l == l) rsd.n = atomic_level(ist).n + 1;
        }
    }
    
    rsd.l = l;
    rsd.dme = dme;
    rsd.enu = enu;
    rsd.auto_enu = auto_enu;
    aw_descriptors_[l].push_back(rsd);
}

}
