#include "atom_type.h"

namespace sirius {

void Atom_type::add_lo_descriptor(int ilo, int n, int l, double enu, int dme, int auto_enu)
{
    if ((int)lo_descriptors_.size() == ilo) 
    {
        lo_descriptors_.push_back(local_orbital_descriptor());
        lo_descriptors_[ilo].l = l;
    }
    else
    {
        if (l != lo_descriptors_[ilo].l)
        {
            std::stringstream s;
            s << "wrong angular quantum number" << std::endl
              << "atom type id: " << id() << " (" << symbol_ << ")" << std::endl
              << "idxlo: " << ilo << std::endl
              << "n: " << l << std::endl
              << "l: " << n << std::endl
              << "expected l: " <<  lo_descriptors_[ilo].l << std::endl;
            TERMINATE(s);
        }
    }
    
    radial_solution_descriptor rsd;
    
    rsd.n = n;
    if (n == -1)
    {
        /* default value for any l */
        rsd.n = l + 1;
        for (int ist = 0; ist < num_atomic_levels(); ist++)
        {
            if (atomic_level(ist).core && atomic_level(ist).l == l)
            {   
                /* take next level after the core */
                rsd.n = atomic_level(ist).n + 1;
            }
        }
    }
    
    rsd.l = l;
    rsd.dme = dme;
    rsd.enu = enu;
    rsd.auto_enu = auto_enu;
    lo_descriptors_[ilo].rsd_set.push_back(rsd);
}

}
