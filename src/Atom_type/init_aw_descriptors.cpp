#include "atom_type.h"

namespace sirius {

void Atom_type::init_aw_descriptors(int lmax)
{
    assert(lmax >= -1);

    if (lmax >= 0 && aw_default_l_.size() == 0) TERMINATE("default AW descriptor is empty"); 

    aw_descriptors_.clear();
    for (int l = 0; l <= lmax; l++)
    {
        aw_descriptors_.push_back(aw_default_l_);
        for (size_t ord = 0; ord < aw_descriptors_[l].size(); ord++)
        {
            aw_descriptors_[l][ord].n = l + 1;
            aw_descriptors_[l][ord].l = l;
        }
    }

    for (size_t i = 0; i < aw_specific_l_.size(); i++)
    {
        int l = aw_specific_l_[i][0].l;
        if (l < lmax) aw_descriptors_[l] = aw_specific_l_[i];
    }
}

}
