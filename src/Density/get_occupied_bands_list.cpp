#include "density.h"

namespace sirius {

std::vector< std::pair<int, double> > Density::get_occupied_bands_list(Band* band, K_point* kp)
{
    std::vector< std::pair<int, double> > bands;
    if (parameters_.esm_type() == full_potential_lapwlo)
    {
        for (int jsub = 0; jsub < kp->num_sub_bands(); jsub++)
        {
            int j = kp->idxbandglob(jsub);
            double wo = kp->band_occupancy(j) * kp->weight();
            if (wo > 1e-14) bands.push_back(std::pair<int, double>(jsub, wo));
        }
    }
    else
    {
        splindex<block> spl_bands(num_occupied_bands(kp), kp->comm().size(), kp->comm().rank());
        for (int jsub = 0; jsub < (int)spl_bands.local_size(); jsub++)
        {
            int j = (int)spl_bands[jsub];
            double wo = kp->band_occupancy(j) * kp->weight();
            if (wo > 1e-14) bands.push_back(std::pair<int, double>(jsub, wo));
        }
    }
    return bands;
}

};

