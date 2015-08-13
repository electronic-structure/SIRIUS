#include "k_point.h"

namespace sirius {

occupied_bands_descriptor K_point::get_occupied_bands_list()
{
    /* explicitly index up and dn bands in case of spin-collinear case */
    int ns = (parameters_.num_mag_dims() == 1) ? 2 : 1;

    occupied_bands_descriptor occupied_bands;

    for (int p = 0; p < ns ; p++)
    {
        for (int jloc = 0; jloc < spinor_wave_functions_[p].num_cols_local(); jloc++)
        {
            int j = spinor_wave_functions_[p].icol(jloc) + p * parameters_.num_fv_states();
            double w = band_occupancy(j) * weight();
            if (w > 1e-14)
            {
                occupied_bands.idx_bnd_loc.push_back(jloc);
                occupied_bands.idx_bnd_glob.push_back(j);
                occupied_bands.weight.push_back(w);
            }
        }
    }

    occupied_bands.num_occupied_bands_ = (int)occupied_bands.idx_bnd_loc.size();
    comm().allreduce(&occupied_bands.num_occupied_bands_, 1);

    return occupied_bands;
}

};
