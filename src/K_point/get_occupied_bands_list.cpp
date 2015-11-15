#include "k_point.h"

namespace sirius {

occupied_bands_descriptor K_point::get_occupied_bands_list(Communicator const& comm__)
{
    /* explicitly index up and dn bands in case of spin-collinear case */
    int ns = (parameters_.num_mag_dims() == 1) ? 2 : 1;
    /* this is how the spinor wave functions are counted */
    int nb = (parameters_.num_mag_dims() == 3) ? parameters_.num_bands() : parameters_.num_fv_states();

    occupied_bands_descriptor occupied_bands;

    //splindex<block_cyclic> spl_nb(nb, comm__.size(), comm__.rank(), 1);
    splindex<block> spl_nb(nb, comm__.size(), comm__.rank());

    for (int p = 0; p < ns ; p++)
    {
        for (int jloc = 0; jloc < (int)spl_nb.local_size(); jloc++)
        {
            int j = spl_nb[jloc] + p * parameters_.num_fv_states();
            double w = band_occupancy(j) * weight();
            if (w > 1e-14)
            {
                occupied_bands.idx_bnd_loc.push_back(jloc);
                occupied_bands.idx_bnd_glob.push_back(j);
                occupied_bands.weight.push_back(w);
            }
        }
    }

    occupied_bands.num_occupied_bands_ = static_cast<int>(occupied_bands.idx_bnd_loc.size());
    comm__.allreduce(&occupied_bands.num_occupied_bands_, 1);

    return occupied_bands;
}

};
