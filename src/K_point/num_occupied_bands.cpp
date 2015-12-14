#include "k_point.h"

namespace sirius {

int K_point::num_occupied_bands(int ispn__)
{
    if ((parameters_.num_mag_dims() == 3 || parameters_.num_mag_dims() == 0) && ispn__ != 0)
    {
        TERMINATE("wrong spin index");
    }

    int nb = (parameters_.num_mag_dims() == 3) ? parameters_.num_bands() : parameters_.num_fv_states();

    int n = 0;
    for (int i = 0; i < nb; i++)
    {
        int j = i + ispn__ * parameters_.num_fv_states();
        if (band_occupancy(j) * weight() > 1e-14) n++;
    }
    return n;
}

};
