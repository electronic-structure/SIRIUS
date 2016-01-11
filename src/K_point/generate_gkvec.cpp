#include "k_point.h"

namespace sirius {

void K_point::generate_gkvec(double gk_cutoff)
{
    PROFILE();

    if (parameters_.full_potential() && (gk_cutoff * unit_cell_.max_mt_radius() > parameters_.lmax_apw()))
    {
        std::stringstream s;
        s << "G+k cutoff (" << gk_cutoff << ") is too large for a given lmax (" 
          << parameters_.lmax_apw() << ") and a maximum MT radius (" << unit_cell_.max_mt_radius() << ")" << std::endl
          << "suggested minimum value for lmax : " << int(gk_cutoff * unit_cell_.max_mt_radius()) + 1;
        WARNING(s);
    }

    if (gk_cutoff * 2 > parameters_.pw_cutoff())
    {
        std::stringstream s;
        s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
          << "  pw cutoff : " << parameters_.pw_cutoff() << std::endl
          << "  doubled G+k cutoff : " << gk_cutoff * 2;
        TERMINATE(s);
    }
    
    /* create G+k vectors */
    gkvec_ = Gvec(vk_, ctx_.unit_cell().reciprocal_lattice_vectors(), gk_cutoff, ctx_.fft(0)->grid(),
                  ctx_.mpi_grid_fft().communicator(1 << 1), ctx_.mpi_grid_fft().dimension_size(0), false, false);
}

};
