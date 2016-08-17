#include "k_point.h"

namespace sirius {

void K_point::generate_gkvec(double gk_cutoff)
{
    PROFILE();

    if (ctx_.full_potential() && (gk_cutoff * unit_cell_.max_mt_radius() > ctx_.lmax_apw()))
    {
        std::stringstream s;
        s << "G+k cutoff (" << gk_cutoff << ") is too large for a given lmax (" 
          << ctx_.lmax_apw() << ") and a maximum MT radius (" << unit_cell_.max_mt_radius() << ")" << std::endl
          << "suggested minimum value for lmax : " << int(gk_cutoff * unit_cell_.max_mt_radius()) + 1;
        WARNING(s);
    }

    if (gk_cutoff * 2 > ctx_.pw_cutoff())
    {
        std::stringstream s;
        s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
          << "  pw cutoff : " << ctx_.pw_cutoff() << std::endl
          << "  doubled G+k cutoff : " << gk_cutoff * 2;
        TERMINATE(s);
    }
    
    /* create G+k vectors */
    gkvec_ = Gvec(vk_, ctx_.unit_cell().reciprocal_lattice_vectors(), gk_cutoff, ctx_.fft().grid(), num_ranks(),
                  ctx_.mpi_grid_fft().communicator(1 << 1), ctx_.gamma_point());
    
    if (!ctx_.full_potential()) {
        gkvec_vloc_ = Gvec(vk_, ctx_.unit_cell().reciprocal_lattice_vectors(), gk_cutoff, ctx_.fft().grid(), num_ranks(),
                           ctx_.mpi_grid_fft_vloc().communicator(1 << 1), ctx_.gamma_point());
        #pragma omp parallel for
        for (int ig = 0; ig < gkvec_.num_gvec(); ig++) {
            auto v1 = gkvec_.gvec(ig);
            auto v2 = gkvec_vloc_.gvec(ig);
            if (!(v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2])) {
                TERMINATE("wrong order of G+k vectors");
            }
        }
    }
}

};
