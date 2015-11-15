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
        warning_local(__FILE__, __LINE__, s);
    }

    if (gk_cutoff * 2 > parameters_.pw_cutoff())
    {
        std::stringstream s;
        s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
          << "  pw cutoff : " << parameters_.pw_cutoff() << std::endl
          << "  doubled G+k cutoff : " << gk_cutoff * 2;
        error_local(__FILE__, __LINE__, s);
    }
    
    /* create G+k vectors using fine FFT grid; 
     * this would provide a correct mapping between \psi(G) and \rho(r) */
    gkvec_ = Gvec(vk_, ctx_.unit_cell().reciprocal_lattice_vectors(), gk_cutoff, ctx_.fft(0)->fft_grid(),
                  ctx_.fft(0)->comm(), blacs_grid_.num_ranks_col(), false, false);
   
    ///* additionally create mapping between \psi(G) and a coarse FFT buffer in order to apply H_loc */
    //if (!parameters_.full_potential())
    //{
    //    gkvec_coarse_ = Gvec(vk_, ctx_.unit_cell().reciprocal_lattice_vectors(), gk_cutoff,
    //                         ctx_.fft_coarse(0)->fft_grid(), ctx_.fft_coarse(0)->comm(), -1, false, false);
    //}
}

};
