#include "k_point.h"

namespace sirius {

void K_point::generate_gkvec(double gk_cutoff)
{
    PROFILE();

    if ((gk_cutoff * unit_cell_.max_mt_radius() > double(parameters_.lmax_apw())) && 
        parameters_.full_potential())
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
    gkvec_ = Gvec(vk_, gk_cutoff, ctx_.unit_cell().reciprocal_lattice_vectors(), fft_, false);
    
    /* additionally create mapping between \psi(G) and a coarse FFT buffer in order to apply H_loc */
    if (!parameters_.full_potential())
        gkvec_coarse_ = Gvec(vk_, gk_cutoff, ctx_.unit_cell().reciprocal_lattice_vectors(), ctx_.fft_coarse(), false);
}

};
