#include "k_point.h"

namespace sirius {

void K_point::generate_gkvec(double gk_cutoff)
{
    PROFILE();

    #ifndef __FFTW_MPI
    TERMINATE("FFT with mpi is required");
    #endif

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

    std::vector< std::pair<double, int> > gkmap;

    /* find G-vectors for which |G+k| < cutoff */
    for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++)
    {
        vector3d<double> vgk;
        for (int x = 0; x < 3; x++) vgk[x] = ctx_.gvec()[ig][x] + vk_[x];

        vector3d<double> v = ctx_.unit_cell().reciprocal_lattice_vectors() * vgk;
        double gklen = v.length();

        if (gklen <= gk_cutoff) gkmap.push_back(std::pair<double, int>(gklen, ig));
    }

    //std::sort(gkmap.begin(), gkmap.end());

    gkvec_ = mdarray<double, 2>(3, gkmap.size());
    gvec_index_.resize(gkmap.size());

    for (int igk = 0; igk < (int)gkmap.size(); igk++)
    {
        int ig = gkmap[igk].second;
        gvec_index_[igk] = ig;
        for (int x = 0; x < 3; x++)
        {
            gkvec_(x, igk) = ctx_.gvec()[ig][x] + vk_[x];
        }
    }

    #ifdef __PRINT_OBJECT_CHECKSUM
    DUMP("checksum(gkvec) : %18.10f", gkvec_.checksum());
    #endif
    
    fft_index_.resize(num_gkvec());
    for (int igk = 0; igk < num_gkvec(); igk++) fft_index_[igk] = ctx_.gvec().index_map()[gvec_index_[igk]];

    if (!parameters_.full_potential())
    {
        fft_index_coarse_.resize(num_gkvec());
        for (int igk = 0; igk < num_gkvec(); igk++)
        {
            /* G-vector index in the fine mesh */
            int ig = gvec_index_[igk];
            /* G-vector fractional coordinates */
            vector3d<int> gvec = ctx_.gvec()[ig];
            /* linear index inside coarse FFT buffer */
            fft_index_coarse_[igk] = ctx_.fft_coarse()->index(gvec[0], gvec[1], gvec[2]);
        }
    }
}

};
