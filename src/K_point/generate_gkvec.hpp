inline void K_point::generate_gkvec(double gk_cutoff__)
{
    PROFILE("sirius::K_point::generate_gkvec");

    if (ctx_.full_potential() && (gk_cutoff__ * unit_cell_.max_mt_radius() > ctx_.lmax_apw())) {
        std::stringstream s;
        s << "G+k cutoff (" << gk_cutoff__ << ") is too large for a given lmax (" 
          << ctx_.lmax_apw() << ") and a maximum MT radius (" << unit_cell_.max_mt_radius() << ")" << std::endl
          << "suggested minimum value for lmax : " << int(gk_cutoff__ * unit_cell_.max_mt_radius()) + 1;
        WARNING(s);
    }

    if (gk_cutoff__ * 2 > ctx_.pw_cutoff()) {
        std::stringstream s;
        s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
          << "  pw cutoff : " << ctx_.pw_cutoff() << std::endl
          << "  doubled G+k cutoff : " << gk_cutoff__ * 2;
        TERMINATE(s);
    }

    /* create G+k vectors; communicator of the coarse FFT grid is used because wave-functions will be transformed 
     * only on the coarse grid; G+k-vectors will be distributed between MPI ranks assigned to the k-point */
    gkvec_ = Gvec(vk_, ctx_.unit_cell().reciprocal_lattice_vectors(), gk_cutoff__, comm(), ctx_.comm_fft_coarse(), ctx_.gamma_point());
}
