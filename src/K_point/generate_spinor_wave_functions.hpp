inline void K_point::generate_spinor_wave_functions()
{
    PROFILE("sirius::K_point::generate_spinor_wave_functions");

    if (use_second_variation) {
        if (!ctx_.need_sv()) {
            /* copy eigen-states and exit */
            spinor_wave_functions(0).copy_from<CPU>(fv_states(), 0, ctx_.num_fv_states(), 0);
            return;
        }

        int nfv = ctx_.num_fv_states();
        int nbnd = (ctx_.num_mag_dims() == 3) ? ctx_.num_bands() : nfv;

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            fv_states().allocate_on_device();
            fv_states().copy_to_device(0, nfv);
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                spinor_wave_functions(ispn).allocate_on_device();
            }
            sv_eigen_vectors_[0].allocate(memory_t::device);
            sv_eigen_vectors_[0].copy_to_device();
            if (ctx_.num_mag_dims() != 3) {
                sv_eigen_vectors_[1].allocate(memory_t::device);
                sv_eigen_vectors_[1].copy_to_device();
            }
        }
        #endif

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int s, o;

            if (ctx_.num_mag_dims() == 3) {
                /* in case of non-collinear magnetism sv_eigen_vectors is a single 2Nx2N matrix */
                s = 0;
                o = ispn * nfv; /* offset for spin up is 0, for spin dn is nfv */
            } else { 
                /* sv_eigen_vectors is composed of two NxN matrices */
                s = ispn;
                o = 0;
            }
            /* multiply consecutively up and dn blocks */
            transform(fv_states(), 0, nfv, sv_eigen_vectors_[s], o, 0, spinor_wave_functions(ispn), 0, nbnd);
        }

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            fv_states().deallocate_on_device();
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                spinor_wave_functions(ispn).copy_to_host(0, nbnd);
                spinor_wave_functions(ispn).deallocate_on_device();
            }
            sv_eigen_vectors_[0].deallocate_on_device();
            if (ctx_.num_mag_dims() != 3) {
                sv_eigen_vectors_[1].deallocate_on_device();
            }
        }
        #endif

    } else {
        TERMINATE_NOT_IMPLEMENTED;
    }
}
