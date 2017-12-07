inline void K_point::generate_spinor_wave_functions()
{
    PROFILE("sirius::K_point::generate_spinor_wave_functions");

    if (use_second_variation) {
        int nfv = ctx_.num_fv_states();

        if (!ctx_.need_sv()) {
            /* copy eigen-states and exit */
            spinor_wave_functions().copy_from(CPU, ctx_.num_fv_states(), fv_states(), 0, 0, 0, 0);
            #ifdef __GPU
            if (ctx_.processing_unit() == GPU && keep_wf_on_gpu) {
                spinor_wave_functions().copy_to_device(0, 0, ctx_.num_fv_states());
            }
            #endif
            return;
        }

        int nbnd = (ctx_.num_mag_dims() == 3) ? ctx_.num_bands() : nfv;

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            fv_states().allocate_on_device(0);
            fv_states().copy_to_device(0, 0, nfv);
            sv_eigen_vectors_[0].allocate(memory_t::device);
            sv_eigen_vectors_[0].copy_to_device();
            if (ctx_.num_mag_dims() == 1) {
                sv_eigen_vectors_[1].allocate(memory_t::device);
                sv_eigen_vectors_[1].copy_to_device();
            }
            if (!keep_wf_on_gpu) {
                for (int ispn = 0; ispn < ctx_.num_mag_dims(); ispn++) {
                    spinor_wave_functions().allocate_on_device(ispn);
                    spinor_wave_functions().copy_to_device(ispn, 0, nbnd);
                }
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
            transform(ctx_.processing_unit(), ispn, fv_states(), 0, nfv, sv_eigen_vectors_[s], o, 0, spinor_wave_functions(), 0, nbnd);
        }

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            fv_states().deallocate_on_device(0);
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                spinor_wave_functions().copy_to_host(ispn, 0, nbnd);
            }
            sv_eigen_vectors_[0].deallocate_on_device();
            if (ctx_.num_mag_dims() == 3) {
                sv_eigen_vectors_[1].deallocate_on_device();
            }
            if (!keep_wf_on_gpu) {
                for (int ispn = 0; ispn < ctx_.num_mag_dims(); ispn++) {
                    spinor_wave_functions().deallocate_on_device(ispn);
                }
            }
        }
        #endif

    } else {
        TERMINATE_NOT_IMPLEMENTED;
    }
}
