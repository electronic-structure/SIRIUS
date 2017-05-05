inline void Potential::generate_pw_coefs()
{
    PROFILE("sirius::Potential::generate_pw_coefs");

    double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);

    int gv_count  = ctx_.gvec().partition().gvec_count_fft();
    int gv_offset = ctx_.gvec().partition().gvec_offset_fft();

    switch (ctx_.valence_relativity()) {
        case relativity_t::iora: {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                double M = 1 - sq_alpha_half * effective_potential()->f_rg(ir);
                ctx_.fft().buffer(ir) = ctx_.step_function().theta_r(ir) / std::pow(M, 2);
            }
            #ifdef __GPU
            if (ctx_.fft().pu() == GPU) {
                ctx_.fft().buffer().copy_to_device();
            }
            #endif
            ctx_.fft().transform<-1>(ctx_.gvec().partition(), &rm2_inv_pw_[gv_offset]);
            ctx_.fft().comm().allgather(&rm2_inv_pw_[0], gv_offset, gv_count);
        }
        case relativity_t::zora: {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                double M = 1 - sq_alpha_half * effective_potential()->f_rg(ir);
                ctx_.fft().buffer(ir) = ctx_.step_function().theta_r(ir) / M;
            }
            #ifdef __GPU
            if (ctx_.fft().pu() == GPU) {
                ctx_.fft().buffer().copy_to_device();
            }
            #endif
            ctx_.fft().transform<-1>(ctx_.gvec().partition(), &rm_inv_pw_[gv_offset]);
            ctx_.fft().comm().allgather(&rm_inv_pw_[0], gv_offset, gv_count);
        }
        default: {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                ctx_.fft().buffer(ir) = effective_potential()->f_rg(ir) * ctx_.step_function().theta_r(ir);
            }
            #ifdef __GPU
            if (ctx_.fft().pu() == GPU) {
                ctx_.fft().buffer().copy_to_device();
            }
            #endif
            ctx_.fft().transform<-1>(ctx_.gvec().partition(), &veff_pw_[gv_offset]);
            ctx_.fft().comm().allgather(&veff_pw_[0], gv_offset, gv_count);
        }
    }

    /* for full diagonalization we also need Beff(G) */
    if (!use_second_variation) {
        TERMINATE_NOT_IMPLEMENTED
    }
}
