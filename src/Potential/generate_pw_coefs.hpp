inline void Potential::generate_pw_coefs()
{
    PROFILE("sirius::Potential::generate_pw_coefs");

    double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);

    int gv_count  = ctx_.gvec_partition().gvec_count_fft();
    
    /* temporaty output buffer */
    mdarray<double_complex, 1> fpw_fft(gv_count);

    switch (ctx_.valence_relativity()) {
        case relativity_t::iora: {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                double M = 1 - sq_alpha_half * effective_potential()->f_rg(ir);
                ctx_.fft().buffer(ir) = ctx_.step_function().theta_r(ir) / std::pow(M, 2);
            }
            if (ctx_.fft().pu() == GPU) {
                ctx_.fft().buffer().copy<memory_t::host, memory_t::device>();
            }
            ctx_.fft().transform<-1>(&fpw_fft[0]);
            ctx_.gvec_partition().gather_pw_global(&fpw_fft[0], &rm2_inv_pw_[0]);
        }
        case relativity_t::zora: {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                double M = 1 - sq_alpha_half * effective_potential()->f_rg(ir);
                ctx_.fft().buffer(ir) = ctx_.step_function().theta_r(ir) / M;
            }
            if (ctx_.fft().pu() == GPU) {
                ctx_.fft().buffer().copy<memory_t::host, memory_t::device>();
            }
            ctx_.fft().transform<-1>(&fpw_fft[0]);
            ctx_.gvec_partition().gather_pw_global(&fpw_fft[0], &rm_inv_pw_[0]);
        }
        default: {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
                ctx_.fft().buffer(ir) = effective_potential()->f_rg(ir) * ctx_.step_function().theta_r(ir);
            }
            if (ctx_.fft().pu() == GPU) {
                ctx_.fft().buffer().copy<memory_t::host, memory_t::device>();
            }
            ctx_.fft().transform<-1>(&fpw_fft[0]);
            ctx_.gvec_partition().gather_pw_global(&fpw_fft[0], &veff_pw_[0]);
        }
    }

    /* for full diagonalization we also need Beff(G) */
    if (!use_second_variation) {
        TERMINATE_NOT_IMPLEMENTED
    }
}
