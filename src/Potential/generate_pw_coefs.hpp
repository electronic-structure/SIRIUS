inline void Potential::generate_pw_coefs()
{
    PROFILE("sirius::Potential::generate_pw_coefs");

    ctx_.fft().prepare(ctx_.gvec().partition());

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
            if (ctx_.fft().hybrid()) {
                ctx_.fft().copy_to_device();
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
            if (ctx_.fft().hybrid()) {
                ctx_.fft().copy_to_device();
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
            if (ctx_.fft().hybrid()) {
                ctx_.fft().copy_to_device();
            }
            #endif
            ctx_.fft().transform<-1>(ctx_.gvec().partition(), &veff_pw_[gv_offset]);
            ctx_.fft().comm().allgather(&veff_pw_[0], gv_offset, gv_count);
        }
    }

    //#ifdef __PRINT_OBJECT_CHECKSUM
    //double_complex z2 = mdarray<double_complex, 1>(&ctx_.fft().buffer(0), ctx_.fft().local_size()).checksum();
    //DUMP("checksum(veff_it): %18.10f", mdarray<double, 1>(&effective_potential()->f_rg(0) , ctx_.fft().local_size()).checksum());
    //DUMP("checksum(fft_buffer): %18.10f %18.10f", z2.real(), z2.imag());
    //#endif
    
    //ctx_.fft().transform<-1>(ctx_.gvec().partition(), &effective_potential()->f_pw(ctx_.gvec().partition().gvec_offset_fft()));
    //ctx_.fft().comm().allgather(&effective_potential()->f_pw(0), ctx_.gvec().partition().gvec_offset_fft(), ctx_.gvec().partition().gvec_count_fft());
    


    

    //#ifdef __PRINT_OBJECT_CHECKSUM
    //double_complex z1 = mdarray<double_complex, 1>(&effective_potential()->f_pw(0), ctx_.gvec().num_gvec()).checksum();
    //DUMP("checksum(veff_pw): %18.10f %18.10f", z1.real(), z1.imag());
    //#endif
    
    /* for full diagonalization we also need Beff(G) */
    if (!use_second_variation) {
        TERMINATE_NOT_IMPLEMENTED
        //for (int i = 0; i < ctx_.num_mag_dims(); i++) {
        //    for (int ir = 0; ir < ctx_.fft().size(); ir++) {
        //        ctx_.fft().buffer(ir) = effective_magnetic_field(i)->f_rg(ir) * ctx_.step_function().theta_r(ir);
        //    }
        //    //fft_.transform(-1, ctx_.gvec().z_sticks_coord());
        //    //fft_.output(ctx_.gvec().num_gvec(), ctx_.gvec().index_map(), &effective_magnetic_field(i)->f_pw(0));
        //}
    }

    //if (ctx_.esm_type() == electronic_structure_method_t::full_potential_pwlo) {
    //    switch (ctx_.processing_unit()) {
    //        case CPU:
    //            STOP();
    //            //add_mt_contribution_to_pw<CPU>();
    //            break;
    //        #ifdef __GPU
    //        //== case GPU:
    //        //== {
    //        //==     add_mt_contribution_to_pw<GPU>();
    //        //==     break;
    //        //== }
    //        #endif
    //        default:
    //            TERMINATE("wrong processing unit");
    //    }
    //}

    ctx_.fft().dismiss();
}
