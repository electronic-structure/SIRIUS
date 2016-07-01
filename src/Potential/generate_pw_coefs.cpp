#include "potential.h"

namespace sirius {

void Potential::generate_pw_coefs()
{
    PROFILE_WITH_TIMER("sirius::Potential::generate_pw_coefs");

    fft_.prepare();

    for (int ir = 0; ir < fft_.local_size(); ir++) {
        fft_.buffer(ir) = effective_potential()->f_rg(ir) * ctx_.step_function().theta_r(ir);
    }

    #ifdef __GPU
    if (ctx_.processing_unit()) {
        fft_.copy_to_device();
    }
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex z2 = mdarray<double_complex, 1>(&fft_.buffer(0), fft_.local_size()).checksum();
    DUMP("checksum(veff_it): %18.10f", mdarray<double, 1>(&effective_potential()->f_rg(0) , fft_.local_size()).checksum());
    DUMP("checksum(fft_buffer): %18.10f %18.10f", z2.real(), z2.imag());
    #endif
    
    fft_.transform<-1>(ctx_.gvec_fft_distr(), &effective_potential()->f_pw(ctx_.gvec_fft_distr().offset_gvec_fft()));
    fft_.comm().allgather(&effective_potential()->f_pw(0), ctx_.gvec_fft_distr().offset_gvec_fft(), ctx_.gvec_fft_distr().num_gvec_fft());

    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex z1 = mdarray<double_complex, 1>(&effective_potential()->f_pw(0), ctx_.gvec().num_gvec()).checksum();
    DUMP("checksum(veff_pw): %18.10f %18.10f", z1.real(), z1.imag());
    #endif
    
    /* for full diagonalization we also need Beff(G) */
    if (!use_second_variation) {
        for (int i = 0; i < ctx_.num_mag_dims(); i++) {
            for (int ir = 0; ir < fft_.size(); ir++) {
                fft_.buffer(ir) = effective_magnetic_field(i)->f_rg(ir) * ctx_.step_function().theta_r(ir);
            }
            STOP();
            //fft_.transform(-1, ctx_.gvec().z_sticks_coord());
            //fft_.output(ctx_.gvec().num_gvec(), ctx_.gvec().index_map(), &effective_magnetic_field(i)->f_pw(0));
        }
    }

    if (ctx_.esm_type() == full_potential_pwlo) {
        switch (ctx_.processing_unit()) {
            case CPU:
                STOP();
                //add_mt_contribution_to_pw<CPU>();
                break;
            #ifdef __GPU
            //== case GPU:
            //== {
            //==     add_mt_contribution_to_pw<GPU>();
            //==     break;
            //== }
            #endif
            default:
                TERMINATE("wrong processing unit");
        }
    }

    fft_.dismiss();
}

};
