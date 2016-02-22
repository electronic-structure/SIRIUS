#include "band.h"

namespace sirius {

void Band::diag_pseudo_potential(K_point* kp__, 
                                 Periodic_function<double>* effective_potential__,
                                 Periodic_function<double>* effective_magnetic_field__[3])
{
    PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential");

    ctx_.fft_coarse().prepare();

    Hloc_operator hloc(ctx_.fft_coarse(), ctx_.gvec_coarse(), kp__->gkvec(), ctx_.num_mag_dims(),
                       effective_potential__, effective_magnetic_field__);
    
    D_operator d_op(ctx_, kp__->beta_projectors());
    Q_operator q_op(ctx_, kp__->beta_projectors());

    //if (itso.type_ == "exact")
    //{
    //    diag_fv_pseudo_potential_exact_serial(kp__, veff_it_coarse);
    //}
    //else if (itso.type_ == "davidson")
    //{
    //    diag_fv_pseudo_potential_davidson(kp__, v0, veff_it_coarse);
    //}
    //else if (itso.type_ == "rmm-diis")
    //{
    //    diag_fv_pseudo_potential_rmm_diis_serial(kp__, v0, veff_it_coarse);
    //}
    //else if (itso.type_ == "chebyshev")
    //{
    //    diag_fv_pseudo_potential_chebyshev_serial(kp__, veff_it_coarse);
    //}
    //else
    //{
    //    TERMINATE("unknown iterative solver type");
    //}


    auto& itso = ctx_.iterative_solver_input_section();
    if (itso.type_ == "exact")
    {
        if (ctx_.num_mag_dims() != 3)
        {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
                diag_pseudo_potential_exact(kp__, ispn, hloc, d_op, q_op);
        }
        else
        {
            STOP();
        }
    }
    else if (itso.type_ == "davidson")
    {
        if (ctx_.num_mag_dims() != 3)
        {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
                diag_pseudo_potential_davidson(kp__, ispn, hloc, d_op, q_op);
        }
        else
        {
            STOP();
        }
    }
    else if (itso.type_ == "davidson_fast")
    {
        if (ctx_.num_mag_dims() != 3)
        {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
                diag_pseudo_potential_davidson_fast(kp__, ispn, hloc, d_op, q_op);
        }
        else
        {
            STOP();
        }
    }
    else
    {
        TERMINATE("unknown iterative solver type");
    }

    ctx_.fft_coarse().dismiss();
}

};
