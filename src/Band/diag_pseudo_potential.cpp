#include "band.h"

namespace sirius {

template <typename T>
void Band::diag_pseudo_potential(K_point* kp__, 
                                 Periodic_function<double>* effective_potential__,
                                 Periodic_function<double>* effective_magnetic_field__[3]) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential");
   
    ctx_.fft_coarse().prepare();

    Hloc_operator hloc(ctx_.fft_coarse(), ctx_.gvec_coarse_fft_distr(), kp__->gkvec_fft_distr_vloc(), ctx_.num_mag_dims(),
                       effective_potential__, effective_magnetic_field__);
    
    D_operator<T> d_op(ctx_, kp__->beta_projectors());
    Q_operator<T> q_op(ctx_, kp__->beta_projectors());

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
    else if (itso.type_ == "rmm-diis")
    {
        if (ctx_.num_mag_dims() != 3)
        {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
                diag_pseudo_potential_rmm_diis(kp__, ispn, hloc, d_op, q_op);
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

template void Band::diag_pseudo_potential<double>(K_point* kp__, 
                                                  Periodic_function<double>* effective_potential__,
                                                  Periodic_function<double>* effective_magnetic_field__[3]) const;

template void Band::diag_pseudo_potential<double_complex>(K_point* kp__, 
                                                          Periodic_function<double>* effective_potential__,
                                                          Periodic_function<double>* effective_magnetic_field__[3]) const;
};
