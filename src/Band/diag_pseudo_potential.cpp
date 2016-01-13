#include "band.h"

namespace sirius {

void Band::diag_pseudo_potential(K_point* kp__, 
                                 Periodic_function<double>* effective_potential__,
                                 Periodic_function<double>* effective_magnetic_field__[3])
{
    PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential");

    ctx_.fft_coarse_ctx().prepare();

    Hloc_operator hloc(ctx_.fft_coarse_ctx(), ctx_.gvec_coarse(), kp__->gkvec(), ctx_.num_mag_dims(),
                       effective_potential__, effective_magnetic_field__);
    
    auto pu = ctx_.processing_unit();
    D_operator d_op(kp__->beta_projectors(), ctx_.num_mag_dims(), pu);
    Q_operator q_op(ctx_, kp__->beta_projectors(), pu);

    //== auto h_diag1 = get_h_diag(kp__, 0, hloc.v0(0), d_op);
    //== auto o_diag1 = get_o_diag(kp__, q_op);
    //== auto h_diag2 = get_h_diag(kp__, 1, hloc.v0(1), d_op);
    //== auto o_diag2 = get_o_diag(kp__, q_op);

    //== for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++)
    //== {
    //==     if (std::abs(h_diag1[ig] - h_diag2[ig]) > 1e-10) printf("wrong hdiag!!!!\n");
    //==     if (std::abs(o_diag1[ig] - o_diag2[ig]) > 1e-10) printf("wrong odiag!!!!\n");
    //== }

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
    else
    {
        TERMINATE("unknown iterative solver type");
    }

    ctx_.fft_coarse_ctx().dismiss();
}

};
