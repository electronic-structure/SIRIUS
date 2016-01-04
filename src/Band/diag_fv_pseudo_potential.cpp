#include "band.h"

namespace sirius {

//== void Band::diag_fv_pseudo_potential(K_point* kp__, 
//==                                     Periodic_function<double>* effective_potential__)
//== {
//==     PROFILE_WITH_TIMER("sirius::Band::diag_fv_pseudo_potential");
//== 
//==     auto fft_coarse = ctx_.fft_coarse_ctx().fft();
//==     auto& gv = ctx_.gvec();
//==     auto& gvc = ctx_.gvec_coarse();
//== 
//==     ctx_.fft_coarse_ctx().allocate_workspace();
//== 
//==     /* map effective potential to a corase grid */
//==     std::vector<double> veff_it_coarse(fft_coarse->local_size());
//==     std::vector<double_complex> veff_pw_coarse(gvc.num_gvec_fft());
//== 
//==     for (int ig = 0; ig < gvc.num_gvec_fft(); ig++)
//==     {
//==         auto G = gvc[ig + gvc.offset_gvec_fft()];
//==         veff_pw_coarse[ig] = effective_potential__->f_pw(gv.index_by_gvec(G));
//==     }
//==     fft_coarse->transform<1>(gvc, &veff_pw_coarse[0]);
//==     fft_coarse->output(&veff_it_coarse[0]);
//== 
//==     #ifdef __PRINT_OBJECT_CHECKSUM
//==     double cs = mdarray<double, 1>(&veff_it_coarse[0], veff_it_coarse.size()).checksum();
//==     DUMP("checksum(veff_it_coarse): %18.10f", cs);
//==     #endif
//== 
//==     double v0 = effective_potential__->f_pw(0).real();
//== 
//==     auto& itso = parameters_.iterative_solver_input_section();
//==     if (itso.type_ == "exact")
//==     {
//==         diag_fv_pseudo_potential_exact_serial(kp__, veff_it_coarse);
//==     }
//==     else if (itso.type_ == "davidson")
//==     {
//==         diag_fv_pseudo_potential_davidson(kp__, v0, veff_it_coarse);
//==     }
//==     else if (itso.type_ == "rmm-diis")
//==     {
//==         diag_fv_pseudo_potential_rmm_diis_serial(kp__, v0, veff_it_coarse);
//==     }
//==     else if (itso.type_ == "chebyshev")
//==     {
//==         diag_fv_pseudo_potential_chebyshev_serial(kp__, veff_it_coarse);
//==     }
//==     else
//==     {
//==         TERMINATE("unknown iterative solver type");
//==     }
//== 
//==     ctx_.fft_coarse_ctx().deallocate_workspace();
//== }

};
