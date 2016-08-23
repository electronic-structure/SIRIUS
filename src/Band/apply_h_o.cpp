#include "band.h"

namespace sirius {

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [out] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
template <typename T>
void Band::apply_h_o(K_point* kp__, 
                     int ispn__,
                     int N__,
                     int n__,
                     Wave_functions<false>& phi__,
                     Wave_functions<false>& hphi__,
                     Wave_functions<false>& ophi__,
                     Hloc_operator& h_op,
                     D_operator<T>& d_op,
                     Q_operator<T>& q_op) const
{
    PROFILE_WITH_TIMER("sirius::Band::apply_h_o");

    /* set initial hphi */
    hphi__.copy_from(phi__, N__, n__);

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU && !ctx_.fft_coarse().gpu_only()) {
        hphi__.copy_to_host(N__, n__);
    }
    #endif
    /* apply local part of Hamiltonian */
    h_op.apply(ispn__, hphi__, N__, n__);
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU && !ctx_.fft_coarse().gpu_only()) {
        hphi__.copy_to_device(N__, n__);
    }
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            phi__.copy_to_host(N__, n__);
            if (ctx_.fft_coarse().gpu_only()) {
                hphi__.copy_to_host(N__, n__);
            }
        }
        #endif
        auto cs1 = mdarray<double_complex, 1>(&phi__(0, N__), kp__->num_gkvec_loc() * n__).checksum();
        auto cs2 = mdarray<double_complex, 1>(&hphi__(0, N__), kp__->num_gkvec_loc() * n__).checksum();
        kp__->comm().allreduce(&cs1, 1);
        kp__->comm().allreduce(&cs2, 1);
        DUMP("checksum(phi): %18.10f %18.10f", cs1.real(), cs1.imag());
        DUMP("checksum(hloc_phi): %18.10f %18.10f", cs2.real(), cs2.imag());
    }
    #endif

    /* set intial ophi */
    ophi__.copy_from(phi__, N__, n__);

    if (!ctx_.unit_cell().mt_lo_basis_size()) {
        return;
    }

    for (int i = 0; i < kp__->beta_projectors().num_beta_chunks(); i++) {
        kp__->beta_projectors().generate(i);

        kp__->beta_projectors().inner<T>(i, phi__, N__, n__);

        if (!ctx_.iterative_solver_input_section().real_space_prj_) {
            d_op.apply(i, ispn__, hphi__, N__, n__);
            q_op.apply(i, 0, ophi__, N__, n__);
        } else {
            STOP();
            //add_nl_h_o_rs(kp__, n__, phi, hphi, ophi, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__, kappa__);
        }
    }

    //== if (!kp__->gkvec().reduced())
    //== {
    //==     // --== DEBUG ==--
    //==     printf("check in apply_h_o\n");
    //==     for (int i = N__; i < N__ + n__; i++)
    //==     {
    //==         bool f1 = false;
    //==         bool f2 = false;
    //==         bool f3 = false;
    //==         double e1 = 0;
    //==         double e2 = 0;
    //==         double e3 = 0;
    //==         for (int igk = 0; igk < kp__->num_gkvec(); igk++)
    //==         {
    //==             auto G = kp__->gkvec()[igk] * (-1);
    //==             int igk1 = kp__->gkvec().index_by_gvec(G);
    //==             if (std::abs(phi__(igk, i) - std::conj(phi__(igk1, i))) > 1e-12)
    //==             {
    //==                 f1 = true;
    //==                 e1 = std::max(e1, std::abs(phi__(igk, i) - std::conj(phi__(igk1, i))));
    //==             }
    //==             if (std::abs(hphi__(igk, i) - std::conj(hphi__(igk1, i))) > 1e-12)
    //==             {
    //==                 f2 = true;
    //==                 e2 = std::max(e2, std::abs(hphi__(igk, i) - std::conj(hphi__(igk1, i))));
    //==             }
    //==             if (std::abs(ophi__(igk, i) - std::conj(ophi__(igk1, i))) > 1e-12)
    //==             {
    //==                 f3 = true;
    //==                 e3 = std::max(e3, std::abs(ophi__(igk, i) - std::conj(ophi__(igk1, i))));
    //==             }
    //==         }
    //==         if (f1) printf("phi[%i] is not real, %20.16f\n", i, e1);
    //==         if (f2) printf("hphi[%i] is not real, %20.16f\n", i, e2);
    //==         if (f3) printf("ophi[%i] is not real, %20.16f\n", i, e3);
    //==     }
    //==     printf("done.\n");
    //== }
}

template void Band::apply_h_o<double_complex>(K_point* kp__, 
                                              int ispn__,
                                              int N__,
                                              int n__,
                                              Wave_functions<false>& phi__,
                                              Wave_functions<false>& hphi__,
                                              Wave_functions<false>& ophi__,
                                              Hloc_operator& h_op,
                                              D_operator<double_complex>& d_op,
                                              Q_operator<double_complex>& q_op) const;

template void Band::apply_h_o<double>(K_point* kp__, 
                                      int ispn__,
                                      int N__,
                                      int n__,
                                      Wave_functions<false>& phi__,
                                      Wave_functions<false>& hphi__,
                                      Wave_functions<false>& ophi__,
                                      Hloc_operator& h_op,
                                      D_operator<double>& d_op,
                                      Q_operator<double>& q_op) const;
};
