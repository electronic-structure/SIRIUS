#include "band.h"
#include "non_local_operator.h"
#include "hloc_operator.h"

namespace sirius {

template <typename T>
void Band::diag_pseudo_potential_exact(K_point* kp__,
                                       int ispn__,
                                       Hloc_operator& h_op__,
                                       D_operator<T>& d_op__,
                                       Q_operator<T>& q_op__) const
{
    PROFILE();

    /* short notation for target wave-functions */
    auto& psi = kp__->spinor_wave_functions<false>(ispn__);

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();     

    int ngk = kp__->num_gkvec();

    auto pu = ctx_.processing_unit();

    Wave_functions<false>  phi(kp__->num_gkvec_loc(), ngk, pu);
    Wave_functions<false> hphi(kp__->num_gkvec_loc(), ngk, pu);
    Wave_functions<false> ophi(kp__->num_gkvec_loc(), ngk, pu);
    
    std::vector<double> eval(ngk);

    phi.coeffs().zero();
    for (int i = 0; i < ngk; i++) phi(i, i) = complex_one;

    apply_h_o(kp__, ispn__, 0, ngk, phi, hphi, ophi, h_op__, d_op__, q_op__);
        
    //Utils::check_hermitian("h", hphi.coeffs(), ngk);
    //Utils::check_hermitian("o", ophi.coeffs(), ngk);

    #ifdef __PRINT_OBJECT_CHECKSUM
    auto z1 = hphi.coeffs().checksum();
    auto z2 = ophi.coeffs().checksum();
    printf("checksum(h): %18.10f %18.10f\n", z1.real(), z1.imag());
    printf("checksum(o): %18.10f %18.10f\n", z2.real(), z2.imag());
    #endif

    if (gen_evp_solver()->solve(ngk, num_bands,
                                hphi.coeffs().at<CPU>(),
                                hphi.coeffs().ld(),
                                ophi.coeffs().at<CPU>(),
                                ophi.coeffs().ld(), 
                                &eval[0],
                                psi.coeffs().at<CPU>(),
                                psi.coeffs().ld()))
    {
        TERMINATE("error in evp solve");
    }

    for (int j = 0; j < ctx_.num_fv_states(); j++)
        kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];
}

template void Band::diag_pseudo_potential_exact<double>(K_point* kp__,
                                                        int ispn__,
                                                        Hloc_operator& h_op__,
                                                        D_operator<double>& d_op__,
                                                        Q_operator<double>& q_op__) const;

template void Band::diag_pseudo_potential_exact<double_complex>(K_point* kp__,
                                                                int ispn__,
                                                                Hloc_operator& h_op__,
                                                                D_operator<double_complex>& d_op__,
                                                                Q_operator<double_complex>& q_op__) const;

};
