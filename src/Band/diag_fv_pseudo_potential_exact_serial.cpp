#include "band.h"
#include "non_local_operator.h"
#include "hloc_operator.h"

namespace sirius {

void Band::diag_fv_pseudo_potential_exact_serial(K_point* kp__,
                                                 std::vector<double>& veff_it_coarse__)
{
    PROFILE();

    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* short notation for target wave-functions */
    auto psi = kp__->fv_states(); //->primary_data_storage_as_matrix();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();     

    int ngk = kp__->num_gkvec();

    Wave_functions phi(ngk, kp__->gkvec(), ctx_.mpi_grid_fft());
    Wave_functions hphi(ngk, kp__->gkvec(), ctx_.mpi_grid_fft());
    Wave_functions ophi(ngk, kp__->gkvec(), ctx_.mpi_grid_fft());
    mdarray<double_complex, 1> kappa;
    
    std::vector<double> eval(ngk);

    phi.coeffs().zero();
    for (int i = 0; i < ngk; i++) phi(i, i) = complex_one;

    D_operator d_op(kp__->beta_projectors());
    Q_operator q_op(kp__->beta_projectors());

    Hloc_operator h_op(ctx_.fft_coarse_ctx(), kp__->gkvec(), pw_ekin, veff_it_coarse__);

    apply_h_o(kp__, 0, ngk, phi, hphi, ophi, kappa, h_op, d_op, q_op);
        
    Utils::check_hermitian("h", hphi.coeffs(), ngk);
    Utils::check_hermitian("o", ophi.coeffs(), ngk);

    Utils::write_matrix("h.txt", true, hphi.coeffs());
    Utils::write_matrix("o.txt", true, ophi.coeffs());
    auto z1 = hphi.coeffs().checksum();
    auto z2 = ophi.coeffs().checksum();

    printf("checksum(h): %18.10f %18.10f\n", z1.real(), z1.imag());
    printf("checksum(o): %18.10f %18.10f\n", z2.real(), z2.imag());

    if (gen_evp_solver()->solve(ngk, ngk, ngk, num_bands,
                                hphi.coeffs().at<CPU>(),
                                hphi.coeffs().ld(),
                                ophi.coeffs().at<CPU>(),
                                ophi.coeffs().ld(), 
                                &eval[0],
                                psi->coeffs().at<CPU>(),
                                psi->coeffs().ld()))
    {
        TERMINATE("error in evp solve");
    }

    for (int i = 0; i < std::min(ngk, num_bands); i++)
    {
        printf("eval[%i]=%f\n", i, eval[i]);
    }

    kp__->set_fv_eigen_values(&eval[0]);
}

};
