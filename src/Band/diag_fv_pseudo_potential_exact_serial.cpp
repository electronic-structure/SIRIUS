#include "band.h"

namespace sirius {

void Band::diag_fv_pseudo_potential_exact_serial(K_point* kp__,
                                                 std::vector<double>& veff_it_coarse__)
{
    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* short notation for target wave-functions */
    auto& psi = kp__->fv_states()->primary_data_storage_as_matrix();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();     

    int ngk = kp__->num_gkvec();

    mdarray<double_complex, 2> phi(ngk, ngk);
    mdarray<double_complex, 2> hphi(ngk, ngk);
    mdarray<double_complex, 2> ophi(ngk, ngk);
    mdarray<double_complex, 1> kappa;
    
    std::vector<double> eval(ngk);

    phi.zero();
    for (int i = 0; i < ngk; i++) phi(i, i) = complex_one;

    /* offset in the packed array of on-site matrices */
    mdarray<int, 1> packed_mtrx_offset(unit_cell_.num_atoms());
    int packed_mtrx_size = 0;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    {   
        int nbf = unit_cell_.atom(ia)->mt_basis_size();
        packed_mtrx_offset(ia) = packed_mtrx_size;
        packed_mtrx_size += nbf * nbf;
    }
    
    /* pack Q and D matrices */
    mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
    mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    {
        int nbf = unit_cell_.atom(ia)->mt_basis_size();
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->d_mtrx(xi1, xi2);
                q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
            }
        }
    }
    
    apply_h_o_serial(kp__, veff_it_coarse__, pw_ekin, 0, ngk, phi, hphi, ophi, kappa, packed_mtrx_offset,
                     d_mtrx_packed, q_mtrx_packed);
        
    Utils::check_hermitian("h", hphi, ngk);
    Utils::check_hermitian("o", ophi, ngk);

    Utils::write_matrix("h.txt", true, hphi);
    Utils::write_matrix("o.txt", true, ophi);
    auto z1 = hphi.checksum();
    auto z2 = ophi.checksum();

    printf("checksum(h): %18.10f %18.10f\n", z1.real(), z1.imag());
    printf("checksum(o): %18.10f %18.10f\n", z2.real(), z2.imag());

    if (gen_evp_solver()->solve(ngk, ngk, ngk, num_bands, hphi.at<CPU>(), hphi.ld(), ophi.at<CPU>(), ophi.ld(), 
                                &eval[0], psi.at<CPU>(), psi.ld()))
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
