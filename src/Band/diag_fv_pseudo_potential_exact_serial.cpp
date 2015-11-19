#include "band.h"
#include "non_local_operator.h"

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

    Wave_functions phi(ngk, kp__->gkvec(), ctx_.mpi_grid_fft(), true);
    Wave_functions hphi(ngk, kp__->gkvec(), ctx_.mpi_grid_fft(), true);
    Wave_functions ophi(ngk, kp__->gkvec(), ctx_.mpi_grid_fft(), false);
    mdarray<double_complex, 1> kappa;
    
    std::vector<double> eval(ngk);

    phi.primary_data_storage_as_matrix().zero();
    for (int i = 0; i < ngk; i++) phi(i, i) = complex_one;

    D_operator d_op(kp__->beta_projectors());
    Q_operator q_op(kp__->beta_projectors());

    ///* offset in the packed array of on-site matrices */
    //mdarray<int, 1> packed_mtrx_offset(unit_cell_.num_atoms());
    //int packed_mtrx_size = 0;
    //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    //{   
    //    int nbf = unit_cell_.atom(ia)->mt_basis_size();
    //    packed_mtrx_offset(ia) = packed_mtrx_size;
    //    packed_mtrx_size += nbf * nbf;
    //}
    //
    ///* pack Q and D matrices */
    //mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
    //mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);

    //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    //{
    //    int nbf = unit_cell_.atom(ia)->mt_basis_size();
    //    for (int xi2 = 0; xi2 < nbf; xi2++)
    //    {
    //        for (int xi1 = 0; xi1 < nbf; xi1++)
    //        {
    //            d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->d_mtrx(xi1, xi2);
    //            q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = unit_cell_.atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
    //        }
    //    }
    //}
    //
    //STOP();
    
    apply_h_o_serial(kp__, veff_it_coarse__, pw_ekin, 0, ngk, phi, hphi, ophi, kappa, d_op, q_op);
        
    Utils::check_hermitian("h", hphi.primary_data_storage_as_matrix(), ngk);
    Utils::check_hermitian("o", ophi.primary_data_storage_as_matrix(), ngk);

    Utils::write_matrix("h.txt", true, hphi.primary_data_storage_as_matrix());
    Utils::write_matrix("o.txt", true, ophi.primary_data_storage_as_matrix());
    auto z1 = hphi.primary_data_storage_as_matrix().checksum();
    auto z2 = ophi.primary_data_storage_as_matrix().checksum();

    printf("checksum(h): %18.10f %18.10f\n", z1.real(), z1.imag());
    printf("checksum(o): %18.10f %18.10f\n", z2.real(), z2.imag());

    if (gen_evp_solver()->solve(ngk, ngk, ngk, num_bands,
                                hphi.primary_data_storage_as_matrix().at<CPU>(),
                                hphi.primary_data_storage_as_matrix().ld(),
                                ophi.primary_data_storage_as_matrix().at<CPU>(),
                                ophi.primary_data_storage_as_matrix().ld(), 
                                &eval[0],
                                psi->primary_data_storage_as_matrix().at<CPU>(),
                                psi->primary_data_storage_as_matrix().ld()))
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
