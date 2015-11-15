#include "band.h"

namespace sirius {

void Band::solve_sv(K_point* kp, Periodic_function<double>* effective_magnetic_field[3])
{
    PROFILE();

    Timer t("sirius::Band::solve_sv");

    if (!parameters_.need_sv())
    {
        kp->bypass_sv();
        return;
    }
    
    if (kp->num_ranks() > 1 && !std_evp_solver()->parallel()) TERMINATE("eigen-value solver is not parallel");

    /* number of h|\psi> components */
    int nhpsi = parameters_.num_mag_dims() + 1;
    if (!std_evp_solver()->parallel() && parameters_.num_mag_dims() == 3) nhpsi = 3;

    /* size of the first-variational state */
    int fvsz = kp->wf_size();

    std::vector<double> band_energies(parameters_.num_bands());

    /* product of the second-variational Hamiltonian and a wave-function */
    std::vector< dmatrix<double_complex> > hpsi_slice(nhpsi);
    for (int i = 0; i < nhpsi; i++)
    {
        hpsi_slice[i] = dmatrix<double_complex>(fvsz, parameters_.num_fv_states(), kp->blacs_grid_slice(), 1, 1);
    }

    /* compute product of magnetic field and wave-function */
    if (parameters_.num_spins() == 2)
    {
        STOP();
        //apply_magnetic_field(kp->fv_states_slice(), kp->num_gkvec(), kp->gkvec().index_map(), effective_magnetic_field, hpsi_slice);
    }

    //== if (parameters_.uj_correction())
    //== {
    //==     apply_uj_correction<uu>(kp->fv_states_col(), hpsi);
    //==     if (parameters_.num_mag_dims() != 0) apply_uj_correction<dd>(kp->fv_states_col(), hpsi);
    //==     if (parameters_.num_mag_dims() == 3) 
    //==     {
    //==         apply_uj_correction<ud>(kp->fv_states_col(), hpsi);
    //==         if (parameters_.std_evp_solver()->parallel()) apply_uj_correction<du>(kp->fv_states_col(), hpsi);
    //==     }
    //== }

    //== if (parameters_.so_correction()) apply_so_correction(kp->fv_states_col(), hpsi);

    std::vector< dmatrix<double_complex> > hpsi(nhpsi);
    for (int i = 0; i < nhpsi; i++)
    {
        hpsi[i] = dmatrix<double_complex>(fvsz, parameters_.num_fv_states(), kp->blacs_grid(),
                                          parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
        /* change data distribution of hpsi to panels */
        linalg<CPU>::gemr2d(fvsz, parameters_.num_fv_states(),
                            hpsi_slice[i], 0, 0,
                            hpsi[i], 0, 0, kp->blacs_grid().context());

    }

    if (parameters_.processing_unit() == GPU && kp->num_ranks() == 1)
    {
        #ifdef __GPU
        kp->fv_states().allocate_on_device();
        kp->fv_states().copy_to_device();
        #endif
    }

    #ifdef __GPU
    double_complex alpha = complex_one;
    double_complex beta = complex_zero;
    #endif

    if (parameters_.num_mag_dims() != 3)
    {
        dmatrix<double_complex> h(parameters_.num_fv_states(), parameters_.num_fv_states(), kp->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
        if (parameters_.processing_unit() == GPU && kp->num_ranks() == 1)
        {
            #ifdef __GPU
            h.allocate_on_device();
            #endif
        }

        /* perform one or two consecutive diagonalizations */
        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
        {
            if (parameters_.processing_unit() == GPU && kp->num_ranks() == 1)
            {
                #ifdef __GPU
                Timer t4("sirius::Band::solve_sv|zgemm");
                hpsi[ispn].allocate_on_device();
                hpsi[ispn].copy_to_device();
                linalg<GPU>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), fvsz, &alpha, 
                                  kp->fv_states().at<GPU>(), kp->fv_states().ld(),
                                  hpsi[ispn].at<GPU>(), hpsi[ispn].ld(), &beta, h.at<GPU>(), h.ld());
                h.copy_to_host();
                hpsi[ispn].deallocate_on_device();
                double tval = t4.stop();
                DUMP("effective zgemm performance: %12.6f GFlops", 
                     8e-9 * parameters_.num_fv_states() * parameters_.num_fv_states() * fvsz / tval);
                #else
                TERMINATE_NO_GPU
                #endif
            }
            else
            {
                STOP();
                /* compute <wf_i | (h * wf_j)> for up-up or dn-dn block */
                //linalg<CPU>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), fvsz, complex_one, 
                //                  kp->fv_states(), hpsi[ispn], complex_zero, h);
            }
            
            for (int i = 0; i < parameters_.num_fv_states(); i++) h.add(i, i, kp->fv_eigen_value(i));
        
            Timer t1("sirius::Band::solve_sv|stdevp");
            std_evp_solver()->solve(parameters_.num_fv_states(), h.at<CPU>(), h.ld(),
                                    &band_energies[ispn * parameters_.num_fv_states()],
                                    kp->sv_eigen_vectors(ispn).at<CPU>(), kp->sv_eigen_vectors(ispn).ld());
        }
    }
    else
    {
        STOP();
    }

    if (parameters_.processing_unit() == GPU && kp->num_ranks() == 1)
    {
        #ifdef __GPU
        kp->fv_states().deallocate_on_device();
        #endif
    }

    kp->set_band_energies(&band_energies[0]);
}

};
