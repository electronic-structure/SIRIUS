#include "k_point.h"

namespace sirius {

void K_point::generate_spinor_wave_functions()
{
    PROFILE_WITH_TIMER("sirius::K_point::generate_spinor_wave_functions");

    double_complex alpha(1, 0);
    double_complex beta(0, 0);

    if (use_second_variation) 
    {
        if (!parameters_.need_sv())
        {
            /* copy eigen-states and exit */
            if (parameters_.full_potential())
            {
                fv_states<true>().coeffs().panel() >> spinor_wave_functions<true>(0).coeffs().panel();
            }
            else
            {
                fv_states<false>().coeffs() >> spinor_wave_functions<false>(0).coeffs();
            }
            return;
        }

        int nfv = parameters_.num_fv_states();
        int nbnd = (parameters_.num_mag_dims() == 3) ? parameters_.num_bands() : nfv;

        /* serial GPU version */
        if (num_ranks() == 1 && parameters_.processing_unit() == GPU)
        {
            STOP();
        }
        else
        {
            if (parameters_.full_potential())
            {
                for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                {
                    int s, o;
                    if (parameters_.num_mag_dims() == 3) // in case of non-collinear magnetism sv_eigen_vectors_ is
                    {                                    // a single 2Nx2N matrix
                        s = 0;
                        o = ispn * nfv; // offset for spin up is 0, for spin dn is nfv
                    }
                    else // sv_eigen_vectors_ is composed of two NxN matrices
                    {
                        s = ispn;
                        o = 0;
                    }
                    /* multiply consecutively up and dn blocks */
                    linalg<CPU>::gemm(0, 0, wf_size(), nbnd, nfv, double_complex(1, 0), fv_states<true>().coeffs(), 0, 0,
                                      sv_eigen_vectors_[s], o, 0, double_complex(0, 0), spinor_wave_functions<true>(ispn).coeffs(), 0, 0);
                }
            }
            else
            {
                matrix<double_complex> evec(nfv, nfv);
                for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                {
                    evec.zero();
                    for (int i = 0; i < sv_eigen_vectors_[ispn].num_cols_local(); i++)
                    {
                        for (int j = 0; j < sv_eigen_vectors_[ispn].num_rows_local(); j++)
                        {
                            evec(sv_eigen_vectors_[ispn].irow(j), sv_eigen_vectors_[ispn].icol(i)) = sv_eigen_vectors_[ispn](j, i);
                        }
                    }
                    comm().allreduce(evec.at<CPU>(), nfv * nfv);
                    spinor_wave_functions<false>(ispn).transform_from(fv_states<false>(), parameters_.num_fv_states(), evec, parameters_.num_fv_states());
                }
            }
        }
 
        ///* serial version */
        //if (num_ranks() == 1)
        //{
        //    //spinor_wave_functions_.zero();
        //    if (parameters_.processing_unit() == GPU)
        //    {
        //        #ifdef __GPU
        //        //fv_states_.allocate_on_device();
        //        //fv_states_.copy_to_device();
        //        #endif
        //    }
 
        //    for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
        //    {
        //        if (parameters_.num_mag_dims() != 3)
        //        {
        //            if (parameters_.processing_unit() == GPU)
        //            {
        //                #ifdef __GPU
        //                STOP();
        //                //sv_eigen_vectors_[ispn].allocate_on_device();
        //                //sv_eigen_vectors_[ispn].copy_to_device();
        //                //spinor_wave_functions_[ispn].allocate_on_device();
 
        //                //linalg<GPU>::gemm(0, 0, wf_size(), nfv, nfv, &alpha, fv_states_.at<GPU>(), fv_states_.ld(), 
        //                //                  sv_eigen_vectors_[ispn].at<GPU>(), sv_eigen_vectors_[ispn].ld(),
        //                //                  &beta, spinor_wave_functions_[ispn].at<GPU>(), spinor_wave_functions_[ispn].ld());
 
        //                //sv_eigen_vectors_[ispn].deallocate_on_device();
        //                //spinor_wave_functions_[ispn].copy_to_host();
        //                //spinor_wave_functions_[ispn].deallocate_on_device();
        //                #else
        //                TERMINATE_NO_GPU
        //                #endif
        //            }
        //        }
        //    }
        //    if (parameters_.processing_unit() == GPU)
        //    {
        //        #ifdef __GPU
        //        //fv_states_.deallocate_on_device();
        //        #endif
        //    }
        //}
    }
    else
    {
        STOP();
     //==     mdarray<double_complex, 2> alm(num_gkvec_row(), unit_cell_.max_mt_aw_basis_size());
 
     //==     /** \todo generalize for non-collinear case */
     //==     spinor_wave_functions_.zero();
     //==     for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
     //==     {
     //==         for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
     //==         {
     //==             Atom* atom = unit_cell_.atom(ia);
     //==             Atom_type* type = atom->type();
     //==             
     //==             /** \todo generate unconjugated coefficients for better readability */
     //==             generate_matching_coefficients<true>(num_gkvec_row(), ia, alm);
 
     //==             blas<CPU>::gemm(2, 0, type->mt_aw_basis_size(), ncol, num_gkvec_row(), &alm(0, 0), alm.ld(), 
     //==                             &fd_eigen_vectors_(0, ispn * ncol), fd_eigen_vectors_.ld(), 
     //==                             &spinor_wave_functions_(atom->offset_wf(), ispn, ispn * ncol), wfld); 
     //==         }
 
     //==         for (int j = 0; j < ncol; j++)
     //==         {
     //==             copy_lo_blocks(&fd_eigen_vectors_(0, j + ispn * ncol), &spinor_wave_functions_(0, ispn, j + ispn * ncol));
 
     //==             copy_pw_block(&fd_eigen_vectors_(0, j + ispn * ncol), &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j + ispn * ncol));
     //==         }
     //==     }
     //==     /** \todo how to distribute states in case of full diagonalziation. num_fv_states will probably be reused. 
     //==               maybe the 'fv' should be renamed. */
     }
     //== 
     //== for (int i = 0; i < parameters_.spl_spinor_wf_col().local_size(); i++)
     //==     Platform::allreduce(&spinor_wave_functions_(0, 0, i), wfld, parameters_.mpi_grid().communicator(1 << _dim_row_));
     //== 
}

};
