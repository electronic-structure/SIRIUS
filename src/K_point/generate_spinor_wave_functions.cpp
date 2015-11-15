#include "k_point.h"

namespace sirius {

void K_point::generate_spinor_wave_functions()
{
    PROFILE();

    Timer t("sirius::K_point::generate_spinor_wave_functions");

    int nfv = parameters_.num_fv_states();
    double_complex alpha(1, 0);
    double_complex beta(0, 0);
    
    if (use_second_variation) 
    {
        if (!parameters_.need_sv())
        {
            fv_states_->primary_data_storage_as_matrix() >> spinor_wave_functions_[0]->primary_data_storage_as_matrix();
            return;
        }
 
        /* serial version */
        if (num_ranks() == 1)
        {
            //spinor_wave_functions_.zero();
            if (parameters_.processing_unit() == GPU)
            {
                #ifdef __GPU
                fv_states_.allocate_on_device();
                fv_states_.copy_to_device();
                #endif
            }
 
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                if (parameters_.num_mag_dims() != 3)
                {
                    if (parameters_.processing_unit() == GPU)
                    {
                        #ifdef __GPU
                        sv_eigen_vectors_[ispn].allocate_on_device();
                        sv_eigen_vectors_[ispn].copy_to_device();
                        spinor_wave_functions_[ispn].allocate_on_device();
 
                        linalg<GPU>::gemm(0, 0, wf_size(), nfv, nfv, &alpha, fv_states_.at<GPU>(), fv_states_.ld(), 
                                          sv_eigen_vectors_[ispn].at<GPU>(), sv_eigen_vectors_[ispn].ld(),
                                          &beta, spinor_wave_functions_[ispn].at<GPU>(), spinor_wave_functions_[ispn].ld());
 
                        sv_eigen_vectors_[ispn].deallocate_on_device();
                        spinor_wave_functions_[ispn].copy_to_host();
                        spinor_wave_functions_[ispn].deallocate_on_device();
                        #else
                        TERMINATE_NO_GPU
                        #endif
                    }
                    else
                    {
                        STOP();
                       // /* multiply up block for first half of the bands, dn block for second half of the bands */
                       // linalg<CPU>::gemm(0, 0, wf_size(), nfv, nfv, fv_states_.at<CPU>(), fv_states_.ld(), 
                       //                   sv_eigen_vectors_[ispn].at<CPU>(), sv_eigen_vectors_[ispn].ld(), 
                       //                   spinor_wave_functions_[ispn].at<CPU>(), spinor_wave_functions_[ispn].ld());
                    }
                }
                else
                {
                    STOP();
                    /* multiply up block and then dn block for all bands */
                    //linalg<CPU>::gemm(0, 0, wf_size(), parameters_.num_bands(), nfv, fv_states_.at<CPU>(), fv_states_.ld(), 
                    //                  sv_eigen_vectors_[0].at<CPU>(ispn * nfv, 0), sv_eigen_vectors_[0].ld(), 
                    //                  spinor_wave_functions_[ispn].at<CPU>(), spinor_wave_functions_[ispn].ld());
                }
            }
            if (parameters_.processing_unit() == GPU)
            {
                #ifdef __GPU
                fv_states_.deallocate_on_device();
                #endif
            }
        }
        /* parallel version */
        else
        {
            STOP();

            //int nst = (parameters_.num_mag_dims() == 3) ? parameters_.num_bands() : parameters_.num_fv_states();
            ///* spin component of spinor wave functions */
            //dmatrix<double_complex> spin_component_panel(wf_size(), nst, blacs_grid_, parameters_.cyclic_block_size(),
            //                                             parameters_.cyclic_block_size());
            //
            //for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            //{
            //    if (parameters_.num_mag_dims() != 3)
            //    {
            //        /* multiply up block for first half of the bands, dn block for second half of the bands */
            //        linalg<CPU>::gemm(0, 0, wf_size(), nfv, nfv, complex_one, fv_states_, sv_eigen_vectors_[ispn],
            //                          complex_zero, spin_component_panel);
            //        
            //    }
            //    else
            //    {
            //        /* multiply up block and then dn block for all bands */
            //        linalg<CPU>::gemm(0, 0, wf_size(), parameters_.num_bands(), nfv, complex_one, fv_states_, 0, 0,
            //                          sv_eigen_vectors_[0], ispn * nfv, 0, complex_zero, spin_component_panel, 0, 0);
 
            //    }

            //    /* change to slice storage */
            //    linalg<CPU>::gemr2d(wf_size(), nst, spin_component_panel, 0, 0,
            //                        spinor_wave_functions_[ispn], 0, 0, blacs_grid_.context());
            //}
        }
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
