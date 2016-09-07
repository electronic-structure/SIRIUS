#include "k_point.h"

namespace sirius {

void K_point::generate_spinor_wave_functions()
{
    PROFILE_WITH_TIMER("sirius::K_point::generate_spinor_wave_functions");

    double_complex alpha(1, 0);
    double_complex beta(0, 0);

    if (use_second_variation) 
    {
        if (!ctx_.need_sv())
        {
            /* copy eigen-states and exit */
            if (ctx_.full_potential()) {
                fv_states<true>().prime() >> spinor_wave_functions<true>(0).prime();
            } else {
                fv_states<false>().prime() >> spinor_wave_functions<false>(0).prime();
            }
            return;
        }

        int nfv = ctx_.num_fv_states();
        int nbnd = (ctx_.num_mag_dims() == 3) ? ctx_.num_bands() : nfv;

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int s, o;

            if (ctx_.num_mag_dims() == 3) {
                /* in case of non-collinear magnetism sv_eigen_vectors is a single 2Nx2N matrix */
                s = 0;
                o = ispn * nfv; // offset for spin up is 0, for spin dn is nfv
            } else { 
                /* sv_eigen_vectors is composed of two NxN matrices */
                s = ispn;
                o = 0;
            }
            /* multiply consecutively up and dn blocks */
            linalg<CPU>::gemm(0, 0, wf_size(), nbnd, nfv, double_complex(1, 0), fv_states<true>().prime(), 0, 0,
                              sv_eigen_vectors_[s], o, 0, double_complex(0, 0), spinor_wave_functions<true>(ispn).prime(), 0, 0);
        }
 
        ///* serial version */
        //if (num_ranks() == 1)
        //{
        //    //spinor_wave_functions_.zero();
        //    if (ctx_.processing_unit() == GPU)
        //    {
        //        #ifdef __GPU
        //        //fv_states_.allocate_on_device();
        //        //fv_states_.copy_to_device();
        //        #endif
        //    }
 
        //    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
        //    {
        //        if (ctx_.num_mag_dims() != 3)
        //        {
        //            if (ctx_.processing_unit() == GPU)
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
        //    if (ctx_.processing_unit() == GPU)
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
     //==     for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
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
     //== for (int i = 0; i < ctx_.spl_spinor_wf_col().local_size(); i++)
     //==     Platform::allreduce(&spinor_wave_functions_(0, 0, i), wfld, ctx_.mpi_grid().communicator(1 << _dim_row_));
     //== 
}

};
