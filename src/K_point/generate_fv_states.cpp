#include "k_point.h"

namespace sirius {

void K_point::generate_fv_states()
{
    PROFILE();

    Timer t("sirius::K_point::generate_fv_states");
    
    if (parameters_.full_potential())
    {
        if (parameters_.processing_unit() == GPU && num_ranks() == 1)
        {
            #ifdef __GPU
            /* copy eigen-vectors to GPU */
            fv_eigen_vectors_panel_.panel().allocate_on_device();
            fv_eigen_vectors_panel_.panel().copy_to_device();

            /* allocate GPU memory for fv_states */
            fv_states_.allocate_on_device();

            double_complex alpha(1, 0);
            double_complex beta(0, 0);

            int num_atoms_in_block = 2 * Platform::max_num_threads();
            int nblk = unit_cell_.num_atoms() / num_atoms_in_block + std::min(1, unit_cell_.num_atoms() % num_atoms_in_block);
            DUMP("nblk: %i", nblk);

            int max_mt_aw = num_atoms_in_block * unit_cell_.max_mt_aw_basis_size();
            DUMP("max_mt_aw: %i", max_mt_aw);

            mdarray<double_complex, 3> alm_row(nullptr, num_gkvec_row(), max_mt_aw, 2);
            alm_row.allocate(1);
            alm_row.allocate_on_device();
            
            int mt_aw_blk_offset = 0;
            for (int iblk = 0; iblk < nblk; iblk++)
            {
                int num_mt_aw_blk = 0;
                std::vector<int> offsets(num_atoms_in_block);
                for (int ia = iblk * num_atoms_in_block; ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++)
                {
                    auto atom = unit_cell_.atom(ia);
                    auto type = atom->type();
                    offsets[ia - iblk * num_atoms_in_block] = num_mt_aw_blk;
                    num_mt_aw_blk += type->mt_aw_basis_size();
                }

                int s = iblk % 2;
                    
                #pragma omp parallel
                {
                    int tid = Platform::thread_id();
                    for (int ia = iblk * num_atoms_in_block; ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++)
                    {
                        if (ia % Platform::num_threads() == tid)
                        {
                            int ialoc = ia - iblk * num_atoms_in_block;
                            auto atom = unit_cell_.atom(ia);
                            auto type = atom->type();

                            mdarray<double_complex, 2> alm_row_tmp(alm_row.at<CPU>(0, offsets[ialoc], s),
                                                                   alm_row.at<GPU>(0, offsets[ialoc], s),
                                                                   num_gkvec_row(), type->mt_aw_basis_size());

                            alm_coeffs_row()->generate(ia, alm_row_tmp);
                            alm_row_tmp.async_copy_to_device(tid);
                        }
                    }
                    cuda_stream_synchronize(tid);
                }
                cuda_stream_synchronize(Platform::max_num_threads());
                /* gnerate aw expansion coefficients */
                linalg<GPU>::gemm(1, 0, num_mt_aw_blk, parameters_.num_fv_states(), num_gkvec_row(), &alpha,
                                  alm_row.at<GPU>(0, 0, s), alm_row.ld(),
                                  fv_eigen_vectors_panel_.panel().at<GPU>(), fv_eigen_vectors_panel_.panel().ld(),
                                  &beta, fv_states_.at<GPU>(mt_aw_blk_offset, 0), fv_states_.ld(), Platform::max_num_threads());
                mt_aw_blk_offset += num_mt_aw_blk;
            }
            cuda_stream_synchronize(Platform::max_num_threads());
            alm_row.deallocate_on_device();

            mdarray<double_complex, 2> tmp_buf(nullptr, unit_cell_.max_mt_aw_basis_size(), parameters_.num_fv_states());
            tmp_buf.allocate_on_device();

            /* copy aw coefficients starting from bottom */
            for (int ia = unit_cell_.num_atoms() - 1; ia >= 0; ia--)
            {
                int offset_wf = unit_cell_.atom(ia)->offset_wf();
                int offset_aw = unit_cell_.atom(ia)->offset_aw();
                int mt_aw_size = unit_cell_.atom(ia)->mt_aw_basis_size();
                
                /* copy to temporary array */
                cuda_memcpy2D_device_to_device(tmp_buf.at<GPU>(), tmp_buf.ld(),
                                               fv_states_.at<GPU>(offset_aw, 0), fv_states_.ld(),
                                               mt_aw_size, parameters_.num_fv_states(), sizeof(double_complex));

                /* copy to proper place in wave-function array */
                cuda_memcpy2D_device_to_device(fv_states_.at<GPU>(offset_wf, 0), fv_states_.ld(),
                                               tmp_buf.at<GPU>(), tmp_buf.ld(),
                                               mt_aw_size, parameters_.num_fv_states(), sizeof(double_complex));
                
                /* copy block of local orbital coefficients */
                cuda_memcpy2D_device_to_device(fv_states_.at<GPU>(offset_wf + mt_aw_size, 0), fv_states_.ld(),
                                               fv_eigen_vectors_panel_.panel().at<GPU>(num_gkvec_row() + unit_cell_.atom(ia)->offset_lo(), 0),
                                               fv_eigen_vectors_panel_.panel().ld(),
                                               unit_cell_.atom(ia)->mt_lo_basis_size(), parameters_.num_fv_states(), sizeof(double_complex));
            }
            /* copy block of pw coefficients */
            cuda_memcpy2D_device_to_device(fv_states_.at<GPU>(unit_cell_.mt_basis_size(), 0), fv_states_.ld(),
                                           fv_eigen_vectors_panel_.panel().at<GPU>(),  fv_eigen_vectors_panel_.panel().ld(),
                                           num_gkvec_row(), parameters_.num_fv_states(), sizeof(double_complex));

            fv_eigen_vectors_panel_.panel().deallocate_on_device();
            fv_states_.copy_to_host();
            //fv_states_.deallocate_on_device();
            #else
            TERMINATE_NO_GPU
            #endif
        }
        else
        {
            /* slices of first-variational eigen-vectors */
            dmatrix<double_complex> fv_eigen_vectors_slice(gklo_basis_size(), parameters_.num_fv_states(), blacs_grid_slice_, 1, 1);
            assert(fv_eigen_vectors_slice.num_cols_local() == fv_states_slice_.num_cols_local());

            /* change from 2d block cyclic to slice storage */
            linalg<CPU>::gemr2d(gklo_basis_size(), parameters_.num_fv_states(), fv_eigen_vectors_, 0, 0, 
                                fv_eigen_vectors_slice, 0, 0, blacs_grid_.context());
            
            #pragma omp parallel
            {
                mdarray<double_complex, 2> alm(num_gkvec(), unit_cell_.max_mt_aw_basis_size());
                
                #pragma omp for
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
                {
                    int mt_aw_size = unit_cell_.atom(ia)->mt_aw_basis_size();
                    int offset_wf = unit_cell_.atom(ia)->offset_wf();
                    alm_coeffs_->generate(ia, alm);
                    linalg<CPU>::gemm(1, 0, mt_aw_size, fv_eigen_vectors_slice.num_cols_local(), num_gkvec(),
                                      alm.at<CPU>(), alm.ld(), fv_eigen_vectors_slice.at<CPU>(), fv_eigen_vectors_slice.ld(),
                                      &fv_states_slice_(offset_wf, 0), fv_states_slice_.ld());
                    for (int i = 0; i < fv_states_slice_.num_cols_local(); i++)
                    {
                        /* lo block */
                        memcpy(&fv_states_slice_(offset_wf + mt_aw_size, i),
                               &fv_eigen_vectors_slice(num_gkvec() + unit_cell_.atom(ia)->offset_lo(), i),
                               unit_cell_.atom(ia)->mt_lo_basis_size() * sizeof(double_complex));

                    }
                }
                #pragma omp for
                for (int i = 0; i < fv_states_slice_.num_cols_local(); i++)
                {
                    /* G+k block */
                    memcpy(&fv_states_slice_(unit_cell_.mt_basis_size(), i), &fv_eigen_vectors_slice(0, i), 
                           num_gkvec() * sizeof(double_complex));
                }
            }

            /* change from slice storage to 2d block cyclic */
            linalg<CPU>::gemr2d(wf_size(), parameters_.num_fv_states(), fv_states_slice_, 0, 0, 
                                fv_states_, 0, 0, blacs_grid_.context());
        }
    }
}

};
