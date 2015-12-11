#include "wave_functions.h"

namespace sirius {

void Wave_functions<false>::swap_forward(int idx0__, int n__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::swap_forward");

    /* this is how n wave-functions will be distributed between panels */
    spl_n_ = splindex<block>(n__, num_ranks_col_, mpi_grid_.communicator(1 << 0).rank());

    /* trivial case */
    if (comm_.size() == 1)
    {
        wf_coeffs_swapped_ = mdarray<double_complex, 2>(&wf_coeffs_(0, idx0__), num_gvec_loc_, n__);
        return;
    }

    /* local number of columns */
    int n_loc = spl_n_.local_size();

    /* send parts of slab
     * +---+---+--+
     * |   |   |  |  <- irow = 0
     * +---+---+--+
     * |   |   |  |
     * ............
     * ranks in flat and 2D grid are related as: rank = irow * ncol + icol */
    for (int icol = 0; icol < num_ranks_col_; icol++)
    {
        if (spl_n_.local_size(icol))
        {
            int dest_rank = comm_.cart_rank({icol, rank_ / num_ranks_col_});
            comm_.isend(&wf_coeffs_(0, idx0__ + spl_n_.global_offset(icol)),
                        num_gvec_loc_ * spl_n_.local_size(icol),
                        dest_rank, rank_ % num_ranks_col_);
        }
    }
    
    /* receive parts of panel
     *                 n_loc
     *                 +---+  
     *                 |   |
     * gvec_slab_distr +---+
     *                 |   | 
     *                 +---+ */
    if (num_ranks_col_ > 1)
    {
        for (int i = 0; i < num_ranks_col_; i++)
        {
            if (n_loc)
            {
                int src_rank = rank_row_ * num_ranks_col_ + i;
                comm_.recv(&send_recv_buf_[gvec_slab_distr_.offsets[i] * n_loc], gvec_slab_distr_.counts[i] * n_loc, src_rank, i);
            }
        }
        
        /* reorder received blocks to make G-vector index continuous */
        #pragma omp parallel for
        for (int i = 0; i < n_loc; i++)
        {
            for (int j = 0; j < num_ranks_col_; j++)
            {
                std::memcpy(&wf_coeffs_swapped_(gvec_slab_distr_.offsets[j], i),
                            &send_recv_buf_[gvec_slab_distr_.offsets[j] * n_loc + gvec_slab_distr_.counts[j] * i],
                            gvec_slab_distr_.counts[j] * sizeof(double_complex));
            }
        }
    }
    else
    {
        int src_rank = rank_row_ * num_ranks_col_;
        comm_.recv(&wf_coeffs_swapped_(0, 0), gvec_slab_distr_.counts[0] * n_loc, src_rank, 0);
    }
    comm_.barrier();
}

void Wave_functions<false>::swap_backward(int idx0__, int n__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::swap_backward");

    if (comm_.size() == 1) return;
    
    /* this is how n wave-functions are distributed between panels */
    splindex<block> spl_n(n__, num_ranks_col_, mpi_grid_.communicator(1 << 0).rank());
    /* local number of columns */
    int n_loc = spl_n.local_size();

    //==std::vector<MPI_Request> req(num_ranks_col_);
    //==/* post a non-blocking recieve request */
    //==for (int icol = 0; icol < num_ranks_col_; icol++)
    //=={
    //==    int src_rank = comm_.cart_rank({icol, rank_ / num_ranks_col_});
    //==    comm_.irecv(&primary_data_storage_[primary_ld_ * (idx0__ + spl_n.global_offset(icol))],
    //==                num_gvec_loc_ * spl_n.local_size(icol),
    //==                src_rank, rank_ % num_ranks_col_, &req[icol]);
    //==}
    
    if (num_ranks_col_ > 1)
    {
        /* reorder sending blocks */
        #pragma omp parallel for
        for (int i = 0; i < n_loc; i++)
        {
            for (int j = 0; j < num_ranks_col_; j++)
            {
                std::memcpy(&send_recv_buf_[gvec_slab_distr_.offsets[j] * n_loc + gvec_slab_distr_.counts[j] * i],
                            &wf_coeffs_swapped_(gvec_slab_distr_.offsets[j], i),
                            gvec_slab_distr_.counts[j] * sizeof(double_complex));
            }
        }

        for (int i = 0; i < num_ranks_col_; i++)
        {
            if (n_loc)
            {
                int dest_rank = rank_row_ * num_ranks_col_ + i;
                comm_.isend(&send_recv_buf_[gvec_slab_distr_.offsets[i] * n_loc], gvec_slab_distr_.counts[i] * n_loc, dest_rank, i);
            }
        }
    }
    else
    {
        int dest_rank = rank_row_ * num_ranks_col_;
        comm_.isend(&wf_coeffs_swapped_(0, 0), gvec_slab_distr_.counts[0] * n_loc, dest_rank, 0);
    }
    
    for (int icol = 0; icol < num_ranks_col_; icol++)
    {
        int src_rank = comm_.cart_rank({icol, rank_ / num_ranks_col_});
        //double t = -omp_get_wtime();
        if (spl_n.local_size(icol))
        {
            comm_.recv(&wf_coeffs_(0, idx0__ + spl_n.global_offset(icol)),
                       num_gvec_loc_ * spl_n.local_size(icol),
                       src_rank, rank_ % num_ranks_col_);
        }
        //t += omp_get_wtime();
        //DUMP("recieve from %i, %li bytes, %f GB/s",
        //     src_rank, 
        //     num_gvec_loc_ * spl_n.local_size(icol) * sizeof(double_complex),
        //     num_gvec_loc_ * spl_n.local_size(icol) * sizeof(double_complex) / double(1 << 30) / t);
    }
    //==std::vector<MPI_Status> stat(num_ranks_col_);
    //==MPI_Waitall(num_ranks_col_, &req[0], &stat[0]);
    comm_.barrier();
}

void Wave_functions<false>::inner(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
                                  mdarray<double_complex, 2>& result__, int irow__, int icol__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::inner");

    assert(num_gvec_loc() == ket__.num_gvec_loc());

    /* single rank, CPU: store result directly in the output matrix */
    if (comm_.size() == 1 && pu_ == CPU)
    {
        linalg<CPU>::gemm(2, 0, m__, n__, num_gvec_loc(), &wf_coeffs_(0, i0__), num_gvec_loc(),
                          &ket__(0, j0__), num_gvec_loc(), &result__(irow__, icol__), result__.ld());
    }
    else
    {
        /* reallocate buffer if necessary */
        if (static_cast<size_t>(m__ * n__) > inner_prod_buf_.size())
        {
            inner_prod_buf_ = mdarray<double_complex, 1>(m__ * n__);
            #ifdef __GPU
            if (pu_ == GPU) inner_prod_buf_.allocate_on_device();
            #endif
        }
        switch (pu_)
        {
            case CPU:
            {
                linalg<CPU>::gemm(2, 0, m__, n__, num_gvec_loc(), &wf_coeffs_(0, i0__), num_gvec_loc(),
                                  &ket__(0, j0__), num_gvec_loc(), &inner_prod_buf_[0], m__);
                break;
            }
            case GPU:
            {
                #ifdef __GPU
                linalg<GPU>::gemm(2, 0, m__, n__, num_gvec_loc(), wf_coeffs_.at<GPU>(0, i0__), num_gvec_loc(),
                                  ket__.wf_coeffs_.at<GPU>(0, j0__), num_gvec_loc(), inner_prod_buf_.at<GPU>(), m__);
                inner_prod_buf_.copy_to_host(m__ * n__);
                #else
                TERMINATE_NO_GPU
                #endif
                break;
            }
        }

        if (comm_.size() > 1) comm_.allreduce(&inner_prod_buf_[0], m__ * n__);

        for (int i = 0; i < n__; i++)
            std::memcpy(&result__(irow__, icol__ + i), &inner_prod_buf_[i * m__], m__ * sizeof(double_complex));
    }
}

};
