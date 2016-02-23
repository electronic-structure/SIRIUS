#include "wave_functions.h"

namespace sirius {

void Wave_functions<false>::swap_forward(int idx0__, int n__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::swap_forward");

    /* this is how n wave-functions will be distributed between columns */
    spl_n_ = splindex<block>(n__, comm_col_.size(), comm_col_.rank());

    /* trivial case */
    if (comm_col_.size() == 1)
    {
        wf_coeffs_swapped_ = mdarray<double_complex, 2>(&wf_coeffs_(0, idx0__), num_gvec_loc_, n__);
        return;
    }

    /* local number of columns */
    int n_loc = spl_n_.local_size();

    block_data_descriptor sd(comm_col_.size());
    block_data_descriptor rd(comm_col_.size());

    for (int j = 0; j < comm_col_.size(); j++)
    {
        sd.counts[j] = spl_n_.local_size(j) * gvec_slab_pile_.counts[comm_col_.rank()];
        rd.counts[j] = spl_n_.local_size(comm_col_.rank()) * gvec_slab_pile_.counts[j];
    }

    sd.calc_offsets();
    rd.calc_offsets();

    comm_col_.alltoall(&wf_coeffs_(0, idx0__), &sd.counts[0], &sd.offsets[0],
                       &send_recv_buf_[0], &rd.counts[0], &rd.offsets[0]);
                      
    /* reorder recieved blocks */
    #pragma omp parallel for
    for (int i = 0; i < n_loc; i++)
    {
        for (int j = 0; j < comm_col_.size(); j++)
        {
            int offset = gvec_slab_pile_.offsets[j];
            int count = gvec_slab_pile_.counts[j];
            std::memcpy(&wf_coeffs_swapped_(offset, i), &send_recv_buf_[offset * n_loc + count * i], count * sizeof(double_complex));
        }
    }
}

void Wave_functions<false>::swap_backward(int idx0__, int n__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::swap_backward");

    if (comm_col_.size() == 1) return;

    /* this is how n wave-functions are distributed between column ranks */
    splindex<block> spl_n(n__, comm_col_.size(), comm_col_.rank());
    /* local number of columns */
    int n_loc = spl_n.local_size();

    /* reorder sending blocks */
    #pragma omp parallel for
    for (int i = 0; i < n_loc; i++)
    {
        for (int j = 0; j < comm_col_.size(); j++)
        {
            int offset = gvec_slab_pile_.offsets[j];
            int count = gvec_slab_pile_.counts[j];
            std::memcpy(&send_recv_buf_[offset * n_loc + count * i], &wf_coeffs_swapped_(offset, i), count * sizeof(double_complex));
        }
    }

    block_data_descriptor sd(comm_col_.size());
    block_data_descriptor rd(comm_col_.size());

    for (int j = 0; j < comm_col_.size(); j++)
    {
        sd.counts[j] = spl_n.local_size(comm_col_.rank()) * gvec_slab_pile_.counts[j];
        rd.counts[j] = spl_n.local_size(j) * gvec_slab_pile_.counts[comm_col_.rank()];
    }

    sd.calc_offsets();
    rd.calc_offsets();

    comm_col_.alltoall(&send_recv_buf_[0], &sd.counts[0], &sd.offsets[0],
                       &wf_coeffs_(0, idx0__), &rd.counts[0], &rd.offsets[0]);
}

template<>
void Wave_functions<false>::inner<double_complex>(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
                                                  mdarray<double_complex, 2>& result__, int irow__, int icol__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::inner");

    assert(num_gvec_loc() == ket__.num_gvec_loc());

    /* single rank, CPU: store result directly in the output matrix */
    if (mpi_grid_.size() == 1 && pu_ == CPU)
    {
        linalg<CPU>::gemm(2, 0, m__, n__, num_gvec_loc(), &wf_coeffs_(0, i0__), num_gvec_loc(),
                          &ket__(0, j0__), num_gvec_loc(), &result__(irow__, icol__), result__.ld());
    }
    else
    {
        /* reallocate buffer if necessary */
        if (static_cast<size_t>(2 * m__ * n__) > inner_prod_buf_.size())
        {
            inner_prod_buf_ = mdarray<double, 1>(2 * m__ * n__);
            #ifdef __GPU
            if (pu_ == GPU) inner_prod_buf_.allocate_on_device();
            #endif
        }
        switch (pu_)
        {
            case CPU:
            {
                linalg<CPU>::gemm(2, 0, m__, n__, num_gvec_loc(), &wf_coeffs_(0, i0__), num_gvec_loc(),
                                  &ket__(0, j0__), num_gvec_loc(), (double_complex*)&inner_prod_buf_[0], m__);
                break;
            }
            case GPU:
            {
                #ifdef __GPU
                linalg<GPU>::gemm(2, 0, m__, n__, num_gvec_loc(), wf_coeffs_.at<GPU>(0, i0__), num_gvec_loc(),
                                  ket__.wf_coeffs_.at<GPU>(0, j0__), num_gvec_loc(), (double_complex*)inner_prod_buf_.at<GPU>(), m__);
                inner_prod_buf_.copy_to_host(2 * m__ * n__);
                #else
                TERMINATE_NO_GPU
                #endif
                break;
            }
        }

        if (mpi_grid_.size() > 1) mpi_grid_.communicator().allreduce(&inner_prod_buf_[0], 2 * m__ * n__);

        for (int i = 0; i < n__; i++)
            std::memcpy(&result__(irow__, icol__ + i), &inner_prod_buf_[2 * i * m__], m__ * sizeof(double_complex));
    }
}

template<>
void Wave_functions<false>::inner<double>(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
                                          mdarray<double, 2>& result__, int irow__, int icol__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::inner");

    assert(num_gvec_loc() == ket__.num_gvec_loc());

    /* single rank, CPU: store result directly in the output matrix */
    if (mpi_grid_.size() == 1 && pu_ == CPU)
    {
        linalg<CPU>::gemm(2, 0, m__, n__, 2 * num_gvec_loc(), (double*)&wf_coeffs_(0, i0__), 2 * num_gvec_loc(),
                          (double*)&ket__(0, j0__), 2 * num_gvec_loc(), &result__(irow__, icol__), result__.ld());
        
        for (int j = 0; j < n__; j++)
        {
            for (int i = 0; i < m__; i++)
            {
                result__(irow__ + i, icol__ + j) = 2 * result__(irow__ + i, icol__ + j) -
                                                   wf_coeffs_(0, i0__ + i).real() * ket__(0, j0__ + j).real();
            }
        }
    }
    else
    {
        /* reallocate buffer if necessary */
        if (static_cast<size_t>(m__ * n__) > inner_prod_buf_.size())
        {
            inner_prod_buf_ = mdarray<double, 1>(m__ * n__);
            #ifdef __GPU
            if (pu_ == GPU) inner_prod_buf_.allocate_on_device();
            #endif
        }
        switch (pu_)
        {
            case CPU:
            {
                linalg<CPU>::gemm(2, 0, m__, n__, 2 * num_gvec_loc(), (double*)&wf_coeffs_(0, i0__), 2 * num_gvec_loc(),
                                  (double*)&ket__(0, j0__), 2 * num_gvec_loc(), &inner_prod_buf_[0], m__);
                break;
            }
            case GPU:
            {
                #ifdef __GPU
                linalg<GPU>::gemm(2, 0, m__, n__, 2 * num_gvec_loc(), (double*)wf_coeffs_.at<GPU>(0, i0__), 2 * num_gvec_loc(),
                                  (double*)ket__.wf_coeffs_.at<GPU>(0, j0__), 2 * num_gvec_loc(), inner_prod_buf_.at<GPU>(), m__);
                inner_prod_buf_.copy_to_host(m__ * n__);
                #else
                TERMINATE_NO_GPU
                #endif
                break;
            }
        }
        if (mpi_grid_.communicator().rank() == 0)
        {
            for (int j = 0; j < n__; j++)
            {
                for (int i = 0; i < m__; i++)
                {
                    inner_prod_buf_[i + j * m__] = 2 * inner_prod_buf_[i + j * m__]  -
                                                   wf_coeffs_(0, i0__ + i).real() * ket__(0, j0__ + j).real();
                }
            }
        }
        else
        {
            for (int j = 0; j < n__; j++)
            {
                for (int i = 0; i < m__; i++) inner_prod_buf_[i + j * m__] *= 2;
            }
        }

        if (mpi_grid_.size() > 1) mpi_grid_.communicator().allreduce(&inner_prod_buf_[0], m__ * n__);

        for (int i = 0; i < n__; i++)
            std::memcpy(&result__(irow__, icol__ + i), &inner_prod_buf_[i * m__], m__ * sizeof(double));
    }
}

template<>
void Wave_functions<false>::transform_from<double_complex>(Wave_functions& wf__, int nwf__, matrix<double_complex>& mtrx__, int n__)
{
    assert(num_gvec_loc() == wf__.num_gvec_loc());

    if (pu_ == CPU)
    {
        linalg<CPU>::gemm(0, 0, num_gvec_loc(), n__, nwf__, &wf__(0, 0), num_gvec_loc(),
                          &mtrx__(0, 0), mtrx__.ld(), &wf_coeffs_(0, 0), num_gvec_loc());
    }
    #ifdef __GPU
    if (pu_ == GPU)
    {
        linalg<GPU>::gemm(0, 0, num_gvec_loc(), n__, nwf__, wf__.coeffs().at<GPU>(), num_gvec_loc(),
                          mtrx__.at<GPU>(), mtrx__.ld(), wf_coeffs_.at<GPU>(), num_gvec_loc());
    }
    #endif
}

template<>
void Wave_functions<false>::transform_from<double>(Wave_functions& wf__, int nwf__, matrix<double>& mtrx__, int n__)
{
    assert(num_gvec_loc() == wf__.num_gvec_loc());

    if (pu_ == CPU)
    {
        linalg<CPU>::gemm(0, 0, 2 * num_gvec_loc(), n__, nwf__, (double*)&wf__(0, 0), 2 * num_gvec_loc(),
                          &mtrx__(0, 0), mtrx__.ld(), (double*)&wf_coeffs_(0, 0), 2 * num_gvec_loc());
    }
    #ifdef __GPU
    if (pu_ == GPU)
    {
        linalg<GPU>::gemm(0, 0, 2 * num_gvec_loc(), n__, nwf__, (double*)wf__.coeffs().at<GPU>(), 2 * num_gvec_loc(),
                          mtrx__.at<GPU>(), mtrx__.ld(), (double*)wf_coeffs_.at<GPU>(), 2 * num_gvec_loc());
    }
    #endif
}

};
