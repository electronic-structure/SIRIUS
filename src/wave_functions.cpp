#include "wave_functions.h"

namespace sirius {

void Wave_functions<false>::swap_forward(int idx0__, int n__, Gvec const& gvec__, Communicator const& comm_col__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::swap_forward");

    /* this is how n wave-functions will be distributed between columns of the MPI grid */
    spl_n_ = splindex<block>(n__, comm_col__.size(), comm_col__.rank());

    /* trivial case */
    if (comm_col__.size() == 1) {
        wf_coeffs_swapped_ = mdarray<double_complex, 2>(&wf_coeffs_(0, idx0__), num_gvec_loc_, n__);
        return;
    } else {
        /* maximum local number of wave-functions */
        int max_nwf_loc = splindex_base<int>::block_size(n__, comm_col__.size());
        /* upper limit for the size of swapped wfs */
        size_t sz = gvec__.gvec_count_fft() * max_nwf_loc;
        /* reallocate buffers if necessary */
        if (wf_coeffs_swapped_buf_.size() < sz) {
            wf_coeffs_swapped_buf_ = mdarray<double_complex, 1>(sz);
            send_recv_buf_ = mdarray<double_complex, 1>(sz, "send_recv_buf_");
        }
        wf_coeffs_swapped_ = mdarray<double_complex, 2>(&wf_coeffs_swapped_buf_[0], gvec__.gvec_count_fft(), max_nwf_loc);
    }
    
    /* local number of columns */
    int n_loc = spl_n_.local_size();
    
    /* send and recieve dimensions */
    block_data_descriptor sd(comm_col__.size()), rd(comm_col__.size());
    for (int j = 0; j < comm_col__.size(); j++)
    {
        sd.counts[j] = spl_n_.local_size(j)                 * gvec__.gvec_fft_slab().counts[comm_col__.rank()];
        rd.counts[j] = spl_n_.local_size(comm_col__.rank()) * gvec__.gvec_fft_slab().counts[j];
    }
    sd.calc_offsets();
    rd.calc_offsets();

    comm_col__.alltoall(&wf_coeffs_(0, idx0__), &sd.counts[0], &sd.offsets[0],
                        &send_recv_buf_[0], &rd.counts[0], &rd.offsets[0]);
                      
    /* reorder recieved blocks */
    #pragma omp parallel for
    for (int i = 0; i < n_loc; i++)
    {
        for (int j = 0; j < comm_col__.size(); j++)
        {
            int offset = gvec__.gvec_fft_slab().offsets[j];
            int count  = gvec__.gvec_fft_slab().counts[j];
            std::memcpy(&wf_coeffs_swapped_(offset, i), &send_recv_buf_[offset * n_loc + count * i], count * sizeof(double_complex));
        }
    }
}

void Wave_functions<false>::swap_backward(int idx0__, int n__, Gvec const& gvec__, Communicator const& comm_col__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::swap_backward");

    if (comm_col__.size() == 1) {
        return;
    }

    /* this is how n wave-functions are distributed between column ranks */
    splindex<block> spl_n(n__, comm_col__.size(), comm_col__.rank());
    /* local number of columns */
    int n_loc = spl_n.local_size();

    /* reorder sending blocks */
    #pragma omp parallel for
    for (int i = 0; i < n_loc; i++) {
        for (int j = 0; j < comm_col__.size(); j++) {
            int offset = gvec__.gvec_fft_slab().offsets[j];
            int count  = gvec__.gvec_fft_slab().counts[j];
            std::memcpy(&send_recv_buf_[offset * n_loc + count * i], &wf_coeffs_swapped_(offset, i), count * sizeof(double_complex));
        }
    }

    /* send and recieve dimensions */
    block_data_descriptor sd(comm_col__.size()), rd(comm_col__.size());
    for (int j = 0; j < comm_col__.size(); j++) {
        sd.counts[j] = spl_n.local_size(comm_col__.rank()) * gvec__.gvec_fft_slab().counts[j];
        rd.counts[j] = spl_n.local_size(j)                 * gvec__.gvec_fft_slab().counts[comm_col__.rank()];
    }
    sd.calc_offsets();
    rd.calc_offsets();

    comm_col__.alltoall(&send_recv_buf_[0], &sd.counts[0], &sd.offsets[0],
                        &wf_coeffs_(0, idx0__), &rd.counts[0], &rd.offsets[0]);
}

template<>
void Wave_functions<false>::inner<double_complex>(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
                                                  mdarray<double_complex, 2>& result__, int irow__, int icol__,
                                                  Communicator const& comm__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::inner");

    assert(num_gvec_loc() == ket__.num_gvec_loc());

    /* single rank, CPU: store result directly in the output matrix */
    if (comm__.size() == 1 && pu_ == CPU)
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

        comm__.allreduce(&inner_prod_buf_[0], 2 * m__ * n__);

        for (int i = 0; i < n__; i++)
            std::memcpy(&result__(irow__, icol__ + i), &inner_prod_buf_[2 * i * m__], m__ * sizeof(double_complex));
    }
}

template<>
void Wave_functions<false>::inner<double>(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
                                          mdarray<double, 2>& result__, int irow__, int icol__, Communicator const& comm__)
{
    PROFILE_WITH_TIMER("sirius::Wave_functions::inner");

    assert(num_gvec_loc() == ket__.num_gvec_loc());

    /* single rank, CPU: store result directly in the output matrix */
    if (comm__.size() == 1 && pu_ == CPU)
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
        double alpha = 2;
        double beta = 0;
        switch (pu_)
        {
            case CPU:
            {
                linalg<CPU>::gemm(1, 0, m__, n__, 2 * num_gvec_loc(),
                                  alpha,
                                  (double*)&wf_coeffs_(0, i0__), 2 * num_gvec_loc(),
                                  (double*)&ket__(0, j0__), 2 * num_gvec_loc(),
                                  beta,
                                  &inner_prod_buf_[0], m__);
                if (comm__.rank() == 0)
                {
                    /* subtract one extra G=0 contribution */
                    linalg<CPU>::ger(m__, n__, -1.0, (double*)&wf_coeffs_(0, i0__), 2 * num_gvec_loc(),
                                    (double*)&ket__(0, j0__), 2 * num_gvec_loc(), &inner_prod_buf_[0], m__); 
                }
                break;
            }
            case GPU:
            {
                #ifdef __GPU
                linalg<GPU>::gemm(1, 0, m__, n__, 2 * num_gvec_loc(),
                                  &alpha,
                                  (double*)wf_coeffs_.at<GPU>(0, i0__), 2 * num_gvec_loc(),
                                  (double*)ket__.wf_coeffs_.at<GPU>(0, j0__), 2 * num_gvec_loc(),
                                  &beta,
                                  inner_prod_buf_.at<GPU>(), m__);
                double alpha1 = -1;
                if (comm__.rank() == 0)
                {
                    /* subtract one extra G=0 contribution */
                    linalg<GPU>::ger(m__, n__, &alpha1, (double*)wf_coeffs_.at<GPU>(0, i0__), 2 * num_gvec_loc(),
                                    (double*)ket__.wf_coeffs_.at<GPU>(0, j0__), 2 * num_gvec_loc(), inner_prod_buf_.at<GPU>(), m__); 
                }
                inner_prod_buf_.copy_to_host(m__ * n__);
                #else
                TERMINATE_NO_GPU
                #endif
                break;
            }
        }

        comm__.allreduce(&inner_prod_buf_[0], m__ * n__);

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
