// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file wf_inner.hpp
 *   
 *  \brief Contains implementation of inner product for wave-functions.
 */

/// Inner product between wave-functions.
/** This function computes the inner product using a moving window scheme plus allreduce.
 *  The input wave-functions data must be previously allocated on the GPU.
 *  The result is always returned in the CPU pointer. In case of a single MPI rank the result is also returned in the
 *  GPU pointer.
 *
 *  The following \f$ m \times n \f$ sub-matrix is computed:
 *  \f[
 *    S_{irow0+i,jcol0+j} = \beta S_{irow0+i,jcol0+j} + \langle \phi_{i0 + i} | \tilde \phi_{j0 + j} \rangle
 *  \f]
 *
 *  \param [in] bra   "bra" wave-functions \f$ \phi \f$.
 *  \param [in] i0    index of the first "bra" wave-function.
 *  \param [in] m     number of "bra" wave-functions.
 *  \param [in] ket   "ket" wave-functions \f$ \tilde \phi \f$.
 *  \param [in] j0    index of the first "ket" wave-function.
 *  \param [in] n     number of "ket" wave-functions.
 *  \param [in] beta  \f$ \beta \f$ parameter.
 *  \param [in,out]   result inner product matrix \f$ S \f$.
 *  \param [in] irow0 first row (in the global matrix) of the inner product sub-matrix.
 *  \param [in] jcol0 first column (in the global matix) of the inner product sub-matrix.
 */
template <typename T>
inline void inner(device_t        pu__,
                  int             ispn__,
                  Wave_functions& bra__,
                  int             i0__,
                  int             m__,
                  Wave_functions& ket__,
                  int             j0__,
                  int             n__,
                  dmatrix<T>&     result__,
                  int             irow0__,
                  int             jcol0__)
{
    PROFILE("sddk::wave_functions::inner");

    static_assert(std::is_same<T, double>::value || std::is_same<T, double_complex>::value, "wrong type");
    
    //assert(&bra__.comm() == &ket__.comm());
    //assert(bra__.pw_coeffs().num_rows_loc() == ket__.pw_coeffs().num_rows_loc());
    //if (bra__.has_mt()) {
    //    assert(bra__.mt_coeffs().num_rows_loc() == ket__.mt_coeffs().num_rows_loc());
    //}

    auto& comm = bra__.comm();
    
    const char* sddk_pp_raw = std::getenv("SDDK_PRINT_PERFORMANCE");
    int sddk_pp = (sddk_pp_raw == NULL) ? 0 : std::atoi(sddk_pp_raw);

    const char* sddk_bs_raw = std::getenv("SDDK_BLOCK_SIZE");
    int sddk_block_size = (sddk_bs_raw == NULL) ? sddk_default_block_size : std::atoi(sddk_bs_raw);

    double ngop{0};
    if (std::is_same<T, double>::value) {
        ngop = 2e-9;
    }
    if (std::is_same<T, double_complex>::value) {
        ngop = 8e-9;
    }

    if (sddk_pp) {
        comm.barrier();
    }
    double time = -omp_get_wtime();

    //result__.zero(i0__, j0__, m__, n__);

    T alpha = (std::is_same<T, double_complex>::value) ? 1 : 2;
    T beta = 0;

    auto local_inner = [&](int i0__,
                           int m__,
                           int j0__,
                           int n__,
                           T*  buf__,
                           int ld__,
                           int stream_id){
        int s0{0};
        int s1{1};
        if (ispn__ != 2) {
            s0 = s1 = ispn__;
        }
        beta = 0;
        for (int s = s0; s <= s1; s++) {
            /* wave-functions are complex and inner product is complex */
            if (std::is_same<T, double_complex>::value) {
                switch (pu__) {
                    case CPU: {
                        linalg<CPU>::gemm(2, 0, m__, n__, bra__.pw_coeffs(s).num_rows_loc(),
                                          *reinterpret_cast<double_complex*>(&alpha),
                                          bra__.pw_coeffs(s).prime().at<CPU>(0, i0__), bra__.pw_coeffs(s).prime().ld(),
                                          ket__.pw_coeffs(s).prime().at<CPU>(0, j0__), ket__.pw_coeffs(s).prime().ld(),
                                          *reinterpret_cast<double_complex*>(&beta),
                                          reinterpret_cast<double_complex*>(buf__), ld__);
                        if (bra__.has_mt()) {
                            linalg<CPU>::gemm(2, 0, m__, n__, bra__.mt_coeffs(s).num_rows_loc(),
                                              *reinterpret_cast<double_complex*>(&alpha),
                                              bra__.mt_coeffs(s).prime().at<CPU>(0, i0__), bra__.mt_coeffs(s).prime().ld(),
                                              ket__.mt_coeffs(s).prime().at<CPU>(0, j0__), ket__.mt_coeffs(s).prime().ld(),
                                              linalg_const<double_complex>::one(),
                                              reinterpret_cast<double_complex*>(buf__), ld__);
                        }
                        break;
                    }
                    case GPU: {
                        #ifdef __GPU
                        linalg<GPU>::gemm(2, 0, m__, n__, bra__.pw_coeffs(s).num_rows_loc(),
                                          reinterpret_cast<double_complex*>(&alpha),
                                          bra__.pw_coeffs(s).prime().at<GPU>(0, i0__), bra__.pw_coeffs(s).prime().ld(),
                                          ket__.pw_coeffs(s).prime().at<GPU>(0, j0__), ket__.pw_coeffs(s).prime().ld(),
                                          reinterpret_cast<double_complex*>(&beta),
                                          reinterpret_cast<double_complex*>(buf__), ld__,
                                          stream_id);
                        if (bra__.has_mt()) {
                            linalg<GPU>::gemm(2, 0, m__, n__, bra__.mt_coeffs(s).num_rows_loc(),
                                              reinterpret_cast<double_complex*>(&alpha),
                                              bra__.mt_coeffs(s).prime().at<GPU>(0, i0__), bra__.mt_coeffs(s).prime().ld(),
                                              ket__.mt_coeffs(s).prime().at<GPU>(0, j0__), ket__.mt_coeffs(s).prime().ld(),
                                              &linalg_const<double_complex>::one(),
                                              reinterpret_cast<double_complex*>(buf__), ld__,
                                              stream_id);
                        }
                        #endif
                        break;
                    }
                }
            }
            /* wave-functions are real and inner product is also real */
            if (std::is_same<T, double>::value) {
                if (bra__.has_mt()) {
                    TERMINATE("not implemented");
                }
                switch (pu__) {
                    case CPU: {
                        linalg<CPU>::gemm(2, 0, m__, n__, 2 * bra__.pw_coeffs(s).num_rows_loc(),
                                          *reinterpret_cast<double*>(&alpha),
                                          reinterpret_cast<double*>(bra__.pw_coeffs(s).prime().at<CPU>(0, i0__)), 2 * bra__.pw_coeffs(s).prime().ld(),
                                          reinterpret_cast<double*>(ket__.pw_coeffs(s).prime().at<CPU>(0, j0__)), 2 * ket__.pw_coeffs(s).prime().ld(),
                                          *reinterpret_cast<double*>(&beta),
                                          reinterpret_cast<double*>(buf__), ld__);
                        /* subtract one extra G=0 contribution */
                        if (comm.rank() == 0) {
                            linalg<CPU>::ger(m__, n__, -1.0,
                                             reinterpret_cast<double*>(bra__.pw_coeffs(s).prime().at<CPU>(0, i0__)), 2 * bra__.pw_coeffs(s).prime().ld(),
                                             reinterpret_cast<double*>(ket__.pw_coeffs(s).prime().at<CPU>(0, j0__)), 2 * ket__.pw_coeffs(s).prime().ld(),
                                             reinterpret_cast<double*>(buf__), ld__); 

                        }
                        break;
                    }
                    case GPU: {
                        #ifdef __GPU
                        linalg<GPU>::gemm(2, 0, m__, n__, 2 * bra__.pw_coeffs(s).num_rows_loc(),
                                          reinterpret_cast<double*>(&alpha),
                                          reinterpret_cast<double*>(bra__.pw_coeffs(s).prime().at<GPU>(0, i0__)), 2 * bra__.pw_coeffs(s).prime().ld(),
                                          reinterpret_cast<double*>(ket__.pw_coeffs(s).prime().at<GPU>(0, j0__)), 2 * ket__.pw_coeffs(s).prime().ld(),
                                          reinterpret_cast<double*>(&beta),
                                          reinterpret_cast<double*>(buf__), ld__,
                                          stream_id);
                        /* subtract one extra G=0 contribution */
                        if (comm.rank() == 0) {
                            linalg<GPU>::ger(m__, n__, &linalg_const<double>::m_one(),
                                             reinterpret_cast<double*>(bra__.pw_coeffs(s).prime().at<GPU>(0, i0__)), 2 * bra__.pw_coeffs(s).prime().ld(),
                                             reinterpret_cast<double*>(ket__.pw_coeffs(s).prime().at<GPU>(0, j0__)), 2 * ket__.pw_coeffs(s).prime().ld(),
                                             reinterpret_cast<double*>(buf__), ld__,
                                             stream_id);
                        }
                        #endif
                        break;
                    }
                }
            }
            beta = 1;
        }
    };

    if (comm.size() == 1) {
        T* buf = (pu__ == CPU) ? result__.template at<CPU>(irow0__, jcol0__) : result__.template at<GPU>(irow0__, jcol0__);
        local_inner(i0__, m__, j0__, n__, buf, result__.ld(), -1);
        #ifdef __GPU
        if (pu__ == GPU) {
            acc::copyout(result__.template at<CPU>(irow0__, jcol0__), result__.ld(),
                         result__.template at<GPU>(irow0__, jcol0__), result__.ld(),
                         m__, n__);
        }
        #endif
        if (sddk_pp) {
            time += omp_get_wtime();
            int k = bra__.gkvec().num_gvec() + bra__.num_mt_coeffs();
            printf("inner() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, time=%f (sec)]\n", ngop * m__ * n__ * k / time, m__, n__, k, time);
        }
        return;
    } else if (result__.comm().size() == 1) {
        mdarray<T, 2> tmp(m__, n__);
        if (pu__ == GPU) {
            tmp.allocate(memory_t::device);
        }
        T* buf = (pu__ == CPU) ? tmp.template at<CPU>(0, 0) : tmp.template at<GPU>(0, 0);
        local_inner(i0__, m__, j0__, n__, buf, m__, -1);
        if (pu__ == GPU) {
            tmp.template copy<memory_t::device, memory_t::host>();
        }
        comm.allreduce(&tmp[0], static_cast<int>(tmp.size()));
        for (int j = 0; j < n__; j++) {
            for (int i = 0; i < m__; i++) {
                result__(irow0__ + i, jcol0__ + j) = tmp(i, j);
            }
        }
        #ifdef __GPU
        if (pu__ == GPU) {
            acc::copyin(result__.template at<GPU>(irow0__, jcol0__), result__.ld(),
                        result__.template at<CPU>(irow0__, jcol0__), result__.ld(),
                        m__, n__);
        }
        #endif
        if (sddk_pp) {
            time += omp_get_wtime();
            int k = bra__.gkvec().num_gvec() + bra__.num_mt_coeffs();
            if (comm.rank() == 0) {
                printf("inner() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, time=%f (sec)]\n", ngop * m__ * n__ * k / time / comm.size(), m__, n__, k, time);
            }
        }
        return;
    }
    
    const int BS = sddk_block_size;

    const int num_streams{4};

    mdarray<T, 2> c_tmp(BS * BS, num_streams, memory_t::host_pinned, "inner::c_tmp");
    if (pu__ == GPU) {
        c_tmp.allocate(memory_t::device);
    }

    /* compute the number of movements of the windows needed to cover the whole matrix size.
     * If m__  is not divided by BS, you need to cover the remaining border; the same for n__
     *
     * +-------------+--------------+------|
     * | <-- BS -->  |              |      |
     * |             |              |      |
     * |             |              |      |
     * |             |              |      |
     * |-------------+--------------+------|
     * | ^           |              |      |
     * | |           |              |      |
     * | | BS        |              |      |
     * | V           |              |      |
     * |-------------+--------------+------|
     * |             |              |      |
     * |-------------+--------------+------|
     *
     */
    /* number of blocks to cover rows of the output matrix */
    int nbr = m__ / BS + std::min(1, m__ % BS);
    /* number of blocks to cover columns of the output matrix */
    int nbc = n__ / BS + std::min(1, n__ % BS);
    
    /* A double buffer method is used in case of CPU */    
    std::array<MPI_Request, 2> req = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    std::array<std::array<int, 4>, 2> dims;
    
    if (pu__ == GPU) {
        #ifdef __GPU
        /* state of the buffers:
         * state = 0: buffer is free
         * state = 1: buffer stores result of local zgemm */
        std::array<int, num_streams> buf_state;
        buf_state.fill(0);
        
        /* The real computation here is done by GPUs.
         * This is to leave the interaction with the GPUs
         * to 1 OMP thread and the MPI compunication
         * and data reshifling to the remaining threads.
         * As a first step, a nested OMP region is needed
         * and at least 2 threads must be available. */
        omp_set_nested(1);
        int nt = omp_get_max_threads();
        if (nt < 2) {
            TERMINATE("minimum two threads are required");
        }

        #pragma omp parallel num_threads(2) shared(buf_state)
        {
            if (omp_get_thread_num() == 0) {
                /* thread 0 spawns as many threads as possible */
                omp_set_num_threads(nt - 1);
            }

            /* this rotates the buffers and the CUDA stream numbers in a round robin way */
            int s{0};
            /* loop over BS sized windows: columns */
            for (int ibc = 0; ibc < nbc; ibc++) {
                /* first column (global index) of the block ibc */
                int j0 = ibc * BS;
                /* actual number of cloumns in the block: either the size of the block or 
                 * what remains of the border */
                int ncol = std::min(n__, (ibc + 1) * BS) - j0;

                /* loop over BS sized windows: rows */
                for (int ibr = 0; ibr < nbr; ibr++) {
                    /* first row (global index) of the block ibr */
                    int i0 = ibr * BS;
                    /* actual number of rows in the block */
                    int nrow = std::min(m__, (ibr + 1) * BS) - i0;

                    /* this thread will call cudaZgemm */
                    if (omp_get_thread_num() == 1) {
                        int state{1};
                        /* wait for the release of the buffer */
                        while (state) {
                            #pragma omp atomic read
                            state = buf_state[s % num_streams];
                        }
                        /* enqueue the gemm kernel */
                        local_inner(i0__ + i0, nrow, j0__ + j0, ncol, c_tmp.template at<GPU>(0, s % num_streams), nrow, s % num_streams);
                        /* enqueue a copyout operation */
                        acc::copyout(c_tmp.template at<CPU>(0, s % num_streams), 
                                     c_tmp.template at<GPU>(0, s % num_streams),
                                     nrow * ncol, s % num_streams);

                        /* lock the buffer */
                        #pragma omp atomic write
                        buf_state[s % num_streams] = 1;
                    } else { /* this thread will do allreduce and store */
                        int state{0};
                        /* wait for the lock of the buffer */
                        while (!state) {
                            #pragma omp atomic read
                            state = buf_state[s % num_streams];
                        }
                        /* wait for the cuda stream to finish (both gemm and copyout) */
                        acc::sync_stream(s % num_streams);
                        /* sum over all MPI ranks */ 
                        comm.allreduce(c_tmp.template at<CPU>(0, s % num_streams), nrow * ncol);

                        /* store panel: go over the elements of the window and add the elements 
                         * to the resulting array; the .add() method skips the elements that are 
                         * not part of the local result matrix. */
                        #pragma omp parallel for
                        for (int jcol = 0; jcol < ncol; jcol++) {
                            for (int irow = 0; irow < nrow; irow++) {
                                /* .add() method takes the global (row, column) indices */
                                result__.set(irow0__ + i0 + irow, jcol0__ + j0 + jcol,
                                             c_tmp(irow + nrow * jcol, s % num_streams));
                            }
                        }

                        /* release the buffer */
                        #pragma omp atomic write
                        buf_state[s % num_streams] = 0;
                    }
                    s++;
                }
            }
        }
        omp_set_nested(0);
        omp_set_num_threads(nt);
        #endif
    }
    
    if (pu__ == CPU) {
        auto store_panel = [&req, &result__, &dims, &c_tmp, irow0__, jcol0__](int s)
        {
            MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);

            #pragma omp parallel for
            for (int jcol = 0; jcol < dims[s % 2][3]; jcol++) {
                for (int irow = 0; irow < dims[s % 2][2]; irow++) {
                    result__.set(irow0__ + irow +  dims[s % 2][0], jcol0__ + jcol +  dims[s % 2][1],
                                 c_tmp(irow + dims[s % 2][2] * jcol, s % 2));
                }
            }
        };

        int s{0};
        for (int ibc = 0; ibc < nbc; ibc++) {
            int j0 = ibc * BS;
            int ncol = std::min(n__, (ibc + 1) * BS) - j0;

            for (int ibr = 0; ibr < nbr; ibr++) {
                int i0 = ibr * BS;
                int nrow = std::min(m__, (ibr + 1) * BS) - i0;

                if (req[s % 2] != MPI_REQUEST_NULL) {
                    store_panel(s);
                }

                dims[s % 2][0] = i0;
                dims[s % 2][1] = j0;
                dims[s % 2][2] = nrow;
                dims[s % 2][3] = ncol;

                T* buf = (pu__ == CPU) ? c_tmp.template at<CPU>(0, s % 2) : c_tmp.template at<GPU>(0, s % 2);
                local_inner(i0__ + i0, nrow, j0__ + j0, ncol, buf, nrow, -1);

                comm.iallreduce(c_tmp.template at<CPU>(0, s % 2), nrow * ncol, &req[s % 2]);

                s++;
            }
        }

        for (int s: {0, 1}) {
            if (req[s % 2] != MPI_REQUEST_NULL) {
                store_panel(s);
            }
        }
    }

    if (sddk_pp) {
        comm.barrier();
        time += omp_get_wtime();
        int k = bra__.gkvec().num_gvec() + bra__.num_mt_coeffs();
        if (comm.rank() == 0) {
            printf("inner() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, time=%f (sec)]\n", ngop * m__ * n__ * k / time / comm.size(), m__, n__, k, time);
        }
    }
}
