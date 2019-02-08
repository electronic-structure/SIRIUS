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

template <typename T>
static void inner_local(memory_t mem__, linalg_t la__, int ispn__, Wave_functions& bra__,
                        int i0__, int m__, Wave_functions& ket__, int j0__, int n__, T* beta__,
                        T* buf__, int ld__, stream_id sid__);

template<>
void inner_local<double>(memory_t mem__, linalg_t la__, int ispn__, Wave_functions& bra__,
                         int i0__, int m__, Wave_functions& ket__, int j0__, int n__, double* beta__,
                         double* buf__, int ld__, stream_id sid__)
{
    utils::timer t1("sddk::inner|local");
    auto& comm = bra__.comm();
    auto spins = get_spins(ispn__);
    *beta__ = 0;
    for (auto s: spins) {
        if (bra__.has_mt()) {
            TERMINATE("not implemented");
        }
        linalg2(la__).gemm('C', 'N', m__, n__, 2 * bra__.pw_coeffs(s).num_rows_loc(),
                           &linalg_const<double>::two(),
                           reinterpret_cast<double*>(bra__.pw_coeffs(s).prime().at(bra__.preferred_memory_t(), 0, i0__)),
                           2 * bra__.pw_coeffs(s).prime().ld(),
                           reinterpret_cast<double*>(ket__.pw_coeffs(s).prime().at(ket__.preferred_memory_t(), 0, j0__)),
                           2 * ket__.pw_coeffs(s).prime().ld(),
                           beta__, buf__, ld__, sid__);
        /* subtract one extra G=0 contribution */
        if (comm.rank() == 0) {
            linalg_t la = is_host_memory(mem__) ? linalg_t::blas : linalg_t::cublas;
            linalg2(la).ger(m__, n__, &linalg_const<double>::m_one(),
                            reinterpret_cast<double*>(bra__.pw_coeffs(s).prime().at(bra__.preferred_memory_t(), 0, i0__)),
                            2 * bra__.pw_coeffs(s).prime().ld(),
                            reinterpret_cast<double*>(ket__.pw_coeffs(s).prime().at(ket__.preferred_memory_t(), 0, j0__)),
                            2 * ket__.pw_coeffs(s).prime().ld(),
                            buf__, ld__);
        }
        *beta__ = 1;
    }
}

template<>
void inner_local<double_complex>(memory_t mem__, linalg_t la__, int ispn__, Wave_functions& bra__,
                                 int i0__, int m__, Wave_functions& ket__, int j0__, int n__, double_complex* beta__,
                                 double_complex* buf__, int ld__, stream_id sid__)

{
    utils::timer t1("sddk::inner|local");
    auto spins = get_spins(ispn__);
    *beta__ = 0;
    for (auto s: spins) {
        linalg2(la__).gemm('C', 'N', m__, n__, bra__.pw_coeffs(s).num_rows_loc(),
                           &linalg_const<double_complex>::one(),
                           bra__.pw_coeffs(s).prime().at(bra__.preferred_memory_t(), 0, i0__),
                           bra__.pw_coeffs(s).prime().ld(),
                           ket__.pw_coeffs(s).prime().at(ket__.preferred_memory_t(), 0, j0__),
                           ket__.pw_coeffs(s).prime().ld(),
                           beta__, buf__, ld__, sid__);
        if (bra__.has_mt()) {
            linalg2(la__).gemm('C', 'N', m__, n__, bra__.mt_coeffs(s).num_rows_loc(),
                               &linalg_const<double_complex>::one(),
                               bra__.mt_coeffs(s).prime().at(bra__.preferred_memory_t(), 0, i0__),
                               bra__.mt_coeffs(s).prime().ld(),
                               ket__.mt_coeffs(s).prime().at(ket__.preferred_memory_t(), 0, j0__),
                               ket__.mt_coeffs(s).prime().ld(),
                               &linalg_const<double_complex>::one(),
                               buf__, ld__, sid__);
        }
        *beta__ = 1;
    }
}

/// Inner product between wave-functions.
/** This function computes the inner product using a moving window scheme plus allreduce.
 *  The input wave-functions data must be previously allocated on the GPU.
 *  The result is always returned in the CPU pointer. In case of a single MPI rank the result is also returned in the
 *  GPU pointer.
 *
 *  The following \f$ m \times n \f$ sub-matrix is computed:
 *  \f[
 *    S_{irow0+i,jcol0+j} = \langle \phi_{i0 + i} | \tilde \phi_{j0 + j} \rangle
 *  \f]
 *
 *  \param [in]  mem     Type of preferred memory for the resulting matrix.
 *  \param [in]  la      Type of BLAS linear algebra driver.
 *  \param [in]  ispn    Index of spin (0, 1, or 2; 2 means the contribution from two spinor components).
 *  \param [in]  bra     "bra" wave-functions \f$ \phi \f$.
 *  \param [in]  i0      Index of the first "bra" wave-function.
 *  \param [in]  m       Number of "bra" wave-functions.
 *  \param [in]  ket     "ket" wave-functions \f$ \tilde \phi \f$.
 *  \param [in]  j0      Index of the first "ket" wave-function.
 *  \param [in]  n       Number of "ket" wave-functions.
 *  \param [out] result  Resulting inner product matrix \f$ S \f$.
 *  \param [in]  irow0   First row (in the global matrix) of the inner product sub-matrix.
 *  \param [in]  jcol0   First column (in the global matix) of the inner product sub-matrix.
 */
template <typename T>
inline void inner(memory_t        mem__,
                  linalg_t        la__,
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
    PROFILE("sddk::inner");

    auto& comm = bra__.comm();

    auto sddk_pp = utils::get_env<int>("SDDK_PRINT_PERFORMANCE");

    auto sddk_bs_raw = utils::get_env<int>("SDDK_BLOCK_SIZE");

    int sddk_block_size = (sddk_bs_raw == nullptr) ? sddk_default_block_size : *sddk_bs_raw;

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

    T beta = 0;

    /* single MPI rank */
    if (comm.size() == 1) {
        inner_local<T>(mem__, la__, ispn__, bra__, i0__, m__, ket__, j0__, n__, &beta,
                       result__.at(mem__, irow0__, jcol0__), result__.ld(), stream_id(-1));
        if (is_device_memory(mem__)) {
            acc::copyout(result__.at(memory_t::host, irow0__, jcol0__), result__.ld(),
                         result__.at(memory_t::device, irow0__, jcol0__), result__.ld(),
                         m__, n__);
        }
        if (sddk_pp) {
            time += omp_get_wtime();
            int k = bra__.gkvec().num_gvec() + bra__.num_mt_coeffs();
            printf("inner() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, time=%f (sec)]\n",
                   ngop * m__ * n__ * k / time, m__, n__, k, time);
        }
        return;
    } else if (result__.comm().size() == 1) { /* parallel wave-functions distribution but sequential diagonalization */
        inner_local<T>(mem__, la__, ispn__, bra__, i0__, m__, ket__, j0__, n__, &beta,
                       result__.at(mem__, irow0__, jcol0__), result__.ld(), stream_id(-1));
        if (is_device_memory(mem__)) {
            utils::timer t1("sddk::inner|device_copy");
            acc::copyout(result__.at(memory_t::host, irow0__, jcol0__), result__.ld(),
                         result__.at(memory_t::device, irow0__, jcol0__), result__.ld(),
                         m__, n__);
            if (sddk_pp) {
                double t = t1.stop();
                if (comm.rank() == 0) {
                    printf("inner() copyout speed: %12.6f GB/s\n", m__ * n__ * sizeof(T) / std::pow(2.0, 30) / t);
                }
            }
        }
        utils::timer t3("sddk::inner|store");
        mdarray<T, 2> tmp(m__, n__);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n__; j++) {
            for (int i = 0; i < m__; i++) {
                tmp(i, j) = result__(irow0__ + i, jcol0__ + j);
            }
        }
        t3.stop();
        utils::timer t1("sddk::inner|mpi");
        comm.allreduce(tmp.at(memory_t::host), m__ * n__);
        t1.stop();
        utils::timer t2("sddk::inner|store");
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n__; j++) {
            for (int i = 0; i < m__; i++) {
                result__(irow0__ + i, jcol0__ + j) = tmp(i, j);
            }
        }
        t2.stop();
        if (is_device_memory(mem__)) {
            utils::timer t1("sddk::inner|device_copy");
            acc::copyin(result__.at(memory_t::device, irow0__, jcol0__), result__.ld(),
                        result__.at(memory_t::host, irow0__, jcol0__), result__.ld(),
                        m__, n__);
            if (sddk_pp) {
                double t = t1.stop();
                if (comm.rank() == 0) {
                    printf("inner() copyin speed: %12.6f GB/s\n", m__ * n__ * sizeof(T) / std::pow(2.0, 30) / t);
                }
            }
        }
        if (sddk_pp) {
            time += omp_get_wtime();
            int k = bra__.gkvec().num_gvec() + bra__.num_mt_coeffs();
            if (comm.rank() == 0) {
                printf("inner() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, time=%f (sec)]\n",
                       ngop * m__ * n__ * k / time / comm.size(), m__, n__, k, time);
            }
        }
        return;
    }

    /* fully parallel case */

    const int BS = sddk_block_size;

    const int num_streams{4};

    mdarray<T, 2> c_tmp;
    if (is_device_memory(mem__)) {
        c_tmp = mdarray<T, 2>(BS * BS, num_streams, memory_t::host_pinned, "inner::c_tmp");
        c_tmp.allocate(memory_t::device);
    } else {
        c_tmp = mdarray<T, 2>(BS * BS, num_streams, memory_t::host, "inner::c_tmp");
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

    if (is_device_memory(mem__)) {
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
                        inner_local<T>(mem__, la__, ispn__, bra__, i0__ + i0, nrow, ket__, j0__ + j0, ncol, &beta,
                                       c_tmp.at(memory_t::device, 0, s % num_streams), nrow, stream_id(-1));
                        /* enqueue a copyout operation */
                        acc::copyout(c_tmp.at(memory_t::host, 0, s % num_streams), 
                                     c_tmp.at(memory_t::device, 0, s % num_streams),
                                     nrow * ncol, stream_id(s % num_streams));

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
                        acc::sync_stream(stream_id(s % num_streams));
                        /* sum over all MPI ranks */ 
                        comm.allreduce(c_tmp.template at(memory_t::host, 0, s % num_streams), nrow * ncol);

                        /* store panel: go over the elements of the window and add the elements 
                         * to the resulting array; the .add() method skips the elements that are 
                         * not part of the local result matrix. */
                        #pragma omp parallel for num_threads(nt - 1)
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
    }

    if (is_host_memory(mem__)) {
        auto store_panel = [&req, &result__, &dims, &c_tmp, irow0__, jcol0__](int s)
        {
            utils::timer t1("sddk::inner|store");
            utils::timer t2("sddk::inner|store|mpi");
            MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);
            t2.stop();

            #pragma omp parallel for schedule(static)
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

                T* buf = c_tmp.at(mem__, 0, s % 2);
                inner_local<T>(mem__, la__, ispn__, bra__, i0__ + i0, nrow, ket__, j0__ + j0, ncol, &beta,
                               buf, nrow, stream_id(-1));

                comm.iallreduce(c_tmp.at(memory_t::host, 0, s % 2), nrow * ncol, &req[s % 2]);

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
