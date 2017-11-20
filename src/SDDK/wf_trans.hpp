/// Linear transformation of the wave-functions.
/** The transformation matrix is expected in the CPU memory. */
template <typename T>
inline void transform(device_t                     pu__,
                      double                       alpha__,
                      std::vector<wave_functions*> wf_in__,
                      int                          i0__,
                      int                          m__,
                      dmatrix<T>&                  mtrx__,
                      int                          irow0__,
                      int                          jcol0__,
                      double                       beta__,
                      std::vector<wave_functions*> wf_out__,
                      int                          j0__,
                      int                          n__)
{
    PROFILE("sddk::wave_functions::transform");

    static_assert(std::is_same<T, double>::value || std::is_same<T, double_complex>::value, "wrong type");

    assert(n__ != 0);
    assert(m__ != 0);

    assert(wf_in__.size() == wf_out__.size());
    int nwf = static_cast<int>(wf_in__.size()); 
    auto& comm = mtrx__.blacs_grid().comm();
    for (int i = 0; i < nwf; i++) { 
        assert(wf_in__[i]->pw_coeffs().num_rows_loc() == wf_out__[i]->pw_coeffs().num_rows_loc());
        if (wf_in__[i]->has_mt()) {
            assert(wf_in__[i]->mt_coeffs().num_rows_loc() == wf_out__[i]->mt_coeffs().num_rows_loc());
        }
        //assert(wf_in__[i]->comm().size() == comm.size());
        //assert(wf_out__[i]->comm().size() == comm.size());
    }
    
    double ngop{0};
    if (std::is_same<T, double>::value) {
        ngop = 2e-9;
    }
    if (std::is_same<T, double_complex>::value) {
        ngop = 8e-9;
    }

    const char* sddk_pp_raw = std::getenv("SDDK_PRINT_PERFORMANCE");
    int sddk_pp = (sddk_pp_raw == NULL) ? 0 : std::atoi(sddk_pp_raw);

    const char* sddk_bs_raw = std::getenv("SDDK_BLOCK_SIZE");
    int sddk_block_size = (sddk_bs_raw == NULL) ? sddk_default_block_size : std::atoi(sddk_bs_raw);

    T alpha = alpha__;
    
    /* perform a local {d,z}gemm; in case of GPU transformation is done in the stream#0 (not in the null stream!) */
    auto local_transform = [pu__](T*              alpha,
                                  wave_functions* wf_in__,
                                  int             i0__,
                                  int             m__,
                                  T*              ptr__,
                                  int             ld__,
                                  wave_functions* wf_out__,
                                  int             j0__,
                                  int             n__,
                                  int             stream_id__)
    {
        if (pu__ == CPU) {
            if (std::is_same<T, double_complex>::value) {
                /* transform plane-wave part */
                linalg<CPU>::gemm(0, 0, wf_in__->pw_coeffs().num_rows_loc(), n__, m__,
                                  *reinterpret_cast<double_complex*>(alpha),
                                  wf_in__->pw_coeffs().prime().at<CPU>(0, i0__), wf_in__->pw_coeffs().prime().ld(),
                                  reinterpret_cast<double_complex*>(ptr__), ld__,
                                  linalg_const<double_complex>::one(),
                                  wf_out__->pw_coeffs().prime().at<CPU>(0, j0__), wf_out__->pw_coeffs().prime().ld());
                /* transform muffin-tin part */
                if (wf_in__->has_mt() && wf_in__->mt_coeffs().num_rows_loc()) {
                    linalg<CPU>::gemm(0, 0, wf_in__->mt_coeffs().num_rows_loc(), n__, m__,
                                      *reinterpret_cast<double_complex*>(alpha),
                                      wf_in__->mt_coeffs().prime().at<CPU>(0, i0__), wf_in__->mt_coeffs().prime().ld(),
                                      reinterpret_cast<double_complex*>(ptr__), ld__,
                                      linalg_const<double_complex>::one(),
                                      wf_out__->mt_coeffs().prime().at<CPU>(0, j0__), wf_out__->mt_coeffs().prime().ld());
                }
            }

            if (std::is_same<T, double>::value) {
                linalg<CPU>::gemm(0, 0, 2 * wf_in__->pw_coeffs().num_rows_loc(), n__, m__,
                                  *reinterpret_cast<double*>(alpha),
                                  reinterpret_cast<double*>(wf_in__->pw_coeffs().prime().at<CPU>(0, i0__)), 2 * wf_in__->pw_coeffs().prime().ld(),
                                  reinterpret_cast<double*>(ptr__), ld__,
                                  linalg_const<double>::one(),
                                  reinterpret_cast<double*>(wf_out__->pw_coeffs().prime().at<CPU>(0, j0__)), 2 * wf_out__->pw_coeffs().prime().ld());
                if (wf_in__->has_mt()) {
                    TERMINATE("not implemented");
                }
            }
        }
        #ifdef __GPU
        if (pu__ == GPU) {
            if (std::is_same<T, double_complex>::value) {
                linalg<GPU>::gemm(0, 0, wf_in__->pw_coeffs().num_rows_loc(), n__, m__,
                                  reinterpret_cast<double_complex*>(alpha),
                                  wf_in__->pw_coeffs().prime().at<GPU>(0, i0__), wf_in__->pw_coeffs().prime().ld(),
                                  reinterpret_cast<double_complex*>(ptr__), ld__,
                                  &linalg_const<double_complex>::one(),
                                  wf_out__->pw_coeffs().prime().at<GPU>(0, j0__), wf_out__->pw_coeffs().prime().ld(),
                                  stream_id__);

                if (wf_in__->has_mt() && wf_in__->mt_coeffs().num_rows_loc()) {
                    linalg<GPU>::gemm(0, 0, wf_in__->mt_coeffs().num_rows_loc(), n__, m__,
                                      reinterpret_cast<double_complex*>(alpha),
                                      wf_in__->mt_coeffs().prime().at<GPU>(0, i0__), wf_in__->mt_coeffs().prime().ld(),
                                      reinterpret_cast<double_complex*>(ptr__), ld__,
                                      &linalg_const<double_complex>::one(),
                                      wf_out__->mt_coeffs().prime().at<GPU>(0, j0__), wf_out__->mt_coeffs().prime().ld(),
                                      stream_id__);
                }
            }

            if (std::is_same<T, double>::value) {
                linalg<GPU>::gemm(0, 0, 2 * wf_in__->pw_coeffs().num_rows_loc(), n__, m__,
                                  reinterpret_cast<double*>(alpha),
                                  reinterpret_cast<double*>(wf_in__->pw_coeffs().prime().at<GPU>(0, i0__)), 2 * wf_in__->pw_coeffs().prime().ld(),
                                  reinterpret_cast<double*>(ptr__), ld__,
                                  &linalg_const<double>::one(),
                                  reinterpret_cast<double*>(wf_out__->pw_coeffs().prime().at<GPU>(0, j0__)), 2 * wf_out__->pw_coeffs().prime().ld(),
                                  stream_id__);
                if (wf_in__->has_mt()) {
                    TERMINATE("not implemented");
                }
            }
        }
        #endif
    };
    
    sddk::timer t1("sddk::wave_functions::transform|init");
    /* initial values for the resulting wave-functions */
    for (int iv = 0; iv < nwf; iv++) {
        if (pu__ == CPU) {
            if (beta__ == 0) {
                /* zero PW part */
                for (int j = 0; j < n__; j++) {
                    std::memset(wf_out__[iv]->pw_coeffs().prime().at<CPU>(0, j0__ + j),
                                0,
                                wf_out__[iv]->pw_coeffs().num_rows_loc() * sizeof(double_complex));
                }
                /* zero MT part */
                if (wf_out__[iv]->has_mt() && wf_out__[iv]->mt_coeffs().num_rows_loc()) {
                    for (int j = 0; j < n__; j++) {
                        std::memset(wf_out__[iv]->mt_coeffs().prime().at<CPU>(0, j0__ + j),
                                    0,
                                    wf_out__[iv]->mt_coeffs().num_rows_loc() * sizeof(double_complex));
                    }
                }

            } else {
                /* scale PW part */
                for (int j = 0; j < n__; j++) {
                    for (int k = 0; k < wf_out__[iv]->pw_coeffs().num_rows_loc(); k++) {
                        wf_out__[iv]->pw_coeffs().prime(k, j0__ + j) *= beta__;
                    }
                    /* scale MT part */
                    if (wf_out__[iv]->has_mt() && wf_out__[iv]->mt_coeffs().num_rows_loc()) {
                        for (int k = 0; k < wf_out__[iv]->mt_coeffs().num_rows_loc(); k++) {
                            wf_out__[iv]->mt_coeffs().prime(k, j0__ + j) *= beta__;
                        }
                    }
                }
            }
        }
        #ifdef __GPU
        if (pu__ == GPU) {
            if (beta__ == 0) {
                /* zero PW part */
                acc::zero(wf_out__[iv]->pw_coeffs().prime().at<GPU>(0, j0__),
                          wf_out__[iv]->pw_coeffs().prime().ld(),
                          wf_out__[iv]->pw_coeffs().num_rows_loc(),
                          n__);
                /* zero MT part */
                if (wf_out__[iv]->has_mt() && wf_out__[iv]->mt_coeffs().num_rows_loc()) {
                    acc::zero(wf_out__[iv]->mt_coeffs().prime().at<GPU>(0, j0__),
                              wf_out__[iv]->mt_coeffs().prime().ld(),
                              wf_out__[iv]->mt_coeffs().num_rows_loc(),
                              n__);
                }
            } else {
                /* scale PW part */
                scale_matrix_elements_gpu((cuDoubleComplex*)wf_out__[iv]->pw_coeffs().prime().at<GPU>(0, j0__),
                                          wf_out__[iv]->pw_coeffs().prime().ld(),
                                          wf_out__[iv]->pw_coeffs().num_rows_loc(),
                                          n__,
                                          beta__);
                /* scale MT part */
                if (wf_out__[iv]->has_mt() && wf_out__[iv]->mt_coeffs().num_rows_loc()) {
                    scale_matrix_elements_gpu((cuDoubleComplex*)wf_out__[iv]->mt_coeffs().prime().at<GPU>(0, j0__),
                                              wf_out__[iv]->mt_coeffs().prime().ld(),
                                              wf_out__[iv]->mt_coeffs().num_rows_loc(),
                                              n__,
                                              beta__);
                }
            }
        }
        #endif
    }
    t1.stop();
    
    if (sddk_pp) {
        comm.barrier();
    }
    double time = -omp_get_wtime();
    
    /* trivial case */
    if (mtrx__.blacs_grid().comm().size() == 1) {
        #ifdef __GPU
        if (pu__ == GPU) {
            acc::copyin(mtrx__.template at<GPU>(irow0__, jcol0__), mtrx__.ld(),
                        mtrx__.template at<CPU>(irow0__, jcol0__), mtrx__.ld(), m__, n__, 0);
        }
        #endif
        T* ptr{nullptr};
        switch (pu__) {
            case CPU: {
                ptr = mtrx__.template at<CPU>(irow0__, jcol0__);
                break;
            }
            case GPU: {
                ptr = mtrx__.template at<GPU>(irow0__, jcol0__);
                break;
            }
        }
                
        for (int iv = 0; iv < nwf; iv++) {
            local_transform(&alpha, wf_in__[iv], i0__, m__, ptr, mtrx__.ld(), wf_out__[iv], j0__, n__, 0);
        }
        #ifdef __GPU
        if (pu__ == GPU) {
            /* wait for the stream to finish zgemm */
            acc::sync_stream(0);
        }
        #endif
        if (sddk_pp) {
            time += omp_get_wtime();
            int k = wf_in__[0]->pw_coeffs().num_rows_loc();
            if (wf_in__[0]->has_mt()) {
                k += wf_in__[0]->mt_coeffs().num_rows_loc();
            }
            printf("transform() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, nvec=%i, time=%f (sec)]\n",
                   ngop * m__ * n__ * k * nwf / time, k, n__, m__, nwf, time);
        }
        return;
    }

    const int BS = sddk_block_size;

    const int num_streams{4};

    mdarray<T, 1> buf(BS * BS, memory_t::host_pinned, "transform::buf");
    mdarray<T, 3> submatrix(BS, BS, num_streams, memory_t::host_pinned, "transform::submatrix");

    if (pu__ == GPU) {
        submatrix.allocate(memory_t::device);
    }

    /* cache cartesian ranks */
    mdarray<int, 2> cart_rank(mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().num_ranks_col());
    for (int i = 0; i < mtrx__.blacs_grid().num_ranks_col(); i++) {
        for (int j = 0; j < mtrx__.blacs_grid().num_ranks_row(); j++) {
            cart_rank(j, i) = mtrx__.blacs_grid().cart_rank(j, i);
        }
    }

    int nbr = m__ / BS + std::min(1, m__ % BS);
    int nbc = n__ / BS + std::min(1, n__ % BS);

    block_data_descriptor sd(comm.size());

    double time_mpi{0};
    
    for (int ibr = 0; ibr < nbr; ibr++) {
        /* global index of row */
        int i0 = ibr * BS;
        /* actual number of rows in the submatrix */
        int nrow = std::min(m__, (ibr + 1) * BS) - i0;

        assert(nrow != 0);

        splindex<block_cyclic> spl_row_begin(irow0__ + i0,        mtrx__.num_ranks_row(), mtrx__.rank_row(), mtrx__.bs_row());
        splindex<block_cyclic>   spl_row_end(irow0__ + i0 + nrow, mtrx__.num_ranks_row(), mtrx__.rank_row(), mtrx__.bs_row());

        int local_size_row = spl_row_end.local_size() - spl_row_begin.local_size();

        int s{0};
        for (int ibc = 0; ibc < nbc; ibc++) {
            /* global index of column */
            int j0 = ibc * BS;
            /* actual number of columns in the submatrix */
            int ncol = std::min(n__, (ibc + 1) * BS) - j0;

            assert(ncol != 0);
            
            splindex<block_cyclic> spl_col_begin(jcol0__ + j0,        mtrx__.num_ranks_col(), mtrx__.rank_col(), mtrx__.bs_col());
            splindex<block_cyclic>   spl_col_end(jcol0__ + j0 + ncol, mtrx__.num_ranks_col(), mtrx__.rank_col(), mtrx__.bs_col());

            int local_size_col = spl_col_end.local_size() - spl_col_begin.local_size();

            /* total number of elements owned by the current rank in the block */
            for (int i = 0; i < mtrx__.blacs_grid().num_ranks_col(); i++) {
                int scol = spl_col_end.local_size(i) - spl_col_begin.local_size(i);
                for (int j = 0; j < mtrx__.blacs_grid().num_ranks_row(); j++) {
                    int l = cart_rank(j, i);
                    sd.counts[l] = (spl_row_end.local_size(j) - spl_row_begin.local_size(j)) * scol;
                }
            }

            sd.calc_offsets();

            assert(sd.offsets.back() + sd.counts.back() <= (int)buf.size());
            /* fetch elements of sub-matrix */
            if (local_size_row) {
                for (int j = 0; j < local_size_col; j++) {
                    std::memcpy(&buf[sd.offsets[comm.rank()] + local_size_row * j],
                                &mtrx__(spl_row_begin.local_size(), spl_col_begin.local_size() + j),
                                local_size_row * sizeof(T));
                }
            }
            double t0 = omp_get_wtime();
            /* collect submatrix */
            comm.allgather(&buf[0], sd.counts.data(), sd.offsets.data());
            time_mpi += (omp_get_wtime() - t0);

            #ifdef __GPU
            if (pu__ == GPU) {
                /* wait for the data copy; as soon as this is done, CPU buffer is free and can be reused */
                acc::sync_stream(s % num_streams);
            }
            #endif
            
            /* unpack data */
            std::vector<int> counts(comm.size(), 0);
            for (int jcol = 0; jcol < ncol; jcol++) {
                auto pos_jcol = mtrx__.spl_col().location(jcol0__ + j0 + jcol);
                for (int irow = 0; irow < nrow; irow++) {
                    auto pos_irow = mtrx__.spl_row().location(irow0__ + i0 + irow);
                    int rank = cart_rank(pos_irow.rank, pos_jcol.rank);

                    submatrix(irow, jcol, s % num_streams) = buf[sd.offsets[rank] + counts[rank]];
                    counts[rank]++;
                }
            }
            for (int rank = 0; rank < comm.size(); rank++) {
                assert(sd.counts[rank] == counts[rank]);
            }
            #ifdef __GPU
            if (pu__ == GPU) {
                acc::copyin(submatrix.template at<GPU>(0, 0, s % num_streams), submatrix.ld(),
                            submatrix.template at<CPU>(0, 0, s % num_streams), submatrix.ld(),
                            nrow, ncol, s % num_streams);
            }
            #endif
            T* ptr{nullptr};
            switch (pu__) {
                case CPU: {
                    ptr = submatrix.template at<CPU>(0, 0, s % num_streams);
                    break;
                }
                case GPU: {
                    ptr = submatrix.template at<GPU>(0, 0, s % num_streams);
                    break;
                }
            }
            for (int iv = 0; iv < nwf; iv++) {
                local_transform(&alpha, wf_in__[iv], i0__ + i0, nrow, ptr, BS, wf_out__[iv], j0__ + j0, ncol, s % num_streams);
            }
            s++;
        } /* loop over ibc */
        #ifdef __GPU
        if (pu__ == GPU) {
            /* wait for the full block of columns (update of different wave-functions); 
             * otherwise cuda streams can start updating the same block of output wave-functions */
            for (int s = 0; s < num_streams; s++) {
                acc::sync_stream(s);
            }
        }
        #endif
    } /* loop over ibr */

    if (sddk_pp) {
        comm.barrier();
        time += omp_get_wtime();
        int k = wf_in__[0]->pw_coeffs().num_rows_loc();
        if (wf_in__[0]->has_mt()) {
            k += wf_in__[0]->mt_coeffs().num_rows_loc();
        }
        comm.allreduce(&k, 1);
        if (comm.rank() == 0) {
            printf("transform() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, nvec=%i, time=%f (sec), time_mpi=%f (sec)]\n",
                   ngop * m__ * n__ * k * nwf / time / comm.size(), k, n__, m__, nwf,  time, time_mpi);
        }
    }
}

template <typename T>
inline void transform(device_t pu__,
                      std::vector<wave_functions*> wf_in__,
                      int i0__,
                      int m__,
                      dmatrix<T>& mtrx__,
                      int irow0__,
                      int jcol0__,
                      std::vector<wave_functions*> wf_out__,
                      int j0__,
                      int n__)
{
    transform<T>(pu__, 1.0, wf_in__, i0__, m__, mtrx__, irow0__, jcol0__, 0.0, wf_out__, j0__, n__);
}

/// Linear transformation of wave-functions.
/** The following operation is performed:
 *  \f[
 *     \psi^{out}_{j} = \alpha \sum_{i} \psi^{in}_{i} Z_{ij} + \beta \psi^{out}_{j}
 *  \f]
 */
template <typename T>
inline void transform(device_t pu__,
                      double          alpha__,
                      wave_functions& wf_in__,
                      int             i0__,
                      int             m__,
                      dmatrix<T>&     mtrx__,
                      int             irow0__,
                      int             jcol0__,
                      double          beta__,
                      wave_functions& wf_out__,
                      int             j0__,
                      int             n__)
{
    transform<T>(pu__, alpha__, {&wf_in__}, i0__, m__, mtrx__, irow0__, jcol0__, beta__, {&wf_out__}, j0__, n__);
}

template <typename T>
inline void transform(device_t        pu__,
                      wave_functions& wf_in__,
                      int             i0__,
                      int             m__,
                      dmatrix<T>&     mtrx__,
                      int             irow0__,
                      int             jcol0__,
                      wave_functions& wf_out__,
                      int             j0__,
                      int             n__)
{
    transform<T>(pu__, 1.0, {&wf_in__}, i0__, m__, mtrx__, irow0__, jcol0__, 0.0, {&wf_out__}, j0__, n__);
}

template <typename T>
inline void transform(device_t                     pu__,
                      double                       alpha__,
                      std::vector<Wave_functions*> wf_in__,
                      int                          i0__,
                      int                          m__,
                      dmatrix<T>&                  mtrx__,
                      int                          irow0__,
                      int                          jcol0__,
                      double                       beta__,
                      std::vector<Wave_functions*> wf_out__,
                      int                          j0__,
                      int                          n__)
{
    assert(wf_in__.size() == wf_out__.size());
    for (size_t i = 0; i < wf_in__.size(); i++) {
        assert(wf_in__[i]->num_components() == wf_in__[0]->num_components());
        assert(wf_in__[i]->num_components() == wf_out__[i]->num_components());
    }
    int num_sc = wf_in__[0]->num_components();
    for (int is = 0; is < num_sc; is++) {
        std::vector<wave_functions*> wf_in;
        std::vector<wave_functions*> wf_out;
        for (size_t i = 0; i < wf_in__.size(); i++) {
            wf_in.push_back(&wf_in__[i]->component(is));
            wf_out.push_back(&wf_out__[i]->component(is));
        }
        transform(pu__, alpha__, wf_in, i0__, m__, mtrx__, irow0__, jcol0__, beta__, wf_out, j0__, n__);
    }
}
