#include <thread>
#include <atomic>
#include <sirius.h>

#ifdef __MKL
#include <mkl_service.h>
#endif

class Measurement: public std::vector<double>
{
    public:

        inline double val(size_t i)
        {
            return (*this)[i];
        }

        double average()
        {
            double d = 0;
            for (size_t i = 0; i < this->size(); i++)
                d += val(i);
            d /= static_cast<double>(this->size());
            return d;
        }

        double sigma()
        {
            double avg = average();
            double variance = 0;
            for (size_t i = 0; i < this->size(); i++)
                variance += std::pow(val(i) - avg, 2);
            variance /= static_cast<double>(this->size());
            return std::sqrt(variance);
        }
};

#define COMMON_CODE_BLOCK                                                         \
                                                                                  \
    int bs = 32;                                                                  \
    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid[0], mpi_grid[1]);            \
                                                                                  \
    splindex<block> spl_K(K, mpi_comm_world().size(), mpi_comm_world().rank());   \
                                                                                  \
    matrix<double_complex> a(spl_K.local_size(), M);                              \
    matrix<double_complex> b(spl_K.local_size(), N);                              \
                                                                                  \
    dmatrix<double_complex> c(M, N, blacs_grid, bs, bs);                          \
    c.zero();                                                                     \
                                                                                  \
    for (int i = 0; i < M; i++)                                                   \
    {                                                                             \
        for (int j = 0; j < spl_K.local_size(); j++) a(j, i) = 0.1;               \
    }                                                                             \
    for (int i = 0; i < N; i++)                                                   \
    {                                                                             \
        for (int j = 0; j < spl_K.local_size(); j++) b(j, i) = 0.1;               \
    }                                                                             \



void test_nested_omp(int BS)
{
    int m{BS}, n{BS}, k{5678};

    mdarray<double_complex, 2> a(k, m);
    mdarray<double_complex, 2> b(k, n);
    mdarray<double_complex, 2> c(m, n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) a(j, i) = 0.1;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) b(j, i) = 0.1;

    c.zero();
    
    /* warmup */
    linalg<device_t::CPU>::gemm(2, 0, m, n, k, a, b, c);

    double t = omp_get_wtime();
    
    linalg<device_t::CPU>::gemm(2, 0, m, n, k, a, b, c);

    double t0 = omp_get_wtime() - t;
    double perf = 8e-9 * m * n * k / t0;

    if (mpi_comm_world().rank() == 0)
    {
        printf("test_nested_omp: performance (all threads): %f Gflops\n", perf);
    }

    omp_set_nested(1);
    int nt = omp_get_max_threads();
    if (nt < 2) TERMINATE("minimum two threads are required");

    double tcomp{0};

    #pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 1)
        {
            #ifdef __MKL
            mkl_set_num_threads_local(nt - 1);
            #endif
            omp_set_num_threads(nt - 1);

            double t = omp_get_wtime();
            linalg<device_t::CPU>::gemm(2, 0, m, n, k, a, b, c);
            tcomp = omp_get_wtime() - t;
        }
        else // thread#0
        {
            //double t = omp_get_wtime();
            //while (omp_get_wtime() - t < 2);
        }
    }

    omp_set_nested(0);
    #ifdef __MKL
    mkl_set_num_threads_local(0);
    #endif
    omp_set_num_threads(nt);

    perf = 8e-9 * m * n * k / tcomp;

    if (mpi_comm_world().rank() == 0)
    {
        printf("test_nested_omp: performance (N-1 threads): %f Gflops\n", perf);
    }

    t = omp_get_wtime();
    
    linalg<device_t::CPU>::gemm(2, 0, m, n, k, a, b, c);

    t0 = omp_get_wtime() - t;
    perf = 8e-9 * m * n * k / t0;

    if (mpi_comm_world().rank() == 0)
    {
        printf("test_nested_omp: performance (all threads): %f Gflops\n", perf);
    }

    omp_set_nested(1);

    #pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 1)
        {
            #ifdef __MKL
            mkl_set_num_threads_local(nt - 1);
            #endif
            omp_set_num_threads(nt - 1);

            for (int i = 0; i < 4; i++)
            {
                double t = omp_get_wtime();
                linalg<device_t::CPU>::gemm(2, 0, m, n, k, a, b, c);
                double perf = 8e-9 * m * n * k / (omp_get_wtime() - t);
                if (mpi_comm_world().rank() == 0)
                {
                    printf("test_nested_omp: performance (N-1 threads): %f Gflops\n", perf);
                    printf("    num_threads: %i\n", omp_get_max_threads());
                }
            }
        }
        else // thread#0
        {
            double t = omp_get_wtime();
            while (omp_get_wtime() - t < 1);
        }
    }

    omp_set_nested(0);
    #ifdef __MKL
    mkl_set_num_threads_local(0);
    #endif
    omp_set_num_threads(nt);
}

double wf_inner_simple(int M, int N, int K, std::vector<int> mpi_grid)
{
    if (mpi_comm_world().rank() == 0)
    {
        printf("=== wf_inner_simple ===\n");
    }

    COMMON_CODE_BLOCK

    mdarray<double_complex, 2> c_tmp(M, N);
    c_tmp.zero();
    
    mpi_comm_world().barrier();
    double t0 = omp_get_wtime();
    
    linalg<device_t::CPU>::gemm(2, 0, M, N, spl_K.local_size(), a, b, c_tmp);

    double t1 = omp_get_wtime();

    mpi_comm_world().allreduce(c_tmp.at<CPU>(), M * N);

    double t2 = omp_get_wtime();

    #pragma omp parallel for
    for (int icol = 0; icol < N; icol++)
    {
        for (int irow = 0; irow < M; irow++)
        {
            c.set(irow, icol, c_tmp(irow, icol));
        }
    }

    mpi_comm_world().barrier();
    double t3 = omp_get_wtime();

    double perf = 8e-9 * M * N * K / (t3 - t0) / mpi_comm_world().size();

    if (mpi_comm_world().rank() == 0)
    {
        printf("  gemm time: %f sec.\n", t1 - t0);
        printf("  comm time: %f sec.\n", t2 - t1);
        printf(" store time: %f sec.\n", t3 - t2);
        printf("performance: %f Gflops / rank\n", perf);
    }

    return perf;
}

double wf_inner_reduce_to_one(int M, int N, int K, std::vector<int> mpi_grid)
{
    if (mpi_comm_world().rank() == 0)
    {
        printf("=== wf_inner_reduce_to_one ===\n");
    }

    COMMON_CODE_BLOCK

    mdarray<double_complex, 1> c_tmp(c.num_rows_local(0) * c.num_cols_local(0));
    c_tmp.zero();

    mdarray<double_complex, 2> a_tmp(spl_K.local_size(), c.num_rows_local(0));
    a_tmp.zero();
    mdarray<double_complex, 2> b_tmp(spl_K.local_size(), c.num_cols_local(0));
    b_tmp.zero();

    if (mpi_grid[0] != mpi_grid[1]) TERMINATE("square grid only");
    block_data_descriptor bdd(mpi_grid[0]);
    for (int i = 0; i < mpi_grid[0]; i++) bdd.counts[i] = c.num_rows_local(i);
    bdd.calc_offsets();

    mpi_comm_world().barrier();
    double t0 = -omp_get_wtime();
    double tcomm{0}, tcomp{0}, tcopy{0};
    for (int rank_col = 0; rank_col < mpi_grid[1]; rank_col++)
    {
        for (int rank_row = 0; rank_row < mpi_grid[0]; rank_row++)
        {
            double t1 = omp_get_wtime();
            //#pragma omp parallel for
            //for (int i = 0; i < c.num_rows_local(rank_row); i++)
            //    std::memcpy(&a_tmp(0, i), &a(0, c.spl_row().global_index(i, rank_row)), spl_K.local_size() * sizeof(double_complex));

            //#pragma omp parallel for
            //for (int i = 0; i < c.num_cols_local(rank_col); i++)
            //    std::memcpy(&b_tmp(0, i), &b(0, c.spl_col().global_index(i, rank_col)), spl_K.local_size() * sizeof(double_complex));

            double t2 = omp_get_wtime();
            //linalg<device_t::CPU>::gemm(2, 0, c.num_rows_local(rank_row), c.num_cols_local(rank_col), spl_K.local_size(),
            //                  a_tmp.at<CPU>(), a_tmp.ld(), b_tmp.at<CPU>(), b_tmp.ld(), c_tmp.at<CPU>(), c.num_rows_local(rank_row));
            linalg<device_t::CPU>::gemm(2, 0, c.num_rows_local(rank_row), c.num_cols_local(rank_col), spl_K.local_size(),
                              a.at<CPU>(0, bdd.offsets[rank_row]), a.ld(), b.at<CPU>(0, bdd.offsets[rank_col]), b.ld(),
                              c_tmp.at<CPU>(), c.num_rows_local(rank_row));

            double t3 = omp_get_wtime();
            mpi_comm_world().reduce(c_tmp.at<CPU>(), c.at<CPU>(), c.num_rows_local(rank_row) * c.num_cols_local(rank_col),
                                    blacs_grid.cart_rank(rank_row, rank_col));
            double t4 = omp_get_wtime();

            tcopy += (t2 - t1);
            tcomp += (t3 - t2);
            tcomm += (t4 - t3);
        }
    }
    mpi_comm_world().barrier();
    t0 += omp_get_wtime();

    double perf = 8e-9 * M * N * K / t0 / mpi_comm_world().size();

    if (mpi_comm_world().rank() == 0)
    {
        printf("  gemm time: %f sec.\n", tcomp);
        printf("  comm time: %f sec.\n", tcomm);
        printf("  copy time: %f sec.\n", tcopy);
        printf("performance: %f Gflops / rank\n", perf);
    }
    
    for (int i = 0; i < c.num_cols_local(); i++)
    {
        for (int j = 0; j < c.num_rows_local(); j++)
        {
            if (std::abs(c(j, i) - 0.01 * K) > 1e-10) TERMINATE("result is wrong");
        }
    }
    return perf;
}

double wf_inner_reduce_to_one_async(int M, int N, int K, std::vector<int> mpi_grid)
{
    if (mpi_comm_world().rank() == 0)
    {
        printf("=== wf_inner_reduce_to_one_async ===\n");
    }

    COMMON_CODE_BLOCK

    mdarray<double_complex, 2> c_tmp(c.num_rows_local(0) * c.num_cols_local(0), 2);
    c_tmp.zero();

    mdarray<double_complex, 2> a_tmp(spl_K.local_size(), c.num_rows_local(0));
    a_tmp.zero();
    mdarray<double_complex, 2> b_tmp(spl_K.local_size(), c.num_cols_local(0));
    b_tmp.zero();

    if (mpi_grid[0] != mpi_grid[1]) TERMINATE("square grid only");
    block_data_descriptor bdd(mpi_grid[0]);
    for (int i = 0; i < mpi_grid[0]; i++) bdd.counts[i] = c.num_rows_local(i);
    bdd.calc_offsets();

    std::array<MPI_Request, 2> req = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    
    mpi_comm_world().barrier();
    double t0 = -omp_get_wtime();
    double tcomm{0}, tcomp{0}, tcopy{0};

    int s = 0;
    for (int rank_col = 0; rank_col < mpi_grid[1]; rank_col++)
    {
        for (int rank_row = 0; rank_row < mpi_grid[0]; rank_row++)
        {
            double t1 = omp_get_wtime();

            //#pragma omp parallel for
            //for (int i = 0; i < c.num_rows_local(rank_row); i++)
            //    std::memcpy(&a_tmp(0, i), &a(0, c.spl_row().global_index(i, rank_row)), spl_K.local_size() * sizeof(double_complex));

            //#pragma omp parallel for
            //for (int i = 0; i < c.num_cols_local(rank_col); i++)
            //    std::memcpy(&b_tmp(0, i), &b(0, c.spl_col().global_index(i, rank_col)), spl_K.local_size() * sizeof(double_complex));

            double t2 = omp_get_wtime();

            if (req[s % 2] != MPI_REQUEST_NULL)
                MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);

            double t3 = omp_get_wtime();

            //linalg<device_t::CPU>::gemm(2, 0, c.num_rows_local(rank_row), c.num_cols_local(rank_col), spl_K.local_size(),
            //                  a_tmp.at<CPU>(), a_tmp.ld(), b_tmp.at<CPU>(), b_tmp.ld(), c_tmp.at<CPU>(0, s % 2), c.num_rows_local(rank_row));
            linalg<device_t::CPU>::gemm(2, 0, c.num_rows_local(rank_row), c.num_cols_local(rank_col), spl_K.local_size(),
                              a.at<CPU>(0, bdd.offsets[rank_row]), a.ld(), b.at<CPU>(0, bdd.offsets[rank_col]), b.ld(),
                              c_tmp.at<CPU>(0, s % 2), c.num_rows_local(rank_row));

            double t4 = omp_get_wtime();

            mpi_comm_world().reduce(c_tmp.at<CPU>(0, s % 2), c.at<CPU>(), c.num_rows_local(rank_row) * c.num_cols_local(rank_col),
                                    blacs_grid.cart_rank(rank_row, rank_col), &req[s % 2]);

            s++;

            tcopy += (t2 - t1);
            tcomm += (t3 - t2);
            tcomp += (t4 - t3);
        }
    }
    double t2 = omp_get_wtime();
    for (int s: {0, 1})
    {
        if (req[s] != MPI_REQUEST_NULL)
            MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);
    }
    double t3 = omp_get_wtime();
    tcomm += (t3 - t2);

    mpi_comm_world().barrier();
    t0 += omp_get_wtime();

    double perf = 8e-9 * M * N * K / t0 / mpi_comm_world().size();

    if (mpi_comm_world().rank() == 0)
    {
        printf("  gemm time: %f sec.\n", tcomp);
        printf("  comm time: %f sec.\n", tcomm);
        printf("  copy time: %f sec.\n", tcopy);
        printf("performance: %f Gflops / rank\n", perf);
    }
    
    for (int i = 0; i < c.num_cols_local(); i++)
    {
        for (int j = 0; j < c.num_rows_local(); j++)
        {
            if (std::abs(c(j, i) - 0.01 * K) > 1e-10) TERMINATE("result is wrong");
        }
    }
    return perf;
}

double wf_inner_allreduce(int M, int N, int K, std::vector<int> mpi_grid, int BS)
{
    if (mpi_comm_world().rank() == 0)
    {
        printf("=== wf_inner_allreduce ===\n");
    }

    COMMON_CODE_BLOCK

    mdarray<double_complex, 1> c_tmp(BS * BS);
    c_tmp.zero();

    int nbr = M / BS + std::min(1, M % BS);
    int nbc = N / BS + std::min(1, N % BS);

    double tcomm{0}, tcomp{0}, tstore{0};
    
    mpi_comm_world().barrier();
    double t0 = -omp_get_wtime();
    for (int ibc = 0; ibc < nbc; ibc++)
    {
        int col0 = ibc * BS;
        int ncol = std::min(N, (ibc + 1) * BS) - col0;

        for (int ibr = 0; ibr < nbr; ibr++)
        {
            int row0 = ibr * BS;
            int nrow = std::min(M, (ibr + 1) * BS) - row0;

            double t = omp_get_wtime();
            linalg<device_t::CPU>::gemm(2, 0, nrow, ncol, spl_K.local_size(),
                              a.at<CPU>(0, row0), a.ld(), b.at<CPU>(0, col0), b.ld(),
                              c_tmp.at<CPU>(), nrow);
            tcomp += (omp_get_wtime() - t);

            t = omp_get_wtime();
            mpi_comm_world().allreduce(c_tmp.at<CPU>(), nrow * ncol);
            tcomm += (omp_get_wtime() - t);

            t = omp_get_wtime();
            #pragma omp parallel for
            for (int icol = 0; icol < ncol; icol++)
            {
                for (int irow = 0; irow < nrow; irow++)
                {
                    c.set(irow + row0, icol + col0, c_tmp(irow + nrow * icol));
                }
            }
            tstore += (omp_get_wtime() - t);
        }
    }

    mpi_comm_world().barrier();
    t0 += omp_get_wtime();

    double perf = 8e-9 * M * N * K / t0 / mpi_comm_world().size();

    if (mpi_comm_world().rank() == 0)
    {
        printf("  gemm time: %f sec.\n", tcomp);
        printf("  comm time: %f sec.\n", tcomm);
        printf(" store time: %f sec.\n", tstore);
        printf("performance: %f Gflops / rank\n", perf);
    }

    for (int i = 0; i < c.num_cols_local(); i++)
    {
        for (int j = 0; j < c.num_rows_local(); j++)
        {
            if (std::abs(c(j, i) - 0.01 * K) > 1e-10) TERMINATE("result is wrong");
        }
    }

    return perf;
}

double wf_inner_allreduce_async(int M, int N, int K, std::vector<int> mpi_grid, int BS)
{
    if (mpi_comm_world().rank() == 0)
    {
        printf("=== wf_inner_allreduce_async ===\n");
    }

    COMMON_CODE_BLOCK

    mdarray<double_complex, 2> c_tmp(BS * BS, 2);
    c_tmp.zero();

    int nbr = M / BS + std::min(1, M % BS);
    int nbc = N / BS + std::min(1, N % BS);

    std::array<MPI_Request, 2> req = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    std::array<std::array<int, 4>, 2> dims;

    double tcomm{0}, tcomp{0}, tstore{0};
    
    mpi_comm_world().barrier();
    double t0 = -omp_get_wtime();
    int s = 0;
    for (int ibc = 0; ibc < nbc; ibc++)
    {
        int col0 = ibc * BS;
        int ncol = std::min(N, (ibc + 1) * BS) - col0;

        for (int ibr = 0; ibr < nbr; ibr++)
        {
            int row0 = ibr * BS;
            int nrow = std::min(M, (ibr + 1) * BS) - row0;

            if (req[s % 2] != MPI_REQUEST_NULL)
            {
                double t = omp_get_wtime();
                MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);
                tcomm += (omp_get_wtime() - t);

                t = omp_get_wtime();
                #pragma omp parallel for
                for (int icol = 0; icol < dims[s % 2][3]; icol++)
                {
                    for (int irow = 0; irow < dims[s % 2][2]; irow++)
                    {
                        c.set(irow +  dims[s % 2][0], icol +  dims[s % 2][1], c_tmp(irow + dims[s % 2][2] * icol, s % 2));
                    }
                }
                tstore += (omp_get_wtime() - t);
            }

            dims[s % 2][0] = row0;
            dims[s % 2][1] = col0;
            dims[s % 2][2] = nrow;
            dims[s % 2][3] = ncol;
            
            double t = omp_get_wtime();
            linalg<device_t::CPU>::gemm(2, 0, nrow, ncol, spl_K.local_size(),
                              a.at<CPU>(0, row0), a.ld(), b.at<CPU>(0, col0), b.ld(),
                              c_tmp.at<CPU>(0, s % 2), nrow);
            tcomp += (omp_get_wtime() - t);

            mpi_comm_world().iallreduce(c_tmp.at<CPU>(0, s % 2), nrow * ncol, &req[s % 2]);
            
            s++;
        }
    }

    for (int s: {0, 1})
    {
        if (req[s % 2] != MPI_REQUEST_NULL)
        {
            double t = omp_get_wtime();
            MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);
            tcomm += (omp_get_wtime() - t);

            t = omp_get_wtime();
            #pragma omp parallel for
            for (int icol = 0; icol < dims[s % 2][3]; icol++)
            {
                for (int irow = 0; irow < dims[s % 2][2]; irow++)
                {
                    c.set(irow +  dims[s % 2][0], icol +  dims[s % 2][1], c_tmp(irow + dims[s % 2][2] * icol, s % 2));
                }
            }
            tstore += (omp_get_wtime() - t);
        }
    }
    mpi_comm_world().barrier();
    t0 += omp_get_wtime();

    double perf = 8e-9 * M * N * K / t0 / mpi_comm_world().size();

    if (mpi_comm_world().rank() == 0)
    {
        printf("  gemm time: %f sec.\n", tcomp);
        printf("  comm time: %f sec.\n", tcomm);
        printf(" store time: %f sec.\n", tstore);
        printf("performance: %f Gflops / rank\n", perf);
    }

    for (int i = 0; i < c.num_cols_local(); i++)
    {
        for (int j = 0; j < c.num_rows_local(); j++)
        {
            if (std::abs(c(j, i) - 0.01 * K) > 1e-10) TERMINATE("result is wrong");
        }
    }

    return perf;
}

double wf_inner_overlap_allreduce_omp(int M, int N, int K, std::vector<int> mpi_grid, int BS)
{
    if (mpi_comm_world().rank() == 0)
    {
        printf("=== wf_inner_overlap_allreduce_omp ===\n");
    }

    COMMON_CODE_BLOCK

    mdarray<double_complex, 2> c_tmp(BS * BS, 2);
    c_tmp.zero();

    int nbr = M / BS + std::min(1, M % BS);
    int nbc = N / BS + std::min(1, N % BS);

   /* state of the buffers:
    * state = 0: buffer is free
    * state = 1: buffer stores result of local zgemm */
    int buf_state[] = {0, 0};
    
    omp_set_nested(1);
    int nt = omp_get_max_threads();
    if (nt < 2) TERMINATE("minimum two threads are required");

    mpi_comm_world().barrier();
    double t0 = -omp_get_wtime();
    double tcomp{0}, tcomm{0}, tstore{0};

    #pragma omp parallel num_threads(2) shared(buf_state)
    {
        if (omp_get_thread_num() == 1)
        {
            int s = 0;
            #ifdef __MKL
            mkl_set_num_threads_local(nt - 1);
            #endif
            omp_set_num_threads(nt - 1);

            for (int ibc = 0; ibc < nbc; ibc++)
            {
                int col0 = ibc * BS;
                int ncol = std::min(N, (ibc + 1) * BS) - col0;

                for (int ibr = 0; ibr < nbr; ibr++)
                {
                    int row0 = ibr * BS;
                    int nrow = std::min(M, (ibr + 1) * BS) - row0;
                    
                    int st = 1;
                    /* wait for the release of the buffer */
                    while (st)
                    {
                        //#pragma omp flush
                        #pragma omp atomic read
                        st = buf_state[s % 2];
                    }
                    
                    double t = omp_get_wtime();
                    linalg<device_t::CPU>::gemm(2, 0, nrow, ncol, spl_K.local_size(),
                                      a.at<CPU>(0, row0), a.ld(), b.at<CPU>(0, col0), b.ld(),
                                      c_tmp.at<CPU>(0, s % 2), nrow);
                    tcomp += (omp_get_wtime() - t);

                    #pragma omp atomic write
                    /* lock the buffer */
                    buf_state[s % 2] = 1;
                    //#pragma omp flush

                    s++;

                }
            }
        }
        else // thread#0
        {
            int s = 0;

            for (int ibc = 0; ibc < nbc; ibc++)
            {
                int col0 = ibc * BS;
                int ncol = std::min(N, (ibc + 1) * BS) - col0;

                for (int ibr = 0; ibr < nbr; ibr++)
                {
                    int row0 = ibr * BS;
                    int nrow = std::min(M, (ibr + 1) * BS) - row0;

                    int st = 0;
                    /* wait for the lock of the buffer */
                    while (!st)
                    {
                        //#pragma omp flush
                        #pragma omp atomic read
                        st = buf_state[s % 2];
                    }

                    double t = omp_get_wtime();
                    mpi_comm_world().allreduce(c_tmp.at<CPU>(0, s % 2), nrow * ncol);
                    tcomm += (omp_get_wtime() - t);

                    t = omp_get_wtime();
                    for (int icol = 0; icol < ncol; icol++)
                    {
                        for (int irow = 0; irow < nrow; irow++)
                        {
                            c.set(irow + row0, icol + col0, c_tmp(irow + nrow * icol, s % 2));
                        }
                    }
                    tstore += (omp_get_wtime() - t);

                    #pragma omp atomic write
                    /* release the buffer */
                    buf_state[s % 2] = 0;
                    //#pragma omp flush

                    s++;
                }
            }
        }
    }

    mpi_comm_world().barrier();
    t0 += omp_get_wtime();

    omp_set_nested(0);
    #ifdef __MKL
    mkl_set_num_threads_local(0);
    #endif
    omp_set_num_threads(nt);

    double perf = 8e-9 * M * N * K / t0 / mpi_comm_world().size();

    if (mpi_comm_world().rank() == 0)
    {
        printf("  gemm time: %f sec.\n", tcomp);
        printf("  comm time: %f sec.\n", tcomm);
        printf(" store time: %f sec.\n", tstore);
        printf("performance: %f Gflops / rank\n", perf);
    }

    for (int i = 0; i < c.num_cols_local(); i++)
    {
        for (int j = 0; j < c.num_rows_local(); j++)
        {
            if (std::abs(c(j, i) - 0.01 * K) > 1e-10) TERMINATE("result is wrong");
        }
    }

    return perf;
}

double wf_inner_overlap_allreduce_async_omp(int M, int N, int K, std::vector<int> mpi_grid, int BS)
{
    if (mpi_comm_world().rank() == 0)
    {
        printf("=== wf_inner_overlap_allreduce_async_omp ===\n");
    }

    COMMON_CODE_BLOCK

    mdarray<double_complex, 2> c_tmp(BS * BS, 2);
    c_tmp.zero();

    int nbr = M / BS + std::min(1, M % BS);
    int nbc = N / BS + std::min(1, N % BS);

   /* state of the buffers:
    * state = 0: buffer is free
    * state = 1: buffer stores result of local zgemm */
    int buf_state[] = {0, 0};

    std::array<MPI_Request, 2> req = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    std::array<std::array<int, 4>, 2> dims;
    
    omp_set_nested(1);
    int nt = omp_get_max_threads();
    if (nt < 2) TERMINATE("minimum two threads are required");
    double t1, t2;

    mpi_comm_world().barrier();
    double t0 = -omp_get_wtime();

    #pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 0)
        {
            t1 = -omp_get_wtime();
            int s = 0;
            omp_set_num_threads(nt - 1);

            for (int ibc = 0; ibc < nbc; ibc++)
            {
                int col0 = ibc * BS;
                int ncol = std::min(N, (ibc + 1) * BS) - col0;

                for (int ibr = 0; ibr < nbr; ibr++)
                {
                    int row0 = ibr * BS;
                    int nrow = std::min(M, (ibr + 1) * BS) - row0;

                    /* wait for the release of the buffer */
                    while (buf_state[s % 2])
                    {
                        #pragma omp flush(buf_state)
                    }

                    dims[s % 2][0] = row0;
                    dims[s % 2][1] = col0;
                    dims[s % 2][2] = nrow;
                    dims[s % 2][3] = ncol;

                    linalg<device_t::CPU>::gemm(2, 0, nrow, ncol, spl_K.local_size(),
                                      a.at<CPU>(0, row0), a.ld(), b.at<CPU>(0, col0), b.ld(),
                                      c_tmp.at<CPU>(0, s % 2), nrow);

                    mpi_comm_world().iallreduce(c_tmp.at<CPU>(0, s % 2), nrow * ncol, &req[s % 2]);

                    /* lock the buffer */
                    buf_state[s % 2] = 1;

                    s++;

                }
            }
            t1 += omp_get_wtime();
        }
        else // thread#1
        {
            t2 = -omp_get_wtime();

            for (int s = 0; s < nbc * nbr; s++)
            {
                /* wait for the lock of the buffer */
                while (!buf_state[s % 2])
                {
                    #pragma omp flush(buf_state)
                }

                MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);

                for (int icol = 0; icol < dims[s % 2][3]; icol++)
                {
                    for (int irow = 0; irow < dims[s % 2][2]; irow++)
                    {
                        c.set(irow +  dims[s % 2][0], icol +  dims[s % 2][1], c_tmp(irow + dims[s % 2][2] * icol, s % 2));
                    }
                }
                /* release the buffer */
                buf_state[s % 2] = 0;
            }

            t2 += omp_get_wtime();
        }
    }

    mpi_comm_world().barrier();
    t0 += omp_get_wtime();

    omp_set_nested(0);
    omp_set_num_threads(nt);

    double perf = 8e-9 * M * N * K / t0 / mpi_comm_world().size();

    if (mpi_comm_world().rank() == 0)
    {
        //printf("  gemm time: %f sec.\n", t1 - t0);
        //printf("  comm time: %f sec.\n", t2 - t1);
        //printf(" store time: %f sec.\n", t3 - t2);
        printf("performance: %f Gflops / rank\n", perf);
    }

    for (int i = 0; i < c.num_cols_local(); i++)
    {
        for (int j = 0; j < c.num_rows_local(); j++)
        {
            if (std::abs(c(j, i) - 0.01 * K) > 1e-10) TERMINATE("result is wrong");
        }
    }
    return perf;
}

double wf_inner_overlap_allreduce_pt(int M, int N, int K, std::vector<int> mpi_grid, int BS)
{
    if (mpi_comm_world().rank() == 0)
    {
        printf("=== wf_inner_overlap_allreduce_pt ===\n");
    }

    COMMON_CODE_BLOCK

    mdarray<double_complex, 2> c_tmp(BS * BS, 2);
    c_tmp.zero();

    int nbr = M / BS + std::min(1, M % BS);
    int nbc = N / BS + std::min(1, N % BS);

    std::atomic<bool> buf_lock[2];
    buf_lock[0].store(false);
    buf_lock[1].store(false);

    int nt = omp_get_max_threads();
    if (nt < 2) TERMINATE("minimum two threads are required");

    omp_set_num_threads(nt - 1);

    mpi_comm_world().barrier();
    double t0 = -omp_get_wtime();

    std::thread work_thread([nbr, nbc, M, N, BS, &buf_lock, &a, &b, &c_tmp, &spl_K]()
    {
        int s = 0;

        for (int ibc = 0; ibc < nbc; ibc++)
        {
            int col0 = ibc * BS;
            int ncol = std::min(N, (ibc + 1) * BS) - col0;

            for (int ibr = 0; ibr < nbr; ibr++)
            {
                int row0 = ibr * BS;
                int nrow = std::min(M, (ibr + 1) * BS) - row0;

                /* wait for the release of the buffer */
                while (buf_lock[s % 2].load());

                linalg<device_t::CPU>::gemm(2, 0, nrow, ncol, spl_K.local_size(),
                                  a.at<CPU>(0, row0), a.ld(), b.at<CPU>(0, col0), b.ld(),
                                  c_tmp.at<CPU>(0, s % 2), nrow);

                buf_lock[s % 2].store(true);

                s++;
            }
        }
    });

    std::thread comm_thread([nbr, nbc, M, N, BS, &buf_lock, &c_tmp, &c]()
    {
        int s = 0;

        for (int ibc = 0; ibc < nbc; ibc++)
        {
            int col0 = ibc * BS;
            int ncol = std::min(N, (ibc + 1) * BS) - col0;

            for (int ibr = 0; ibr < nbr; ibr++)
            {
                int row0 = ibr * BS;
                int nrow = std::min(M, (ibr + 1) * BS) - row0;
                
                /* wait for the release of the buffer */
                while (!buf_lock[s % 2].load());

                mpi_comm_world().allreduce(c_tmp.at<CPU>(0, s % 2), nrow * ncol);

                for (int icol = 0; icol < ncol; icol++)
                {
                    for (int irow = 0; irow < nrow; irow++)
                    {
                        c.set(irow + row0, icol + col0, c_tmp(irow + nrow * icol, s % 2));
                    }
                }
                /* release the buffer */
                buf_lock[s % 2].store(false);

                s++;
            }
        }
    });

    work_thread.join();
    comm_thread.join();

    omp_set_num_threads(nt);

    mpi_comm_world().barrier();
    t0 += omp_get_wtime();

    double perf = 8e-9 * M * N * K / t0 / mpi_comm_world().size();

    if (mpi_comm_world().rank() == 0)
    {
        //printf("  gemm time: %f sec.\n", t1 - t0);
        //printf("  comm time: %f sec.\n", t2 - t1);
        //printf(" store time: %f sec.\n", t3 - t2);
        printf("performance: %f Gflops / rank\n", perf);
    }

    for (int i = 0; i < c.num_cols_local(); i++)
    {
        for (int j = 0; j < c.num_rows_local(); j++)
        {
            if (std::abs(c(j, i) - 0.01 * K) > 1e-10) TERMINATE("result is wrong");
        }
    }
    return perf;
}

double wf_inner_overlap_allreduce_async_pt(int M, int N, int K, std::vector<int> mpi_grid, int BS)
{
    if (mpi_comm_world().rank() == 0)
    {
        printf("=== wf_inner_overlap_allreduce_async_pt ===\n");
    }

    COMMON_CODE_BLOCK

    mdarray<double_complex, 2> c_tmp(BS * BS, 2);
    c_tmp.zero();

    int nbr = M / BS + std::min(1, M % BS);
    int nbc = N / BS + std::min(1, N % BS);

    std::array<MPI_Request, 2> req = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    std::array<std::array<int, 4>, 2> dims;

    std::atomic<bool> buf_lock[2];
    buf_lock[0].store(false);
    buf_lock[1].store(false);

    int nt = omp_get_max_threads();
    if (nt < 2) TERMINATE("minimum two threads are required");

    omp_set_num_threads(nt - 1);

    mpi_comm_world().barrier();
    double t0 = -omp_get_wtime();

    std::thread work_thread([nbr, nbc, M, N, BS, &buf_lock, &a, &b, &c_tmp, &spl_K, &req, &dims]()
    {
        int s = 0;

        for (int ibc = 0; ibc < nbc; ibc++)
        {
            int col0 = ibc * BS;
            int ncol = std::min(N, (ibc + 1) * BS) - col0;

            for (int ibr = 0; ibr < nbr; ibr++)
            {
                int row0 = ibr * BS;
                int nrow = std::min(M, (ibr + 1) * BS) - row0;

                /* wait for the release of the buffer */
                while (buf_lock[s % 2].load());

                dims[s % 2][0] = row0;
                dims[s % 2][1] = col0;
                dims[s % 2][2] = nrow;
                dims[s % 2][3] = ncol;

                linalg<device_t::CPU>::gemm(2, 0, nrow, ncol, spl_K.local_size(),
                                  a.at<CPU>(0, row0), a.ld(), b.at<CPU>(0, col0), b.ld(),
                                  c_tmp.at<CPU>(0, s % 2), nrow);

                mpi_comm_world().iallreduce(c_tmp.at<CPU>(0, s % 2), nrow * ncol, &req[s % 2]);
                
                buf_lock[s % 2].store(true);

                s++;
            }
        }
    });

    std::thread comm_thread([nbr, nbc, M, N, BS, &buf_lock, &c_tmp, &c, &req, &dims]()
    {
        for (int s = 0; s < nbc * nbr; s++)
        {
            /* wait for the lock of the buffer */
            while (!buf_lock[s % 2].load());
            MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);

            for (int icol = 0; icol < dims[s % 2][3]; icol++)
            {
                for (int irow = 0; irow < dims[s % 2][2]; irow++)
                {
                    c.set(irow +  dims[s % 2][0], icol +  dims[s % 2][1], c_tmp(irow + dims[s % 2][2] * icol, s % 2));
                }
            }
            /* release the buffer */
            buf_lock[s % 2].store(false);
        }
    });

    comm_thread.join();
    work_thread.join();

    omp_set_num_threads(nt);

    //for (int s: {0, 1})
    //{
    //    if (req[s % 2] != MPI_REQUEST_NULL)
    //    {
    //        MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);

    //        #pragma omp parallel for
    //        for (int icol = 0; icol < dims[s % 2][3]; icol++)
    //        {
    //            for (int irow = 0; irow < dims[s % 2][2]; irow++)
    //            {
    //                c.set(irow +  dims[s % 2][0], icol +  dims[s % 2][1], c_tmp(irow + dims[s % 2][2] * icol, s % 2));
    //            }
    //        }
    //    }
    //}

    mpi_comm_world().barrier();
    t0 += omp_get_wtime();

    double perf = 8e-9 * M * N * K / t0 / mpi_comm_world().size();

    if (mpi_comm_world().rank() == 0)
    {
        //printf("  gemm time: %f sec.\n", t1 - t0);
        //printf("  comm time: %f sec.\n", t2 - t1);
        //printf(" store time: %f sec.\n", t3 - t2);
        printf("performance: %f Gflops / rank\n", perf);
    }

    for (int i = 0; i < c.num_cols_local(); i++)
    {
        for (int j = 0; j < c.num_rows_local(); j++)
        {
            if (std::abs(c(j, i) - 0.01 * K) > 1e-10) TERMINATE("result is wrong");
        }
    }
    return perf;
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--M=", "{int} M");
    args.register_key("--N=", "{int} N");
    args.register_key("--K=", "{int} K");
    args.register_key("--BS=", "{int} BS");
    args.register_key("--mpi_grid=", "{vector<int>} 2D MPI grid");
    args.register_key("--repeat=", "{int} repeat test number of times");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    int M = args.value<int>("M", 1000);
    int N = args.value<int>("N", 1000);
    int K = args.value<int>("K", 1000);
    int BS = args.value<int>("BS", 256);
    int repeat = args.value<int>("repeat", 2);
    std::vector<int> mpi_grid = args.value< std::vector<int> >("mpi_grid", {1, 1});

    sirius::initialize(true);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
    }

    test_nested_omp(BS);
    
    if (mpi_comm_world().rank() == 0)
    {
        printf("global matrix sizes: %i %i %i\n", M, N, K);
        printf("number of ranks: %i\n", mpi_comm_world().size());
        printf("\n");
    }
    
    Measurement perf1, perf2, perf3, perf4, perf5, perf6, perf7;
    
    for (int i = 0; i < repeat; i++)
    {
        perf1.push_back(wf_inner_simple(M, N, K, mpi_grid));
        perf2.push_back(wf_inner_reduce_to_one(M, N, K, mpi_grid));
        perf3.push_back(wf_inner_reduce_to_one_async(M, N, K, mpi_grid));
        perf4.push_back(wf_inner_allreduce(M, N, K, mpi_grid, BS));
        perf5.push_back(wf_inner_allreduce_async(M, N, K, mpi_grid, BS));
        perf6.push_back(wf_inner_overlap_allreduce_omp(M, N, K, mpi_grid, BS));
        //perf6.push_back(wf_inner_overlap_allreduce_async_omp(M, N, K, mpi_grid, BS));
        //perf6.push_back(wf_inner_overlap_allreduce_pt(M, N, K, mpi_grid, BS));
        //perf7.push_back(wf_inner_overlap_allreduce_async_pt(M, N, K, mpi_grid, BS));
    }

    if (mpi_comm_world().rank() == 0)
    {
        std::stringstream s;
        s << "output" << mpi_comm_world().size() << ".json";

        JSON_write jw(s.str());
        std::vector<double> v;

        jw.single("num_ranks", mpi_comm_world().size());
        jw.single("M", M);
        jw.single("N", N);
        jw.single("K", K);

        v = {perf1.average(), perf1.sigma()};
        jw.single("wf_inner_simple", v);
        
        v = {perf2.average(), perf2.sigma()};
        jw.single("wf_inner_reduce_to_one", v);
        
        v = {perf3.average(), perf3.sigma()};
        jw.single("wf_inner_reduce_to_one_async", v);
        
        v = {perf4.average(), perf4.sigma()};
        jw.single("wf_inner_allreduce", v);

        v = {perf5.average(), perf5.sigma()};
        jw.single("wf_inner_allreduce_async", v);
        
        v = {perf6.average(), perf6.sigma()};
        jw.single("wf_inner_overlap_allreduce_omp", v);

        printf("\n");
        printf("wf_inner_simple                : %12.6f GFlops / rank,  sigma: %12.6f\n", perf1.average(), perf1.sigma());
        printf("wf_inner_reduce_to_one         : %12.6f GFlops / rank,  sigma: %12.6f\n", perf2.average(), perf2.sigma());
        printf("wf_inner_reduce_to_one_async   : %12.6f GFlops / rank,  sigma: %12.6f\n", perf3.average(), perf3.sigma());
        printf("wf_inner_allreduce             : %12.6f GFlops / rank,  sigma: %12.6f\n", perf4.average(), perf4.sigma());
        printf("wf_inner_allreduce_async       : %12.6f GFlops / rank,  sigma: %12.6f\n", perf5.average(), perf5.sigma());
        printf("wf_inner_overlap_allreduce_omp : %12.6f GFlops / rank,  sigma: %12.6f\n", perf6.average(), perf6.sigma());
        //printf("wf_inner_overlap_allreduce_async_omp : %12.6f GFlops / rank,  sigma: %12.6f\n", perf6.average(), perf6.sigma());
        //printf("wf_inner_overlap_allreduce_pt  : %12.6f GFlops / rank,  sigma: %12.6f\n", perf6.average(), perf6.sigma());
        //printf("wf_inner_overlap_allreduce_async_pt : %12.6f GFlops / rank,  sigma: %12.6f\n", perf7.average(), perf7.sigma());
    }

    runtime::Timer::print();

    sirius::finalize();
}
