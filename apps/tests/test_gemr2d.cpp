#include <sirius.h>

using namespace sirius;

void test_gemr2d(int M, int N, int repeat)
{
    Communicator comm(MPI_COMM_WORLD);

    BLACS_grid grid_col(comm, 1, comm.size());
    BLACS_grid grid_row(comm, comm.size(), 1);

    int gcontext = grid_row.context();

    dmatrix<double_complex> A(M, N, grid_row, splindex_base::block_size(M, comm.size()), 1);
    int n = 0;
    for (int i = 0; i < N; i++)
    {
        //for (int j = 0; j < M; j++) A.set(j, i, type_wrapper<double_complex>::random());
        for (int j = 0; j < M; j++) A.set(j, i, double_complex(n++, 0));
    }
    auto h = A.panel().hash();

    dmatrix<double_complex> B(M, N, grid_col, 1, 1);
    B.zero();
    //redist::gemr2d(A, B);
    //dmatrix<double_complex> Anew(M, N, grid_row, splindex_base::block_size(M, comm.size()), 1);
    //redist::gemr2d(B, Anew);
    //for (int rank = 0; rank < comm.size(); rank++)
    //{
    //    if (rank == comm.rank())
    //    {
    //        printf("rank: %i\n", rank);
    //        for (int i = 0; i < A.num_cols_local(); i++)
    //        {
    //            for (int j = 0; j < A.num_rows_local(); j++)
    //            {
    //                double d = std::abs(A(j, i) - Anew(j, i));
    //                if (d > 1e-12)
    //                {
    //                    printf("diff(%4i, %4i): exp %12.6f, actual %12.6f\n", j, i, std::abs(A(j, i)), std::abs(Anew(j, i)));
    //                }
    //            }
    //        }
    //    }
    //    comm.barrier();
    //}

    ////for (int rank = 0; rank < comm.size(); rank++)
    ////{
    ////    if (rank == comm.rank())
    ////    {
    ////        printf("rank: %i\n", rank);
    ////        for (int i = 0; i < B.num_cols_local(); i++)
    ////        {
    ////            for (int j = 0; j < B.num_rows_local(); j++) printf("B(%i, %i)=%f\n", j, i, B(j, i).real());
    ////        }
    ////    }
    ////    comm.barrier();
    ////}
    redist::gemr2d(M, N - 1, A, 0, 1, B, 0, 0);
    redist::gemr2d(M, N - 1, B, 0, 0, A, 0, 1);
    if (A.panel().hash() != h)
    {
        TERMINATE("wrong hash");
    }




    for (int i = 0; i < repeat; i++)
    {
        double t1 = -Utils::current_time();

        redist::gemr2d(M, N, A, 0, 0, B, 0, 0);
        redist::gemr2d(M, N, B, 0, 0, A, 0, 0);
        t1 += Utils::current_time();
        if (comm.rank() == 0) std::cout << "custom swap time: " << t1 << " sec." << std::endl;
    }
    if (A.panel().hash() != h)
    {
        TERMINATE("wrong hash");
    }
    if (comm.rank() == 0) printf("Done.\n");
    
    double t0 = -Utils::current_time();
    for (int i = 0; i < repeat; i++)
    {
        linalg<device_t::CPU>::gemr2d(M, N, A, 0, 0, B, 0, 0, gcontext);
        linalg<device_t::CPU>::gemr2d(M, N, B, 0, 0, A, 0, 0, gcontext);
    }
    t0 += Utils::current_time();

    if (comm.rank() == 0)
    {
        printf("average time %.4f sec, swap speed: %.4f GB/sec\n", t0 / repeat,
               sizeof(double_complex) * 2 * repeat * M * N / double(1 << 30) / t0);
    }
    
    if (A.panel().hash() != h)
    {
        TERMINATE("wrong hash");
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--M=", "{int} number of rows");
    args.register_key("--N=", "{int} number of columns");
    args.register_key("--repeat=", "{int} number of repeats");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }
    int M = args.value<int>("M", 10000);
    int N = args.value<int>("N", 100);
    int repeat = args.value<int>("repeat", 10);

    Platform::initialize(1);
    
    test_gemr2d(M, N, repeat);

    Platform::finalize();
}
