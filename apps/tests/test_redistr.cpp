#include <sirius.h>

using namespace sirius;

void test_redistr(std::vector<int> mpi_grid_dims, int M, int N)
{
    if (mpi_grid_dims.size() != 2) {
        TERMINATE("2d MPI grid is expected");
    }

    MPI_Win win;

    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid_dims[0], mpi_grid_dims[1]);

    dmatrix<double> mtrx(M, N, blacs_grid, 16, 16);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mtrx.set(j, i, double((j + 1) * (i + 1)));
        }
    }

    splindex<block> spl_row(M, mpi_comm_world().size(), mpi_comm_world().rank());
    matrix<double> mtrx2(spl_row.local_size(), N);

    MPI_Win_create(mtrx2.at<CPU>(), mtrx2.size(), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    MPI_Win_fence(0, win);

    runtime::Timer t1("MPI_Put");

    for (int icol = 0; icol < mtrx.num_cols_local(); icol++) {
        int icol_glob = mtrx.icol(icol);
        for (int irow = 0; irow < mtrx.num_rows_local(); irow++) {
            int irow_glob = mtrx.irow(irow);

            auto location = spl_row.location(irow_glob);
            MPI_Put(&mtrx(irow, icol), 1, mpi_type_wrapper<double>::kind(), location.second, icol_glob * spl_row.local_size(location.second) + location.first , 1, mpi_type_wrapper<double>::kind(), win);
        }
    }

    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    double tval = t1.stop();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < spl_row.local_size(); j++) {
            int jglob = spl_row[j];
            if (std::abs(mtrx2(j, i) - double((jglob + 1) * (i + 1))) > 1e-14) {
                TERMINATE("error");
            }
            //pout.printf("%4i ", mtrx2(j, i));
        }
        //pout.printf("\n");
    }

    printf("time: %f\n", tval);

}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--M=", "{int} global number of matrix rows");
    args.register_key("--N=", "{int} global number of matrix columns");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    auto mpi_grid_dims = args.value< std::vector<int> >("mpi_grid_dims", {1, 1});
    auto M = args.value<int>("M", 10000);
    auto N = args.value<int>("N", 1000);

    sirius::initialize(1);
    test_redistr(mpi_grid_dims, M, N);
    sirius::finalize();
}
