#include <sirius.h>

using namespace sirius;

void test(cmd_args& args)
{
    std::vector<int> mpi_grid_dims;
    mpi_grid_dims = args.value< std::vector<int> >("mpi_grid", mpi_grid_dims);

    linalg<scalapack>::set_cyclic_block_size(32);

    MPI_grid mpi_grid(mpi_grid_dims, Platform::comm_world());
    
    BLACS_grid blacs_grid(mpi_grid.communicator(), mpi_grid.dimension_size(0), mpi_grid.dimension_size(1));
    
    generalized_evp_elpa2 evp(blacs_grid);

    HDF5_tree h_in("1_h.h5", false);
    HDF5_tree o_in("2_o.h5", false);

    int nrow, ncol;

    h_in.read("nrow", &nrow);
    h_in.read("ncol", &ncol);

    mdarray<double_complex, 2> h(nrow, ncol);
    mdarray<double_complex, 2> o(nrow, ncol);

    h_in.read_mdarray("matrix", h);
    o_in.read_mdarray("matrix", o);

    dmatrix<double_complex> h1(nrow, ncol, blacs_grid);
    dmatrix<double_complex> o1(nrow, ncol, blacs_grid);

    int num_bands = 1234;
    std::vector<double> eval(num_bands);
    dmatrix<double_complex> z1(nrow, num_bands, blacs_grid);
    z1.zero();
    
    for (int k = 0; k < 10; k++)
    {
        for (int i = 0; i < ncol; i++)
        {
            for (int j = 0; j < ncol; j++) 
            {
                h1.set(i, j, h(i, j));
                o1.set(i, j, o(i, j));
            }
        }

        Timer t("solve_evp");
        evp.solve(nrow, h1.num_rows_local(), h1.num_cols_local(), num_bands, 
                  h1.ptr(), h1.ld(), o1.ptr(), o1.ld(), &eval[0], z1.ptr(), z1.ld());
        t.stop();
    }
    double tval = Timer::value("solve_evp");
    if (mpi_grid.communicator().rank() == 0)
    {
        printf("mpi gird: %i %i\n", mpi_grid_dims[0], mpi_grid_dims[1]);
        printf("matrix size: %i\n", nrow);
        printf("average time on 10 runs: %f\n", tval / 10.0);
    }
}

int main(int argn, char** argv)
{
    Platform::initialize(1);

    cmd_args args;
    args.register_key("--mpi_grid=", "{vector int} MPI grid dimensions");
    args.parse_args(argn, argv);

    if (argn == 1)
    {
        printf("Usage: ./dft_loop [options] \n");
        args.print_help();
        exit(0);
    }
    
    test(args);

    Platform::finalize();
}
