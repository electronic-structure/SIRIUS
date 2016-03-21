#include <sirius.h>

using namespace sirius;

void test_wf_write(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__)
{
    //MPI_grid mpi_grid(mpi_grid_dims__, mpi_comm_world());

    std::vector<int> wf_mpi_grid = {1, mpi_comm_world().size()};

    MPI_grid mpi_grid(wf_mpi_grid, mpi_comm_world());

    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    FFT3D_grid fft_grid(cutoff__, M);

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_grid, mpi_grid.communicator(1 << 0),
              mpi_grid.dimension_size(1), false, false);
    
    printf("num_gvec: %i\n", gvec.num_gvec());
    
    Wave_functions<false> wf(num_bands__, num_bands__, gvec, mpi_grid, CPU);
    for (int i = 0; i < num_bands__; i++)
    {
        for (int ig = 0; ig < gvec.num_gvec(mpi_comm_world().rank()); ig++) wf(ig, i) = type_wrapper<double_complex>::random();
    }
    
    if (mpi_comm_world().rank() == 0)
    {
        sirius::HDF5_tree f("wf.h5", true);
        f.create_node("wf");
    }
    wf.swap_forward(0, num_bands__);

    for (int r = 0; r < mpi_comm_world().size(); r++)
    {
        if (r == mpi_comm_world().rank())
        {
            sirius::HDF5_tree f("wf.h5", false);
            std::vector<double_complex> single_wf(gvec.num_gvec());
            for (int i = 0; i < wf.spl_num_swapped().local_size(); i++)
            {
                int idx = wf.spl_num_swapped().global_index(i, r);
                std::memcpy(&single_wf[0], wf[i], gvec.num_gvec() * sizeof(double_complex));
                f["wf"].write(idx, single_wf);
            }
        }
        mpi_comm_world().barrier();
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--mpi_grid=", "{vector2d<int>} MPI grid");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    double cutoff = args.value<double>("cutoff", 10);
    int num_bands = args.value<int>("num_bands", 50);
    std::vector<int> mpi_grid = args.value< std::vector<int> >("mpi_grid", {1, 1});

    sirius::initialize(1);
    test_wf_write(mpi_grid, cutoff, num_bands);
    sirius::finalize();
}
