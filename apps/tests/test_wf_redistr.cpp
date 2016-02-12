#include <sirius.h>
#include <thread>

using namespace sirius;

void test_wf_redistr(double alat, double pw_cutoff, double wf_cutoff, int num_bands, std::vector<int>& mpi_grid)
{
    Communicator comm(MPI_COMM_WORLD);
    BLACS_grid blacs_grid(comm, mpi_grid[0], mpi_grid[1]);

    double a1[] = {alat, 0, 0};
    double a2[] = {0, alat, 0};
    double a3[] = {0, 0, alat};

    Simulation_parameters p;
    Unit_cell uc(p, comm);
    uc.set_lattice_vectors(a1, a2, a3);

    auto& rlv = uc.reciprocal_lattice_vectors();
    auto dims = Utils::find_translation_limits(pw_cutoff, rlv);

    MPI_FFT3D fft(dims, Platform::max_num_threads(), blacs_grid.comm_row());

    auto gv_rho = fft.init_gvec(vector3d<double>(0, 0, 0), pw_cutoff, rlv);
    auto gv_wf = fft.init_gvec(vector3d<double>(0, 0, 0), wf_cutoff, rlv);

    if (comm.rank() == 0)
    {
        printf("MPI grid: %i %i\n", mpi_grid[0], mpi_grid[1]);
        printf("FFT dimensions: %i %i %i\n", fft.size(0), fft.size(1), fft.size(2));
        printf("num_gvec_rho: %i\n", gv_rho.num_gvec_);
        printf("num_gvec_wf: %i\n", gv_wf.num_gvec_);
    }

    splindex<block> spl_gv_wf(gv_wf.num_gvec_, blacs_grid.num_ranks_row(), blacs_grid.rank_row());

    dmatrix<double_complex> psi(gv_wf.num_gvec_, num_bands, blacs_grid, (int)spl_gv_wf.block_size(), 1);
    psi.zero();

    for (int i = 0; i < num_bands; i++) 
    {
        for (int ig = 0; ig < gv_wf.num_gvec_; ig++)
        {
            psi.set(ig, i, double_complex(1, 0) / std::sqrt(double((gv_wf.num_gvec_))));
        }
    }

    if (psi.num_rows_local() != (int)spl_gv_wf.local_size())
    {
        TERMINATE("wrong index splitting");
    }

    auto h = psi.panel().hash();

    BLACS_grid blacs_grid_1d(comm, mpi_grid[0] * mpi_grid[1], 1);

    splindex<block> spl_gv_wf_1d(gv_wf.num_gvec_, blacs_grid_1d.num_ranks_row(), blacs_grid_1d.rank_row());

    dmatrix<double_complex> psi_slab(gv_wf.num_gvec_, num_bands, blacs_grid_1d, (int)spl_gv_wf_1d.block_size(), 1);

    Timer t2("swap_data");
    linalg<CPU>::gemr2d(gv_wf.num_gvec_, num_bands, psi, 0, 0, psi_slab, 0, 0, blacs_grid.context());
    double tval= t2.stop();


    if (comm.rank() == 0)
    {
        printf("time to swap %li Mb: %f sec.", (sizeof(double_complex) * gv_wf.num_gvec_ * num_bands) >> 20, tval);
    }
    
    psi.zero();
    linalg<CPU>::gemr2d(gv_wf.num_gvec_, num_bands, psi_slab, 0, 0, psi, 0, 0, blacs_grid.context());
    if (h != psi.panel().hash())
    {
        DUMP("wrong hash");
    }
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--alat=", "{double} lattice constant");
    args.register_key("--pw_cutoff=", "{double} plane-wave cutoff [a.u.^-1]");
    args.register_key("--wf_cutoff=", "{double} wave-function cutoff [a.u.^-1]");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--mpi_grid=", "{vector<int>} MPI grid");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    double alat = 7.0;
    double pw_cutoff = 16.0;
    double wf_cutoff = 5.0;
    int num_bands = 10;
    std::vector<int> mpi_grid;

    alat = args.value<double>("alat", alat);
    pw_cutoff = args.value<double>("pw_cutoff", pw_cutoff);
    wf_cutoff = args.value<double>("wf_cutoff", wf_cutoff);
    num_bands = args.value<int>("num_bands", num_bands);
    mpi_grid = args.value< std::vector<int> >("mpi_grid", std::vector<int>({1, 1}));


    Platform::initialize(1);

    test_wf_redistr(alat, pw_cutoff, wf_cutoff, num_bands, mpi_grid);

    //#ifdef __PRINT_MEMORY_USAGE
    //MEMORY_USAGE_INFO();
    //#endif
    Timer::print();

    Platform::finalize();
}
