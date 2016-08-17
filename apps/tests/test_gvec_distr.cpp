#include <sirius.h>

using namespace sirius;

void test_gvec_distr(double cutoff__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    FFT3D_grid fft_box(2.01 * cutoff__, M);

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_box, mpi_comm_world().size(), false, false);

    MPI_grid mpi_grid(mpi_comm_world());
    Gvec_FFT_distribution gvec_fft_distr(gvec, mpi_grid);

    Gvec_FFT_distribution gvec_fft_distr1(gvec, mpi_comm_world());

    runtime::pstdout pout(mpi_comm_world());
    pout.printf("-----------------------\n");
    pout.printf("rank: %i\n", mpi_comm_world().rank());
    pout.printf("num_gvec : %i\n", gvec.num_gvec());
    pout.printf("num_gvec_loc : %i\n", gvec.num_gvec(mpi_comm_world().rank()));
    //pout.printf("num_zcols : %i\n", static_cast<int>(gvec.z_columns().size()));
    pout.printf("num_gvec_fft: %i\n", gvec_fft_distr.num_gvec_fft());
    pout.printf("offset_gvec_fft: %i\n", gvec_fft_distr.offset_gvec_fft());
    pout.printf("num_zcols_local : %i\n", gvec_fft_distr.zcol_fft_distr().counts[mpi_comm_world().rank()]);
}

void test_gvec(double cutoff__, bool reduce__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    FFT3D_grid fft_box(2.01 * cutoff__, M);

    experimental::Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_box, mpi_comm_world().size(), mpi_comm_world(), reduce__);

    for (int ig = 0; ig < gvec.num_gvec(); ig++) {
        auto G = gvec.gvec(ig);
        printf("ig: %i, G: %i %i %i\n", ig, G[0], G[1], G[2]);
        auto idx = gvec.index_by_gvec(G);
        if (idx != ig) {
            std::stringstream s;
            s << "wrong reverce index" << std::endl
              << "direct index: " << ig << std::endl
              << "reverce index: " << idx;
            TERMINATE(s);
        }
    }
}

int main(int argn, char** argv)
{
    cmd_args args;

    args.register_key("--cutoff=", "{double} wave-functions cutoff");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    auto cutoff = args.value<double>("cutoff", 2.0);

    sirius::initialize(1);
    test_gvec_distr(cutoff);
    test_gvec(cutoff, false);
    test_gvec(cutoff, true);
    sirius::finalize();
}
