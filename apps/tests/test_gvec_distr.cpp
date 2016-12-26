#include <sirius.h>

using namespace sirius;

void test_gvec_distr(double cutoff__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    //FFT3D_grid fft_box(2.01 * cutoff__, M);
    FFT3D_grid fft_box(cutoff__, M);

    Gvec gvec;

    gvec = Gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_box, mpi_comm_world().size(), mpi_comm_world(), false);

    MPI_grid mpi_grid(mpi_comm_world());

    runtime::pstdout pout(mpi_comm_world());
    pout.printf("-----------------------\n");
    pout.printf("rank: %i\n", mpi_comm_world().rank());
    pout.printf("FFT box size: %i %i %i\n", fft_box.size(0), fft_box.size(1), fft_box.size(2));
    pout.printf("num_gvec : %i\n", gvec.num_gvec());
    pout.printf("num_gvec_loc : %i\n", gvec.gvec_count(mpi_comm_world().rank()));
    pout.printf("num_zcols : %i\n", gvec.num_zcol());
    pout.printf("num_gvec_fft: %i\n", gvec.partition().gvec_count_fft());
    pout.printf("offset_gvec_fft: %i\n", gvec.partition().gvec_offset_fft());
    pout.printf("num_zcols_local : %i\n", gvec.partition().zcol_distr_fft().counts[mpi_comm_world().rank()]);

    auto& gvp = gvec.partition();

    //for (int i = 0; i < gvp.num_zcol(); i++) {
    //    for (size_t j = 0; j < gvp.zcol(i).z.size(); j++) {
    //        printf("icol: %i idx: %li z: %i\n", i, j, gvp.zcol(i).z[j]);
    //    }
    //}
}

void test_gvec(double cutoff__, bool reduce__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    //FFT3D_grid fft_box(2.01 * cutoff__, M);
    FFT3D_grid fft_box(cutoff__, M);

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_box, mpi_comm_world().size(), mpi_comm_world(), reduce__);

    for (int ig = 0; ig < gvec.num_gvec(); ig++) {
        auto G = gvec.gvec(ig);
        //printf("ig: %i, G: %i %i %i\n", ig, G[0], G[1], G[2]);
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
