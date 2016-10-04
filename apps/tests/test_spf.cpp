#include <sirius.h>

using namespace sirius;

void test_spf(std::vector<int> mpi_grid_dims__,
              double cutoff__,
              int use_gpu__)
{
    device_t pu = static_cast<device_t>(use_gpu__);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    MPI_grid mpi_grid(mpi_grid_dims__, mpi_comm_world());
    
    /* create FFT box */
    FFT3D_grid fft_box(2.01 * cutoff__, M);
    /* create FFT driver */
    FFT3D fft(fft_box, mpi_grid.communicator(1 << 1), pu);
    /* create G-vectors */
    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_box, mpi_comm_world().size(), fft.comm(), false);

    if (mpi_comm_world().rank() == 0) {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.gvec_count(0));
        printf("FFT grid size: %i %i %i\n", fft_box.size(0), fft_box.size(1), fft_box.size(2));
    }

    experimental::Smooth_periodic_function<double_complex> spf(fft, gvec, mpi_comm_world());
    
    mdarray<double_complex, 1> tmp(gvec.gvec_count(mpi_comm_world().rank()));

    for (int i = 0; i < gvec.gvec_count(mpi_comm_world().rank()); i++) {
        spf.f_pw_local(i) = tmp[i] = type_wrapper<double_complex>::random();
    }
    fft.prepare(gvec.partition());
    spf.fft_transform(1);
    spf.fft_transform(-1);
    fft.dismiss();

    for (int i = 0; i < gvec.gvec_count(mpi_comm_world().rank()); i++) {
        if (std::abs(spf.f_pw_local(i) - tmp[i]) > 1e-12) {
            std::stringstream s;
            s << "large difference: " << std::abs(spf.f_pw_local(i) - tmp[i]);
            TERMINATE(s);
        }
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value< std::vector<int> >("mpi_grid_dims", {1, 1});
    auto cutoff = args.value<double>("cutoff", 2.0);
    auto use_gpu = args.value<int>("use_gpu", 0);

    sirius::initialize(1);
    test_spf(mpi_grid_dims, cutoff, use_gpu);

    mpi_comm_world().barrier();
    runtime::Timer::print();
    runtime::Timer::print_all();
    sirius::finalize();
}
