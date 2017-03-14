#include <sirius.h>

using namespace sirius;

template <device_t ptr_pu>
void test_fft_complex(double cutoff__, device_t fft_pu__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    FFT3D fft(find_translations(cutoff__, M), mpi_comm_world(), fft_pu__);

    Gvec gvec(M, cutoff__, mpi_comm_world(), mpi_comm_world(), false);

    if (mpi_comm_world().rank() == 0) {
        printf("num_gvec: %i\n", gvec.num_gvec());
    }
    MPI_grid mpi_grid(mpi_comm_world());

    printf("num_gvec_fft: %i\n", gvec.partition().gvec_count_fft());
    printf("offset_gvec_fft: %i\n", gvec.partition().gvec_offset_fft());

    fft.prepare(gvec.partition());

    mdarray<double_complex, 1> f(gvec.partition().gvec_count_fft());
    for (int ig = 0; ig < gvec.partition().gvec_count_fft(); ig++) {
        f[ig] = type_wrapper<double_complex>::random();
    }
    mdarray<double_complex, 1> g(gvec.partition().gvec_count_fft());
    #ifdef __GPU
    if (ptr_pu == GPU) {
        f.allocate(memory_t::device);
        f.copy_to_device();
        g.allocate(memory_t::device);
    }
    #endif

    fft.transform<1, ptr_pu>(gvec.partition(), f.at<ptr_pu>());
    fft.transform<-1, ptr_pu>(gvec.partition(), g.at<ptr_pu>());

    #ifdef __GPU
    if (ptr_pu == GPU) {
        g.copy_to_host();
    }
    #endif

    double diff = 0;
    for (int ig = 0; ig < gvec.partition().gvec_count_fft(); ig++) {
        diff += std::pow(std::abs(f[ig] - g[ig]), 2);
    }
    mpi_comm_world().allreduce(&diff, 1);
    diff = std::sqrt(diff / gvec.num_gvec());
    if (mpi_comm_world().rank() == 0) {
        printf("error : %18.10e", diff);
        if (diff < 1e-10) {
            printf("  OK\n");
        } else {
            printf("  Fail\n");
            exit(1);
        }
    }

    fft.dismiss();
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    double cutoff = args.value<double>("cutoff", 40);

    sirius::initialize(1);

    test_fft_complex<CPU>(cutoff, CPU);
    #ifdef __GPU
    test_fft_complex<CPU>(cutoff, GPU);
    test_fft_complex<GPU>(cutoff, GPU);
    #endif

    sddk::timer::print();
    
    sirius::finalize();
    return 0;
}
