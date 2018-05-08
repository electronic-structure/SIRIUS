#include <sirius.h>

using namespace sirius;

void test_fft(double cutoff__, device_t pu__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    FFT3D fft(find_translations(cutoff__, M), mpi_comm_world(), pu__);

    Gvec gvec(M, cutoff__, mpi_comm_world(), false);

    Gvec_partition gvp(gvec, mpi_comm_world(), mpi_comm_self());

    if (mpi_comm_world().rank() == 0) {
        printf("num_gvec: %i\n", gvec.num_gvec());
    }
    MPI_grid mpi_grid(mpi_comm_world());

    printf("num_gvec_fft: %i\n", gvp.gvec_count_fft());
    //printf("offset_gvec_fft: %i\n", gvec.partition().gvec_offset_fft());

    fft.prepare(gvp);

    mdarray<double_complex, 1> f(gvec.num_gvec());
    if (pu__ == GPU) {
        f.allocate(memory_t::device);
    }

    for (int ig = 0; ig < gvec.num_gvec(); ig++) {
        auto v = gvec.gvec_cart(ig);
        f[ig] = 1.0 / std::pow(v.length() + 1, 2);
    }
    print_hash("f(G)", f.hash());

    switch (pu__) {
        case CPU: {
            fft.transform<1>(&f[gvec.partition().gvec_offset_fft()]);
            break;
        }
        case GPU: {
            f.copy<memory_t::host, memory_t::device>();
            //fft.transform<1, GPU>(gvec.partition(), f.at<GPU>(gvec.partition().gvec_offset_fft()));
            fft.transform<1, CPU>(f.at<CPU>(gvec.partition().gvec_offset_fft()));
            fft.buffer().copy<memory_t::device, memory_t::host>();
            break;
        }
    }
    print_hash("f(r)", fft.buffer().hash());

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

    test_fft(cutoff, CPU);
    #ifdef __GPU
    //test_fft(cutoff, GPU);
    #endif

    sddk::timer::print();
    
    sirius::finalize();
    return 0;
}
