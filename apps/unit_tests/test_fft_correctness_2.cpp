#include <sirius.h>

/* test FFT: tranfrom random function to real space, transfrom back and compare with the original function */

using namespace sirius;

int test_fft_complex(cmd_args& args, device_t fft_pu__)
{
    double cutoff = args.value<double>("cutoff", 40);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    FFT3D fft(find_translations(cutoff, M), Communicator::world(), fft_pu__);

    Gvec gvec(M, cutoff, Communicator::world(), false);

    Gvec_partition gvp(gvec, fft.comm(), Communicator::self());

    fft.prepare(gvp);

    mdarray<double_complex, 1> f(gvp.gvec_count_fft());
    for (int ig = 0; ig < gvp.gvec_count_fft(); ig++) {
        f[ig] = utils::random<double_complex>();
    }
    mdarray<double_complex, 1> g(gvp.gvec_count_fft());

    fft.transform<1>(f.at(memory_t::host));
    fft.transform<-1>(g.at(memory_t::host));

    double diff{0};
    for (int ig = 0; ig < gvp.gvec_count_fft(); ig++) {
        diff += std::pow(std::abs(f[ig] - g[ig]), 2);
    }
    Communicator::world().allreduce(&diff, 1);
    diff = std::sqrt(diff / gvec.num_gvec());

    fft.dismiss();

    if (diff > 1e-10) {
        return 1;
    } else {
        return 0;
    }
}

int run_test(cmd_args& args)
{
    int result = test_fft_complex(args, CPU);
#ifdef __GPU
    result += test_fft_complex(args, GPU);
#endif
    return result;
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

    sirius::initialize(true);
    printf("running %-30s : ", argv[0]);
    int result = run_test(args);
    if (result) {
        printf("\x1b[31m" "Failed" "\x1b[0m" "\n");
    } else {
        printf("\x1b[32m" "OK" "\x1b[0m" "\n");
    }
    sirius::finalize();

    return result;
}
