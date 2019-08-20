#include <sirius.h>

/* test FFT: tranfrom random function to real space, transfrom back and compare with the original function */

using namespace sirius;

int test_fft_complex(cmd_args& args, device_t fft_pu__)
{
    double cutoff = args.value<double>("cutoff", 40);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    auto fft_grid = get_min_fft_grid(cutoff, M);

    auto spl_z = split_fft_z(fft_grid[2], Communicator::world());

    Gvec gvec(M, cutoff, Communicator::world(), false);

    Gvec_partition gvp(gvec, Communicator::world(), Communicator::self());

    spfft::Grid spfft_grid(fft_grid[0], fft_grid[1], fft_grid[2], gvp.zcol_count_fft(), spl_z.local_size(),
                           SPFFT_PU_HOST, -1, Communicator::world().mpi_comm(), SPFFT_EXCH_DEFAULT);

    const auto fft_type = gvec.reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

    auto gv = gvp.get_gvec();
    spfft::Transform spfft(spfft_grid.create_transform(SPFFT_PU_HOST, fft_type, fft_grid[0], fft_grid[1], fft_grid[2],
        spl_z.local_size(), gvp.gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
        gv.at(memory_t::host)));

    mdarray<double_complex, 1> f(gvp.gvec_count_fft());
    for (int ig = 0; ig < gvp.gvec_count_fft(); ig++) {
        f[ig] = utils::random<double_complex>();
    }
    mdarray<double_complex, 1> g(gvp.gvec_count_fft());

    spfft.backward(reinterpret_cast<double const*>(&f[0]), spfft.processing_unit());
    spfft.forward(spfft.processing_unit(), reinterpret_cast<double*>(&g[0]), SPFFT_FULL_SCALING);

    double diff{0};
    for (int ig = 0; ig < gvp.gvec_count_fft(); ig++) {
        diff += std::pow(std::abs(f[ig] - g[ig]), 2);
    }
    Communicator::world().allreduce(&diff, 1);
    diff = std::sqrt(diff / gvec.num_gvec());

    if (diff > 1e-10) {
        return 1;
    } else {
        return 0;
    }
}

int run_test(cmd_args& args)
{
    int result = test_fft_complex(args, device_t::CPU);
#ifdef __GPU
    result += test_fft_complex(args, device_t::GPU);
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
