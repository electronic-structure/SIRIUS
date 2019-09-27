#include <sirius.h>

/* test FFT: transform single harmonic and compare with plane wave exp(iGr) */

using namespace sirius;

int test_fft(cmd_args& args, device_t pu__)
{
    bool verbose{true};

    double cutoff = args.value<double>("cutoff", 8);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    auto fft_grid = get_min_fft_grid(cutoff, M);

    auto spl_z = split_fft_z(fft_grid[2], Communicator::world());

    Gvec gvec(M, cutoff, Communicator::world(), false);

    Gvec_partition gvp(gvec, Communicator::world(), Communicator::self());

    auto spfft_pu = (pu__ == device_t::CPU) ? SPFFT_PU_HOST : SPFFT_PU_GPU;

    spfft::Grid spfft_grid(fft_grid[0], fft_grid[1], fft_grid[2], gvp.zcol_count_fft(), spl_z.local_size(),
                           spfft_pu, -1, Communicator::world().mpi_comm(), SPFFT_EXCH_DEFAULT);

    const auto fft_type = SPFFT_TRANS_C2C;

    auto gv = gvp.get_gvec();
    spfft::Transform spfft(spfft_grid.create_transform(spfft_pu, fft_type, fft_grid[0], fft_grid[1], fft_grid[2],
        spl_z.local_size(), gvp.gvec_count_fft(), SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host)));

    mdarray<double_complex, 1> f(gvec.num_gvec());
    if (pu__ == device_t::GPU) {
        f.allocate(memory_t::device);
    }
    mdarray<double_complex, 1> ftmp(gvp.gvec_count_fft());

    int result{0};

    if (Communicator::world().rank() == 0 && verbose) {
        std::cout << "Number of G-vectors: " << gvec.num_gvec() << "\n"; 
        std::cout << "FFT grid: " << spfft.dim_x() << " " << spfft.dim_y() << " " << spfft.dim_z() << "\n";
    }

    for (int ig = 0; ig < gvec.num_gvec(); ig++) {
        auto v = gvec.gvec(ig);
        if (Communicator::world().rank() == 0 && verbose) {
            printf("ig: %6i, gvec: %4i %4i %4i   ", ig, v[0], v[1], v[2]);
        }
        f.zero();
        f[ig] = 1.0;
        /* load local set of PW coefficients */
        for (int igloc = 0; igloc < gvp.gvec_count_fft(); igloc++) {
            ftmp[igloc] = f[gvp.idx_gvec(igloc)];
        }
        spfft.backward(reinterpret_cast<double const*>(&ftmp[0]), SPFFT_PU_HOST);

        auto ptr = reinterpret_cast<double_complex*>(spfft.space_domain_data(SPFFT_PU_HOST));

        double diff = 0;
        /* loop over 3D array (real space) */
        for (int j0 = 0; j0 < fft_grid[0]; j0++) {
            for (int j1 = 0; j1 < fft_grid[1]; j1++) {
                for (int j2 = 0; j2 < spfft.local_z_length(); j2++) {
                    /* get real space fractional coordinate */
                    auto rl = vector3d<double>(double(j0) / fft_grid[0],
                                               double(j1) / fft_grid[1],
                                               double(spfft.local_z_offset() + j2) / fft_grid[2]);
                    int idx = fft_grid.index_by_coord(j0, j1, j2);

                    /* compare value with the exponent */
                    diff += std::pow(std::abs(ptr[idx] - std::exp(double_complex(0.0, twopi * dot(rl, v)))), 2);
                }
            }
        }
        Communicator::world().allreduce(&diff, 1);
        diff = std::sqrt(diff / fft_grid.num_points());
        if (diff > 1e-10) {
            result++;
        }
        if (verbose) {
            if (diff > 1e-10) {
                printf("Fail\n");
            } else {
                printf("OK\n");
            }
        }
    }

    return result;
}

int run_test(cmd_args& args)
{
    int result = test_fft(args, device_t::CPU);
#ifdef __GPU
    result += test_fft(args, device_t::GPU);
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
