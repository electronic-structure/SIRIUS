#include <sirius.h>

/* test FFT: transform single harmonic and compare with plane wave exp(iGr) */

using namespace sirius;

int test_fft(cmd_args& args, device_t pu__)
{
    double cutoff = args.value<double>("cutoff", 10);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    FFT3D fft(find_translations(cutoff, M), Communicator::world(), pu__);

    auto spl_z = split_fft_z(fft.size(2), Communicator::world());

    Gvec gvec(M, cutoff, Communicator::world(), false);
    Gvec_partition gvecp(gvec, Communicator::world(), Communicator::self());

    spfft::Grid spfft_grid(fft.size(0), fft.size(1), fft.size(2), gvecp.zcol_count_fft(), spl_z.local_size(),
                           SPFFT_PU_HOST, -1, fft.comm().mpi_comm(), SPFFT_EXCH_DEFAULT);

    const auto fft_type = gvec.reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

    spfft::Transform spfft(spfft_grid.create_transform(SPFFT_PU_HOST, fft_type, fft.size(0), fft.size(1), fft.size(2),
        spl_z.local_size(), gvecp.gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
        gvecp.gvec_coord().at(memory_t::host)));

    mdarray<double_complex, 1> f(gvec.num_gvec());
    if (pu__ == device_t::GPU) {
        f.allocate(memory_t::device);
    }
    mdarray<double_complex, 1> ftmp(gvecp.gvec_count_fft());

    int result{0};

    for (int ig = 0; ig < gvec.num_gvec(); ig++) {
        auto v = gvec.gvec(ig);
        //if (Communicator::world().rank() == 0) {
        //    printf("ig: %6i, gvec: %4i %4i %4i   ", ig, v[0], v[1], v[2]);
        //}
        f.zero();
        f[ig] = 1.0;
        /* load local set of PW coefficients */
        for (int igloc = 0; igloc < gvecp.gvec_count_fft(); igloc++) {
            ftmp[igloc] = f[gvecp.idx_gvec(igloc)];
        }
        switch (pu__) {
            case device_t::CPU: {
                //fft.transform<1>(&ftmp[0]);
                spfft.backward(reinterpret_cast<double const*>(&ftmp[0]), spfft.processing_unit());
                break;
            }
            case device_t::GPU: {
                //f.copy<memory_t::host, memory_t::device>();
                //fft.transform<1, GPU>(gvec.partition(), f.at<GPU>(gvec.partition().gvec_offset_fft()));
                //fft.transform<1, memory_t::host>(ftmp.at(memory_t::host));
                //fft.buffer().copy_to(memory_t::host);
                break;
            }
        }

        auto ptr = reinterpret_cast<double_complex*>(spfft.space_domain_data(SPFFT_PU_HOST));

        double diff = 0;
        /* loop over 3D array (real space) */
        for (int j0 = 0; j0 < fft.size(0); j0++) {
            for (int j1 = 0; j1 < fft.size(1); j1++) {
                for (int j2 = 0; j2 < spfft.local_z_length(); j2++) {
                    /* get real space fractional coordinate */
                    auto rl = vector3d<double>(double(j0) / fft.size(0), 
                                               double(j1) / fft.size(1), 
                                               double(spfft.local_z_offset() + j2) / fft.size(2));
                    int idx = fft.index_by_coord(j0, j1, j2);

                    /* compare value with the exponent */
                    diff += std::pow(std::abs(ptr[idx] - std::exp(double_complex(0.0, twopi * dot(rl, v)))), 2);
                }
            }
        }
        Communicator::world().allreduce(&diff, 1);
        diff = std::sqrt(diff / fft.size());
        if (diff > 1e-10) {
            result++;
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
