#include <sirius.hpp>

using namespace sirius;

void test_wf_inner(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__, int bs__,
                   sddk::memory_t mem__)
{
    spla::Context spla_ctx(sddk::is_host_memory(mem__) ? SPLA_PU_HOST : SPLA_PU_GPU);

    std::unique_ptr<sddk::BLACS_grid> blacs_grid;
    if (mpi_grid_dims__[0] * mpi_grid_dims__[1] == 1) {
        blacs_grid = std::unique_ptr<sddk::BLACS_grid>(new sddk::BLACS_grid(mpi::Communicator::self(), 1, 1));
    } else {
        blacs_grid = std::unique_ptr<sddk::BLACS_grid>(new sddk::BLACS_grid(mpi::Communicator::world(), mpi_grid_dims__[0], mpi_grid_dims__[1]));
    }

    /* create G-vectors */
    auto gvec = fft::gkvec_factory(cutoff__, mpi::Communicator::world());

    if (mpi::Communicator::world().rank() == 0) {
        printf("number of bands          : %i\n", num_bands__);
        printf("total number of G-vectors: %i\n", gvec->num_gvec());
        printf("local number of G-vectors: %i\n", gvec->count());
    }

    wf::Wave_functions<double> phi1(gvec, wf::num_mag_dims(3), wf::num_bands(num_bands__), sddk::memory_t::host);
    wf::Wave_functions<double> phi2(gvec, wf::num_mag_dims(3), wf::num_bands(num_bands__), sddk::memory_t::host);

    auto sr = wf::spin_range(0, 2);

    for (auto s = sr.begin(); s != sr.end(); s++) {
        for (int i = 0; i < num_bands__; i++) {
            for (int igloc = 0; igloc < gvec->count(); igloc++) {
                int ig = igloc + gvec->offset();
                phi1.pw_coeffs(igloc, s, wf::band_index(i)) =
                    static_cast<double>(i + 1) / (ig + 1);
                phi2.pw_coeffs(igloc, s, wf::band_index(i)) =
                    static_cast<double>(ig + 1) / (i + 1) / gvec->num_gvec();
            }
        }
    }

    auto mg1 = phi1.memory_guard(mem__, wf::copy_to::device);
    auto mg2 = phi2.memory_guard(mem__, wf::copy_to::device);

    sddk::dmatrix<std::complex<double>> ovlp(num_bands__, num_bands__, *blacs_grid, bs__, bs__);

    /* warmup call */
    wf::inner(spla_ctx, mem__, sr, phi1, wf::band_range(0, num_bands__), phi2, wf::band_range(0, num_bands__), ovlp, 0, 0);
    mpi::Communicator::world().barrier();

    double t = -utils::wtime();
    wf::inner(spla_ctx, mem__, sr, phi1, wf::band_range(0, num_bands__), phi2, wf::band_range(0, num_bands__), ovlp, 0, 0);
    mpi::Communicator::world().barrier();
    t += utils::wtime();

    double perf = sr.size() * 8e-9 * num_bands__ * num_bands__ *  gvec->num_gvec() / t;
    if (mpi::Communicator::world().rank() == 0) {
        printf("execution time (sec) : %12.6f\n", t);
        printf("performance (GFlops) : %12.6f\n", perf);
    }

    double max_diff{0};
    for (int j = 0; j < ovlp.num_cols_local(); j++) {
        auto jcol = ovlp.icol(j);
        for (int i = 0; i < ovlp.num_rows_local(); i++) {
            auto irow = ovlp.irow(i);
            /* 2 is accumulated from two spins */
            std::complex<double> z = ovlp(i, j) - 2 * static_cast<double>(irow + 1) / (jcol + 1);
            max_diff = std::max(max_diff, std::abs(z));
        }
    }
    mpi::Communicator::world().reduce<double, mpi::op_t::max>(&max_diff, 1, 0);
    if (mpi::Communicator::world().rank() == 0) {
        printf("maximum difference: %18.12f\n", max_diff);
        if (max_diff > 1e-10) {
            printf("\x1b[31m" "Fail\n" "\x1b[0m" "\n");
        } else {
            printf("\x1b[32m" "OK\n" "\x1b[0m" "\n");
        }
    }


    //for (auto s = sr.begin(); s != sr.end(); s++) {
    //    for (int i = 0; i < num_bands__; i++) {
    //        for (int igloc = 0; igloc < gvec->count(); igloc++) {
    //            phi1.pw_coeffs(sddk::memory_t::host, igloc, s, wf::band_index(i)) = utils::random<std::complex<double>>();
    //        }
    //    }
    //}
    //orthogonalize(spla_ctx, sddk::memory_t::host, sr, wf::band_range(0, 0),
    //        wf::band_range(0, num_bands__), phi1, phi1, ovlp, {&phi1}, phi2, true);
    //wf::inner(spla_ctx, sddk::memory_t::host, sr, phi1, wf::band_range(0, num_bands__), phi1, wf::band_range(0, num_bands__), ovlp, 0, 0);
    //max_diff = sddk::check_identity(ovlp, num_bands__);
    //if (mpi::Communicator::world().rank() == 0) {
    //    printf("checking identity\n");
    //    printf("maximum difference: %18.12f\n", max_diff);
    //    if (max_diff > 1e-10) {
    //        printf("\x1b[31m" "Fail\n" "\x1b[0m" "\n");
    //    } else {
    //        printf("\x1b[32m" "OK\n" "\x1b[0m" "\n");
    //    }
    //}
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--bs=", "{int} block size");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--memory_t=", "{string} type of the memory");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value("mpi_grid_dims", std::vector<int>({1, 1}));
    auto cutoff = args.value<double>("cutoff", 8.0);
    auto bs = args.value<int>("bs", 32);
    auto num_bands = args.value<int>("num_bands", 100);
    std::string memory_t_str = args.value<std::string>("memory_t", "host");

    sirius::initialize(1);

    test_wf_inner(mpi_grid_dims, cutoff, num_bands, bs, sddk::get_memory_t(memory_t_str));

    mpi::Communicator::world().barrier();
    int my_rank = mpi::Communicator::world().rank();

    sirius::finalize(1);

    if (my_rank == 0)  {
        const auto timing_result = ::utils::global_rtgraph_timer.process();
        std::cout << timing_result.print();
        //std::ofstream ofs("timers.json", std::ofstream::out | std::ofstream::trunc);
        //ofs << timing_result.json();
    }
}
