#include <sirius.h>

using namespace sirius;

void test_wf_inner(std::vector<int> mpi_grid_dims__,
                   double cutoff__,
                   int num_bands__,
                   int bs__,
                   linalg_t la__,
                   memory_t mem__)
{
    std::unique_ptr<BLACS_grid> blacs_grid;
    if (mpi_grid_dims__[0] * mpi_grid_dims__[1] == 1) {
        blacs_grid = std::unique_ptr<BLACS_grid>(new BLACS_grid(Communicator::self(), mpi_grid_dims__[0], mpi_grid_dims__[1]));
    } else {
        blacs_grid = std::unique_ptr<BLACS_grid>(new BLACS_grid(Communicator::world(), mpi_grid_dims__[0], mpi_grid_dims__[1]));
    }

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    /* create G-vectors */
    Gvec gvec(M, cutoff__, Communicator::world(), false);

    Gvec_partition gvp(gvec, Communicator::world(), Communicator::self());

    if (Communicator::world().rank() == 0) {
        printf("number of bands          : %i\n", num_bands__);
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.count());
    }

    int nsp{1};
    Wave_functions phi(gvp, num_bands__, memory_t::host, nsp);

    for (int is = 0; is < nsp; is++) {
        phi.zero(device_t::CPU, is, 0, num_bands__);
        for (int i = 0; i < num_bands__; i++) {
            for (int igloc = 0; igloc < gvec.count(); igloc++) {
                if (igloc + gvec.offset() == i) {
                    phi.pw_coeffs(is).prime(igloc, i) = 1.0;
                }
            }
        }
    }

    dmatrix<double_complex> ovlp(num_bands__, num_bands__, *blacs_grid, bs__, bs__);

    if (is_device_memory(mem__)) {
        for (int ispn = 0; ispn < nsp; ispn++) {
            phi.allocate(spin_range(ispn), memory_t::device);
            phi.copy_to(spin_range(ispn), memory_t::device, 0, num_bands__);
        }
        ovlp.allocate(memory_t::device);
    }

    /* warmup call */
    inner(mem__, la__, 0, phi, 0, num_bands__, phi, 0, num_bands__, ovlp, 0, 0);
    Communicator::world().barrier();

    double t = -utils::wtime();
    inner(mem__, la__, 0, phi, 0, num_bands__, phi, 0, num_bands__, ovlp, 0, 0);
    Communicator::world().barrier();
    t += utils::wtime();

    double perf = 8e-9 * num_bands__ * num_bands__ *  gvec.num_gvec() / t;
    if (Communicator::world().rank() == 0) {
        printf("execution time (sec) : %12.6f\n", t);
        printf("performance (GFlops) : %12.6f\n", perf);
    }

    double max_diff{0};
    for (int j = 0; j < ovlp.num_cols_local(); j++) {
        for (int i = 0; i < ovlp.num_rows_local(); i++) {
            double_complex z = (ovlp.irow(i) == ovlp.icol(j)) ? ovlp(i, j) - 1.0 : ovlp(i, j);
            max_diff = std::max(max_diff, std::abs(z));
        }
    }
    Communicator::world().reduce<double, mpi_op_t::max>(&max_diff, 1, 0);
    if (Communicator::world().rank() == 0) {
        printf("maximum difference: %18.12f\n", max_diff);
        if (max_diff > 1e-12) {
            printf("\x1b[31m" "Fail\n" "\x1b[0m" "\n");
        } else {
            printf("\x1b[32m" "OK\n" "\x1b[0m" "\n");
        }
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--bs=", "{int} block size");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--linalg_t=", "{string} type of the linear algebra driver");
    args.register_key("--memory_t=", "{string} type of the memory");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});
    auto cutoff = args.value<double>("cutoff", 8.0);
    auto bs = args.value<int>("bs", 32);
    auto num_bands = args.value<int>("num_bands", 100);
    std::string linalg_t_str = args.value<std::string>("linalg_t", "blas");
    std::string memory_t_str = args.value<std::string>("memory_t", "host");

    sirius::initialize(1);

    test_wf_inner(mpi_grid_dims, cutoff, num_bands, bs, get_linalg_t(linalg_t_str), get_memory_t(memory_t_str));

    Communicator::world().barrier();
    int my_rank = Communicator::world().rank();

    sirius::finalize(1);

    if (my_rank == 0)  {
        const auto timing_result = ::utils::global_rtgraph_timer.process();
        std::cout << timing_result.print();
        //std::ofstream ofs("timers.json", std::ofstream::out | std::ofstream::trunc);
        //ofs << timing_result.json();
    }
}
