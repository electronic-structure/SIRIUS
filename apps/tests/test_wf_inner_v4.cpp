#include <sirius.h>

using namespace sirius;

void test_wf_inner(std::vector<int> mpi_grid_dims__,
                   double cutoff__,
                   int num_bands__,
                   int bs__,
                   linalg_t la__,
                   memory_t mem_bra__,
                   memory_t mem_ket__,
                   memory_t mem_o__)
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
    Wave_functions phi(gvp, 2 * num_bands__, mem_bra__, nsp);

    for (int is = 0; is < nsp; is++) {
        for (int i = 0; i < 2 * num_bands__; i++) {
            for (int igloc = 0; igloc < gvec.count(); igloc++) {
                phi.pw_coeffs(is).prime(igloc, i) = utils::random<double_complex>();
            }
        }
    }

    if (is_device_memory(mem_bra__)) {
        for (int ispn = 0; ispn < nsp; ispn++) {
            phi.allocate(spin_range(ispn), mem_bra__);
            phi.copy_to(spin_range(ispn), mem_bra__, 0, 2 * num_bands__);
        }
    }

    Wave_functions phi1(gvp, 2 * num_bands__, mem_ket__, nsp);
    if (is_device_memory(mem_ket__)) {
        for (int ispn = 0; ispn < nsp; ispn++) {
            phi1.allocate(spin_range(ispn), mem_ket__);
        }
    }
    for (int ispn = 0; ispn < nsp; ispn++) {
        phi1.copy_from(phi, 2 * num_bands__, ispn, 0, ispn, 0);
    }

    dmatrix<double_complex> ovlp(2 * num_bands__, 2 * num_bands__, *blacs_grid, bs__, bs__);

    if (is_device_memory(mem_o__)) {
        ovlp.allocate(mem_o__);
    }
    ovlp.zero();

    inner(mem_o__, la__, 0, phi, 0,           num_bands__, phi1, 0,           num_bands__, ovlp, 0,           0);
    inner(mem_o__, la__, 0, phi, 0,           num_bands__, phi1, num_bands__, num_bands__, ovlp, 0,           num_bands__);
    inner(mem_o__, la__, 0, phi, num_bands__, num_bands__, phi1, 0,           num_bands__, ovlp, num_bands__, 0);
    inner(mem_o__, la__, 0, phi, num_bands__, num_bands__, phi1, num_bands__, num_bands__, ovlp, num_bands__, num_bands__);

    //ovlp.serialize("ovlp", 2 * num_bands__);

    auto max_diff = check_hermitian(ovlp, 2 * num_bands__);
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
    args.register_key("--mem_bra=", "{string} memory type of the <bra| states");
    args.register_key("--mem_ket=", "{string} memory type of the |ket> states");
    args.register_key("--mem_o=", "{string} memory type of the resulting overlap matrix");

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
    auto la = get_linalg_t(args.value<std::string>("linalg_t", "blas"));
    auto mem_bra = get_memory_t(args.value<std::string>("mem_bra", "host"));
    auto mem_ket = get_memory_t(args.value<std::string>("mem_ket", "host"));
    auto mem_o = get_memory_t(args.value<std::string>("mem_o", "host"));

    sirius::initialize(1);

    test_wf_inner(mpi_grid_dims, cutoff, num_bands, bs, la, mem_bra, mem_ket, mem_o);

    Communicator::world().barrier();
    int rank = Communicator::world().rank();

    sirius::finalize(1);

    if (rank == 0)  {
        const auto timing_result = ::utils::global_rtgraph_timer.process();
        std::cout << timing_result.print();
    }
}
