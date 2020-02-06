#include <sirius.h>

using namespace sirius;

void test_wf_ortho(BLACS_grid const& blacs_grid__,
                   double cutoff__,
                   int num_bands__,
                   int bs__,
                   int num_mag_dims__,
                   memory_t mem__,
                   linalg_t la__)
{
    int nsp = (num_mag_dims__ == 0) ? 1 : 2;
    int num_spin_steps = (num_mag_dims__ == 3) ? 1 : nsp;

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Gvec gvec(M, cutoff__, Communicator::world(), false);
    Gvec_partition gvp(gvec, Communicator::world(), Communicator::self());
    if (Communicator::world().rank() == 0) {
        printf("number of bands          : %i\n", num_bands__);
        printf("number of spins          : %i\n", nsp);
        printf("full spinors             : %i\n", num_mag_dims__ == 3);
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.count());
    }

    int num_atoms = 31;
    auto nmt = [](int i) {
        return 123;
    };

    Wave_functions phi(gvp, num_atoms, nmt, 2 * num_bands__, mem__, nsp);
    Wave_functions tmp(gvp, num_atoms, nmt, 2 * num_bands__, mem__, nsp);

    for (int is = 0; is < nsp; is++) {
        phi.pw_coeffs(is).prime() = [](int64_t i0, int64_t i1){return utils::random<double_complex>();};
        phi.mt_coeffs(is).prime() = [](int64_t i0, int64_t i1){return utils::random<double_complex>();};
    }

    dmatrix<double_complex> ovlp(2 * num_bands__, 2 * num_bands__, blacs_grid__, bs__, bs__);

    if (is_device_memory(mem__)) {
        ovlp.allocate(mem__);
        for (int ispn = 0; ispn < nsp; ispn++) {
            phi.allocate(spin_range(ispn), mem__);
            phi.copy_to(spin_range(ispn), mem__, 0, 2 * num_bands__);
            tmp.allocate(spin_range(ispn), mem__);
        }
    }

    for (int iss = 0; iss < num_spin_steps; iss++) {
        orthogonalize<double_complex, 0, 0>(mem__, la__, num_mag_dims__ == 3 ? 2 : iss, {&phi}, 0,           num_bands__, ovlp, tmp);
        orthogonalize<double_complex, 0, 0>(mem__, la__, num_mag_dims__ == 3 ? 2 : iss, {&phi}, num_bands__, num_bands__, ovlp, tmp);
    }

    for (int iss = 0; iss < num_spin_steps; iss++) {
        inner(mem__, la__, num_mag_dims__ == 3 ? 2 : iss, phi, 0, 2 * num_bands__, phi, 0, 2 * num_bands__, ovlp, 0, 0);
        auto max_diff = check_identity(ovlp, 2 * num_bands__);
        if (Communicator::world().rank() == 0) {
            printf("maximum difference: %18.12f\n", max_diff);
            if (max_diff > 1e-12) {
                printf("\x1b[31m" "Fail\n" "\x1b[0m" "\n");
            } else {
                printf("\x1b[32m" "OK\n" "\x1b[0m" "\n");
            }
        }
    }
}

void call_test(std::vector<int> mpi_grid_dims__,
               double cutoff__,
               int num_bands__,
               int bs__,
               int num_mag_dims__,
               memory_t mem__,
               linalg_t la__,
               int repeat__)
{
    std::unique_ptr<BLACS_grid> blacs_grid;
    if (mpi_grid_dims__[0] * mpi_grid_dims__[1] == 1) {
        blacs_grid = std::unique_ptr<BLACS_grid>(new BLACS_grid(Communicator::self(), mpi_grid_dims__[0], mpi_grid_dims__[1]));
    } else {
        blacs_grid = std::unique_ptr<BLACS_grid>(new BLACS_grid(Communicator::world(), mpi_grid_dims__[0], mpi_grid_dims__[1]));
    }
    for (int i = 0; i < repeat__; i++) {
        test_wf_ortho(*blacs_grid, cutoff__, num_bands__, bs__, num_mag_dims__, mem__, la__);
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--bs=", "{int} block size");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--num_mag_dims=", "{int} number of magnetic dimensions");
    args.register_key("--linalg_t=", "{string} type of the linear algebra driver");
    args.register_key("--memory_t=", "{string} type of memory");

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
    auto num_mag_dims = args.value<int>("num_mag_dims", 0);
    auto la = get_linalg_t(args.value<std::string>("linalg_t", "blas"));
    auto mem = get_memory_t(args.value<std::string>("memory_t", "host"));

    sirius::initialize(1);
    call_test(mpi_grid_dims, cutoff, num_bands, bs, num_mag_dims, mem, la, 1);
    int my_rank = Communicator::world().rank();

    sirius::finalize(1);

    if (my_rank == 0)  {
        const auto timing_result = ::utils::global_rtgraph_timer.process();
        std::cout << timing_result.print();
        //std::ofstream ofs("timers.json", std::ofstream::out | std::ofstream::trunc);
        //ofs << timing_result.json();
    }
}
