#include <sirius.hpp>
#include "utils/cmd_args.hpp"
#include "SDDK/dmatrix.hpp"
#include <spla/spla.hpp>

void test(int M, int N, int BS, std::vector<int> mpi_grid)
{
    std::vector<int> counts({85, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 17, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 13, 13, 29, 64, 28, 0, 0});

    BLACS_grid blacs_grid(Communicator::world(), mpi_grid[0], mpi_grid[1]);

    int rank = Communicator::world().rank();

    bool local_has_mt  = counts[rank] != 0;
    bool global_has_mt = false;
    // Not all ranks may have mt, but all must call spla if at least one does
    MPI_Allreduce(&local_has_mt, &global_has_mt, 1, MPI_C_BOOL, MPI_LOR, Communicator::world().mpi_comm());

    int nr{0};
    nr = std::accumulate(counts.begin(), counts.end(), nr);

    dmatrix<std::complex<double>> result(M, N, blacs_grid, BS, BS);

    dmatrix<std::complex<double>, matrix_distribution_t::slab> A(counts, M, Communicator::world());
    A.allocate(memory_t::device);
    dmatrix<std::complex<double>, matrix_distribution_t::slab> B(counts, N, Communicator::world());
    B.allocate(memory_t::device);

    auto result_ptr = result.size_local() ? result.at(memory_t::host, 0, 0) : nullptr;
    std::shared_ptr<::spla::Context> spla_ctx{new ::spla::Context{SPLA_PU_GPU}};
    spla_ctx->set_tile_size_gpu(1688); // limit GPU memory usage to around 500MB

    std::cout << "rank" << Communicator::world().rank() << " in" << std::endl;
    if (local_has_mt) {
        spla::pgemm_ssb(
            M, N, counts[rank], SPLA_OP_CONJ_TRANSPOSE, 1.0,
            A.at(memory_t::device, 0, 0), A.ld(), B.at(memory_t::device, 0, 0), B.ld(),
            1.0, result_ptr, result.ld(), 0, 0, result.spla_distribution(), *spla_ctx);
    } else {
        spla::pgemm_ssb(
            M, N, 0, SPLA_OP_CONJ_TRANSPOSE, 1.0,
            nullptr, 0, nullptr, 0,
            1.0, result_ptr, result.ld(), 0, 0, result.spla_distribution(), *spla_ctx);
    }
    std::cout << "rank" << Communicator::world().rank() << " out" << std::endl;
    Communicator::world().barrier();
}

int main(int argn, char **argv)
{
    cmd_args args(argn, argv, {
        {"M=", "{int} M"},
        {"N=", "{int} N"},
        {"BS=", "{int} BS"},
        {"mpi_grid=", "{vector<int>} 2D MPI grid"},
        {"repeat=", "{int} repeat test number of times"}});


    int M = args.value<int>("M", 1000);
    int N = args.value<int>("N", 1000);
    int BS = args.value<int>("BS", 256);
    int repeat = args.value<int>("repeat", 2);
    auto mpi_grid = args.value("mpi_grid", std::vector<int>({1, 1}));

    sirius::initialize(true);

    for (int i = 0; i < repeat; i++) {
        test(M, N, BS, mpi_grid);
    }

    sirius::finalize();
}
