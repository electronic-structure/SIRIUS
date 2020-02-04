#include <sirius.h>
#include <fstream>
#include <string>

using namespace sirius;

void test_hloc(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__, int reduce_gvec__,
               int use_gpu__, int gpu_ptr__)
{
    device_t pu = static_cast<device_t>(use_gpu__);

    matrix3d<double> M = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};

    for (int i = 0; i < 3; i++) {
        printf("  a%1i : %18.10f %18.10f %18.10f \n", i + 1, M(0, i), M(1, i), M(2, i));
    }

    Simulation_context params;
    params.set_processing_unit(pu);
    params.unit_cell().set_lattice_vectors(M);
    params.mpi_grid_dims(mpi_grid_dims__);
    params.pw_cutoff(cutoff__ + 1);
    params.gk_cutoff(cutoff__ / 2);
    params.electronic_structure_method("pseudopotential");
    params.use_symmetry(false);
    params.initialize();

    auto& gvec = params.gvec();
    auto& gvecp = params.gvec_partition();
    auto& fft = params.spfft();

    if (Communicator::world().rank() == 0) {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.gvec_count(0));
        printf("FFT grid size: %i %i %i\n", fft.dim_x(), fft.dim_y(), fft.dim_z());
        printf("number of FFT threads: %i\n", omp_get_max_threads());
        printf("number of FFT groups: %i\n", params.comm_ortho_fft().size());
        //printf("MPI grid: %i %i\n", mpi_grid.communicator(1 << 0).size(), mpi_grid.communicator(1 << 1).size());
        printf("number of z-columns: %i\n", gvec.num_zcol());
    }

    Local_operator hloc(params, fft, gvecp);

    Wave_functions phi(gvecp, 4 * num_bands__, memory_t::host);
    for (int i = 0; i < 4 * num_bands__; i++) {
        for (int j = 0; j < phi.pw_coeffs(0).num_rows_loc(); j++) {
            phi.pw_coeffs(0).prime(j, i) = utils::random<double_complex>();
        }
        phi.pw_coeffs(0).prime(0, i) = 1.0;
    }
    Wave_functions hphi(gvecp, 4 * num_bands__, memory_t::host);

    if (pu == device_t::GPU) {
        phi.pw_coeffs(0).allocate(memory_t::device);
        phi.pw_coeffs(0).copy_to(memory_t::device, 0, 4 * num_bands__);
        hphi.pw_coeffs(0).allocate(memory_t::device);
    }
    hloc.prepare_k(gvecp); 
    for (int i = 0; i < 4; i++) {
        hloc.apply_h(fft, gvecp, spin_range(0), phi, hphi, i * num_bands__, num_bands__);
    }
    //hloc.dismiss();

    double diff{0};
    for (int i = 0; i < 4 * num_bands__; i++) {
        for (int j = 0; j < phi.pw_coeffs(0).num_rows_loc(); j++) {
            int ig = gvec.offset() + j;
            auto gc = gvec.gvec_cart<index_domain_t::global>(ig);
            diff += std::pow(std::abs((2.71828 + 0.5 * dot(gc, gc)) * phi.pw_coeffs(0).prime(j, i) - hphi.pw_coeffs(0).prime(j, i)), 2);
        }
    }
    if (diff != diff) {
        TERMINATE("NaN");
    }
    Communicator::world().allreduce(&diff, 1);
    diff = std::sqrt(diff / 4 / num_bands__ / gvec.num_gvec());
    if (Communicator::world().rank() == 0) {
        printf("RMS: %18.16f\n", diff);
    }
    if (diff > 1e-12) {
        TERMINATE("RMS is too large");
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--reduce_gvec=", "{int} 0: use full set of G-vectors, 1: use reduced set of G-vectors");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");
    args.register_key("--gpu_ptr=", "{int} 0: start from CPU, 1: start from GPU");
    args.register_key("--repeat=", "{int} number of repetitions");
    args.register_key("--t_file=", "{string} name of timing output file");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value< std::vector<int> >("mpi_grid_dims", {1, 1});
    auto cutoff = args.value<double>("cutoff", 10.0);
    auto reduce_gvec = args.value<int>("reduce_gvec", 0);
    auto num_bands = args.value<int>("num_bands", 10);
    auto use_gpu = args.value<int>("use_gpu", 0);
    auto gpu_ptr = args.value<int>("gpu_ptr", 0);
    auto repeat = args.value<int>("repeat", 3);
    auto t_file = args.value<std::string>("t_file", std::string(""));

    sirius::initialize(1);
    for (int i = 0; i < repeat; i++) {
        test_hloc(mpi_grid_dims, cutoff, num_bands, reduce_gvec, use_gpu, gpu_ptr);
    }
    int my_rank = Communicator::world().rank();

    sirius::finalize(1);

    if (my_rank == 0)  {
        const auto timing_result = ::utils::global_rtgraph_timer.process();
        std::cout << timing_result.print();
        //if (!t_file.empty()) {
        //    std::ofstream json_file(t_file);
        //    json_file << std::setw(2) << utils::timer::serialize() << std::endl;
        //}
    }
}
