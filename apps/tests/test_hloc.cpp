#include <sirius.hpp>
#include <fstream>

auto simulation_context_factory(bool use_gpu__, std::vector<int> mpi_grid_dims__, double cutoff__, bool gamma__)
{
    //json conf = {
    //    {"control", {
    //                    {"fft_mode", "serial"}
    //                }
    //    }
    //};
    //std::cout << conf << std::endl;
    //auto ctx = std::make_unique<sirius::Simulation_context>(conf.dump());

    auto ctx = std::make_unique<sirius::Simulation_context>();
    if (use_gpu__) {
        ctx->processing_unit("GPU");
    } else {
        ctx->processing_unit("CPU");
    }

    geometry3d::matrix3d<double> M = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};

    ctx->unit_cell().set_lattice_vectors(M);
    ctx->mpi_grid_dims(mpi_grid_dims__);
    ctx->pw_cutoff(cutoff__ + 1);
    ctx->gk_cutoff(cutoff__ / 2);
    ctx->electronic_structure_method("pseudopotential");
    ctx->use_symmetry(false);
    ctx->gamma_point(gamma__);
    ctx->initialize();
    return ctx;
}

template <typename T>
void test_hloc(sirius::Simulation_context& ctx__, int num_bands__, int use_gpu__)
{
    auto gvec = ctx__.gvec_coarse_sptr();
    auto gvec_fft = ctx__.gvec_coarse_fft_sptr();
    auto& fft = ctx__.spfft_coarse<T>();

    if (sddk::Communicator::world().rank() == 0) {
        printf("total number of G-vectors: %i\n", gvec->num_gvec());
        printf("local number of G-vectors: %i\n", gvec->count());
        printf("FFT grid size: %i %i %i\n", fft.dim_x(), fft.dim_y(), fft.dim_z());
        printf("number of FFT threads: %i\n", omp_get_max_threads());
        printf("number of FFT groups:  %i\n", gvec_fft->comm_ortho_fft().size());
        printf("FTT comm size:         %i\n", gvec_fft->comm_fft().size());
        printf("number of z-columns: %i\n", gvec->num_zcol());
        printf("fft_mode           : %s\n", ctx__.cfg().control().fft_mode().c_str());
    }

    sirius::Local_operator<T> hloc(ctx__, fft, gvec_fft);

    wf::Wave_functions<T> phi(gvec, wf::num_mag_dims(0), wf::num_bands(4 * num_bands__), sddk::memory_t::host);
    for (int i = 0; i < 4 * num_bands__; i++) {
        for (int j = 0; j < phi.ld(); j++) {
            phi.pw_coeffs(j, wf::spin_index(0), wf::band_index(i)) = utils::random<std::complex<T>>();
        }
        phi.pw_coeffs(0, wf::spin_index(0), wf::band_index(i)) = 1.0;
    }
    wf::Wave_functions<T> hphi(gvec, wf::num_mag_dims(0), wf::num_bands(4 * num_bands__), sddk::memory_t::host);

    {
        auto mem_phi = (use_gpu__ && gvec_fft->comm_ortho_fft().size() == 1) ? sddk::memory_t::device : sddk::memory_t::host;
        auto copy_policy_phi = (mem_phi == sddk::memory_t::device) ? wf::copy_to::device : wf::copy_to::none;

        auto mem_hphi = (use_gpu__ && gvec_fft->comm_ortho_fft().size() == 1) ? sddk::memory_t::device : sddk::memory_t::host;
        auto copy_policy_hphi = (mem_hphi == sddk::memory_t::device) ? wf::copy_to::host : wf::copy_to::none;

        auto mg1 = phi.memory_guard(mem_phi, copy_policy_phi);
        auto mg2 = hphi.memory_guard(mem_hphi, copy_policy_hphi);

        hloc.prepare_k(*gvec_fft);
        for (int i = 0; i < 4; i++) {
            hloc.apply_h(fft, gvec_fft, wf::spin_range(0), phi, hphi,
                    wf::band_range(i * num_bands__, (i + 1) * num_bands__));
        }
    }

    double diff{0};
    for (int i = 0; i < 4 * num_bands__; i++) {
        for (int j = 0; j < phi.ld(); j++) {
            int ig = gvec->offset() + j;
            auto gc = gvec->gvec_cart<sddk::index_domain_t::global>(ig);
            diff += std::pow(std::abs(static_cast<T>(2.71828 + 0.5 * dot(gc, gc)) * phi.pw_coeffs(j, wf::spin_index(0),
                            wf::band_index(i)) - hphi.pw_coeffs(j, wf::spin_index(0), wf::band_index(i))), 2);
        }
    }
    if (diff != diff) {
        TERMINATE("NaN");
    }
    sddk::Communicator::world().allreduce(&diff, 1);
    diff = std::sqrt(diff / 4 / num_bands__ / gvec->num_gvec());
    if (sddk::Communicator::world().rank() == 0) {
        printf("RMS: %18.16f\n", diff);
    }
    if (diff > 1e-12) {
        RTE_THROW("RMS is too large");
    }
    if (sddk::Communicator::world().rank() == 0) {
        std::cout << "number of hamiltonian applications : " << ctx__.num_loc_op_applied() << std::endl;
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
    args.register_key("--fp32", "use FP32 arithmetics");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value("mpi_grid_dims", std::vector<int>({1, 1}));
    auto cutoff = args.value<double>("cutoff", 10.0);
    auto reduce_gvec = args.value<int>("reduce_gvec", 0);
    auto num_bands = args.value<int>("num_bands", 10);
    auto use_gpu = args.value<int>("use_gpu", 0);
    auto repeat = args.value<int>("repeat", 3);
    auto t_file = args.value<std::string>("t_file", std::string(""));
    auto fp32 = args.exist("fp32");

    sirius::initialize(1);
    int my_rank = sddk::Communicator::world().rank();
    {
        auto ctx = simulation_context_factory(use_gpu, mpi_grid_dims, cutoff, reduce_gvec);
        for (int i = 0; i < repeat; i++) {
            if (fp32) {
#if defined(USE_FP32)
                test_hloc<float>(*ctx, num_bands, use_gpu);
#else
                RTE_THROW("Not compiled with FP32 support");
#endif
            } else {
            test_hloc<double>(*ctx, num_bands, use_gpu);
            }
        }
    }
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
