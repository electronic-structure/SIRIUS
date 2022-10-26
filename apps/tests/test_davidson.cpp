#include <sirius.hpp>
#include "band/davidson.hpp"

using namespace sirius;

template <typename T>
void init_wf(K_point<T>* kp__, wf::Wave_functions<T>& phi__, int num_bands__, int num_mag_dims__)
{
    std::vector<double> tmp(0xFFFF);
    for (int i = 0; i < 0xFFFF; i++) {
        tmp[i] = utils::random<double>();
    }

    phi__.zero(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, num_bands__));

    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_bands__; i++) {
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->gkvec().offset() + igk_loc;
            if (igk == 0) {
                phi__.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(i)) = 1.0;
            }
            if (igk == i + 1) {
                phi__.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(i)) = 0.5;
            }
            if (igk == i + 2) {
                phi__.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(i)) = 0.25;
            }
            if (igk == i + 3) {
                phi__.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(i)) = 0.125;
            }
            phi__.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(i)) += tmp[(igk + i) & 0xFFFF] * 1e-5;
        }
    }

    if (num_mag_dims__ == 3) {
        /* make pure spinor up- and dn- wave functions */
        wf::copy(sddk::memory_t::host, phi__, wf::spin_index(0), wf::band_range(0, num_bands__), phi__, wf::spin_index(1),
                wf::band_range(num_bands__, 2 * num_bands__));
    }
}

template <typename T, typename F>
void
diagonalize(Simulation_context& ctx__, std::array<double, 3> vk__, Potential& pot__, double res_tol__,
            double eval_tol__, bool only_kin__, int subspace_size__, bool estimate_eval__, bool extra_ortho__)
{
    K_point<T> kp(ctx__, &vk__[0], 1.0);
    kp.initialize();
    std::cout << "num_gkvec=" << kp.num_gkvec() << "\n";
    for (int i = 0; i < ctx__.num_bands(); i++) {
        kp.band_occupancy(i, 0, 2);
    }

    Hamiltonian0<T> H0(pot__, true);
    auto Hk = H0(kp);
    sirius::initialize_subspace<T, F>(Hk, ctx__.unit_cell().num_ps_atomic_wf().first);
    for (int i = 0; i < ctx__.num_bands(); i++) {
        kp.band_energy(i, 0, 0);
    }
    init_wf(&kp, kp.spinor_wave_functions(), ctx__.num_bands(), 0);


    /*
     * debug kinetic energy Hamiltonian (single MPI rank only)
     */

    //const int bs = ctx__.cyclic_block_size();
    const int num_bands = ctx__.num_bands();
    //auto& gv = kp.gkvec();
    //auto& phi = kp.spinor_wave_functions();
    //sddk::dmatrix<std::complex<T>> hmlt(num_bands, num_bands, ctx__.blacs_grid(), bs, bs);
    //hmlt.zero();

    //for (int i = 0; i < num_bands; i++) {
    //    for (int j = 0; j < num_bands; j++) {
    //        for (int ig = 0; ig < gv.num_gvec(); ig++) {
    //            auto gvc = gv.template gkvec_cart<index_domain_t::global>(ig);
    //            T ekin = 0.5 * dot(gvc, gvc);
    //            hmlt(i, j) += std::conj(phi.pw_coeffs(0).prime(ig, i)) * ekin * phi.pw_coeffs(0).prime(ig, j);
    //        }
    //    }
    //}
    //auto max_diff = check_hermitian(hmlt, num_bands);
    //std::cout << "Simple kinetic Hamiltonian: error in hermiticity: " << std::setw(24) << std::scientific << max_diff << std::endl;
    //hmlt.serialize("hmlt", num_bands);
    //
    bool locking{true};

    auto result = davidson<T, F, davidson_evp_t::hamiltonian>(Hk, wf::num_bands(num_bands),
            wf::num_mag_dims(ctx__.num_mag_dims()), kp.spinor_wave_functions(), [&](int i, int ispn){return eval_tol__;}, res_tol__,
            60, locking, subspace_size__, estimate_eval__, extra_ortho__, std::cout, 2);

    if (sddk::Communicator::world().rank() == 0 && only_kin__) {
        std::vector<double> ekin(kp.num_gkvec());
        for (int i = 0; i < kp.num_gkvec(); i++) {
            ekin[i] = 0.5 * kp.gkvec().template gkvec_cart<sddk::index_domain_t::global>(i).length2();
        }
        std::sort(ekin.begin(), ekin.end());

        double max_diff{0};
        for (int i = 0; i < ctx__.num_bands(); i++) {
            max_diff = std::max(max_diff, std::abs(ekin[i] - result.eval(i, 0)));
            printf("%20.16f %20.16f %20.16e\n", ekin[i], result.eval(i, 0), std::abs(ekin[i] - result.eval(i, 0)));
        }
        printf("maximum eigen-value difference: %20.16e\n", max_diff);
    }

    if (sddk::Communicator::world().rank() == 0 && !only_kin__) {
        std::cout << "Converged eigen-values" << std::endl;
        for (int i = 0; i < ctx__.num_bands(); i++) {
            printf("e[%i] = %20.16f\n", i, result.eval(i, 0));
        }
    }
}

void test_davidson(cmd_args const& args__)
{
    auto pw_cutoff     = args__.value<double>("pw_cutoff", 30);
    auto gk_cutoff     = args__.value<double>("gk_cutoff", 10);
    auto N             = args__.value<int>("N", 1);
    auto mpi_grid      = args__.value("mpi_grid", std::vector<int>({1, 1}));
    auto solver        = args__.value<std::string>("solver", "lapack");
    auto precision_wf  = args__.value<std::string>("precision_wf", "fp64");
    auto precision_hs  = args__.value<std::string>("precision_hs", "fp64");
    auto res_tol       = args__.value<double>("res_tol", 1e-5);
    auto eval_tol      = args__.value<double>("eval_tol", 1e-7);
    auto only_kin      = args__.exist("only_kin");
    auto subspace_size = args__.value<int>("subspace_size", 2);
    auto estimate_eval = !args__.exist("use_res_norm");
    auto extra_ortho   = args__.exist("extra_ortho");

    int num_bands{-1};
    num_bands = args__.value<int>("num_bands", num_bands);

    bool add_dion{!only_kin};
    bool add_vloc{!only_kin};

    PROFILE_START("test_davidson|setup")

    /* create simulation context */
    Simulation_context ctx(
        "{"
        "   \"parameters\" : {"
        "        \"electronic_structure_method\" : \"pseudopotential\""
        "    },"
        "   \"control\" : {"
        "       \"verification\" : 0,"
        "       \"print_checksum\" : false"
        "    }"
        "}");

    /* add a new atom type to the unit cell */
    auto& atype = ctx.unit_cell().add_atom_type("Cu");
    /* set pseudo charge */
    atype.zn(11);
    /* set radial grid */
    atype.set_radial_grid(radial_grid_t::lin_exp, 1000, 0.0, 100.0, 6);
    /* cutoff at ~1 a.u. */
    int icut = atype.radial_grid().index_of(1.0);
    double rcut = atype.radial_grid(icut);
    /* create beta radial function */
    std::vector<double> beta(icut + 1);
    std::vector<double> beta1(icut + 1);
    for (int l = 0; l <= 2; l++) {
        for (int i = 0; i <= icut; i++) {
            double x = atype.radial_grid(i);
            beta[i] = utils::confined_polynomial(x, rcut, l, l + 1, 0);
            beta1[i] = utils::confined_polynomial(x, rcut, l, l + 2, 0);
        }
        /* add radial function for l */
        atype.add_beta_radial_function(l, beta);
        atype.add_beta_radial_function(l, beta1);
    }

    std::vector<double> ps_wf(atype.radial_grid().num_points());
    for (int l = 0; l <= 2; l++) {
        for (int i = 0; i < atype.radial_grid().num_points(); i++) {
            double x = atype.radial_grid(i);
            ps_wf[i] = std::exp(-x) * std::pow(x, l);
        }
        /* add radial function for l */
        atype.add_ps_atomic_wf(3, sirius::experimental::angular_momentum(l), ps_wf);
    }

    /* set local part of potential */
    std::vector<double> vloc(atype.radial_grid().num_points(), 0);
    if (add_vloc) {
        for (int i = 0; i < atype.radial_grid().num_points(); i++) {
            double x = atype.radial_grid(i);
            vloc[i] = -atype.zn() / (std::exp(-x * (x + 1)) + x);
        }
    }
    atype.local_potential(vloc);
    /* set Dion matrix */
    int nbf = atype.num_beta_radial_functions();
    sddk::matrix<double> dion(nbf, nbf);
    dion.zero();
    if (add_dion) {
        for (int i = 0; i < nbf; i++) {
            dion(i, i) = -10.0;
        }
    }
    atype.d_mtrx_ion(dion);
    /* set atomic density */
    std::vector<double> arho(atype.radial_grid().num_points());
    for (int i = 0; i < atype.radial_grid().num_points(); i++) {
        double x = atype.radial_grid(i);
        arho[i] = 2 * atype.zn() * std::exp(-x * x) * x;
    }
    atype.ps_total_charge_density(arho);

    /* lattice constant */
    double a{5};
    /* set lattice vectors */
    ctx.unit_cell().set_lattice_vectors({{a * N, 0, 0},
                                         {0, a * N, 0},
                                         {0, 0, a * N}});
    /* add atoms */
    double p = 1.0 / N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                ctx.unit_cell().add_atom("Cu", {i * p, j * p, k * p});
            }
        }
    }

    /* initialize the context */
    ctx.verbosity(4);
    ctx.pw_cutoff(pw_cutoff);
    ctx.gk_cutoff(gk_cutoff);
    ctx.processing_unit(args__.value<std::string>("device", "CPU"));
    ctx.mpi_grid_dims(mpi_grid);
    ctx.gen_evp_solver_name(solver);
    ctx.std_evp_solver_name(solver);
    if (num_bands >= 0) {
        ctx.num_bands(num_bands);
    }

    PROFILE_STOP("test_davidson|setup")

    //ctx.cfg().iterative_solver().type("exact");

    /* initialize simulation context */
    ctx.initialize();

    std::cout << "number of atomic orbitals: " << ctx.unit_cell().num_ps_atomic_wf().first << "\n";

    Density rho(ctx);
    rho.initial_density();
    rho.zero();

    Potential pot(ctx);
    pot.generate(rho, ctx.use_symmetry(), true);
    pot.zero();

    /* repeat several times for the accurate performance measurment */
    for (int r = 0; r < 1; r++) {
        std::array<double, 3> vk({0.1, 0.1, 0.1});
        if (ctx.comm().rank() == 0) {
            std::cout << "precision_wf: " << precision_wf << ", precision_hs: " << precision_hs << std::endl;
        }
        if (precision_wf == "fp32" && precision_hs == "fp32") {
#if defined(USE_FP32)
            diagonalize<float, std::complex<float>>(ctx, vk, pot, res_tol, eval_tol, only_kin, subspace_size, estimate_eval, extra_ortho);
#endif
        }
        if (precision_wf == "fp32" && precision_hs == "fp64") {
#if defined(USE_FP32)
            diagonalize<float, std::complex<double>>(ctx, vk, pot, res_tol, eval_tol, only_kin, subspace_size, estimate_eval, extra_ortho);
#endif
        }
        if (precision_wf == "fp64" && precision_hs == "fp64") {
            diagonalize<double, std::complex<double>>(ctx, vk, pot, res_tol, eval_tol, only_kin, subspace_size, estimate_eval, extra_ortho);
        }
    }
}

int main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"device=",        "(string) CPU or GPU"},
                               {"pw_cutoff=",     "(double) plane-wave cutoff for density and potential"},
                               {"gk_cutoff=",     "(double) plane-wave cutoff for wave-functions"},
                               {"num_bands=",     "(int) number of bands"},
                               {"N=",             "(int) cell multiplicity"},
                               {"mpi_grid=",      "(int[2]) dimensions of the MPI grid for band diagonalization"},
                               {"solver=",        "(string) eigen-value solver"},
                               {"res_tol=",       "(double) residual L2-norm tolerance"},
                               {"eval_tol=",      "(double) eigan-value tolerance"},
                               {"subspace_size=", "(int) size of the diagonalization subspace"},
                               {"use_res_norm",   "use residual norm to estimate the convergence"},
                               {"extra_ortho",    "use second orthogonalisation"},
                               {"precision_wf=",  "{string} precision of wave-functions"},
                               {"precision_hs=",  "{string} precision of the Hamiltonian subspace"},
                               {"only_kin",       "use kinetic-operator only"}
                              });

    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test_davidson(args);
    int rank = sddk::Communicator::world().rank();
    sirius::finalize();
    if (rank == 0)  {
        const auto timing_result = ::utils::global_rtgraph_timer.process();
        std::cout<< timing_result.print();
    }
}
