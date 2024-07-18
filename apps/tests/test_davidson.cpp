/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "hamiltonian/davidson.hpp"
#include "hamiltonian/initialize_subspace.hpp"

using namespace sirius;

template <typename T>
void
init_wf(K_point<T>* kp__, wf::Wave_functions<T>& phi__, int num_bands__, int num_mag_dims__)
{
    std::vector<double> tmp(0xFFFF);
    for (int i = 0; i < 0xFFFF; i++) {
        tmp[i] = random<double>();
    }

    phi__.zero(memory_t::host, wf::spin_index(0), wf::band_range(0, num_bands__));

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
        wf::copy(memory_t::host, phi__, wf::spin_index(0), wf::band_range(0, num_bands__), phi__, wf::spin_index(1),
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
    sirius::initialize_subspace<T, F>(Hk, kp, ctx__.unit_cell().num_ps_atomic_wf().first);
    for (int i = 0; i < ctx__.num_bands(); i++) {
        kp.band_energy(i, 0, 0);
    }
    init_wf(&kp, kp.spinor_wave_functions(), ctx__.num_bands(), 0);

    const int num_bands = ctx__.num_bands();
    bool locking{true};

    auto result = davidson<T, F, davidson_evp_t::hamiltonian>(
            Hk, kp, wf::num_bands(num_bands), wf::num_mag_dims(ctx__.num_mag_dims()), kp.spinor_wave_functions(),
            [&](int i, int ispn) { return eval_tol__; }, res_tol__, 60, locking, subspace_size__, estimate_eval__,
            extra_ortho__, std::cout, 2);

    if (mpi::Communicator::world().rank() == 0 && only_kin__) {
        std::vector<double> ekin(kp.num_gkvec());
        for (int i = 0; i < kp.num_gkvec(); i++) {
            ekin[i] = 0.5 * kp.gkvec().gkvec_cart(gvec_index_t::global(i)).length2();
        }
        std::sort(ekin.begin(), ekin.end());

        double max_diff{0};
        for (int i = 0; i < ctx__.num_bands(); i++) {
            max_diff = std::max(max_diff, std::abs(ekin[i] - result.eval(i, 0)));
            printf("%20.16f %20.16f %20.16e\n", ekin[i], result.eval(i, 0), std::abs(ekin[i] - result.eval(i, 0)));
        }
        printf("maximum eigen-value difference: %20.16e\n", max_diff);
    }

    if (mpi::Communicator::world().rank() == 0 && !only_kin__) {
        std::cout << "Converged eigen-values" << std::endl;
        for (int i = 0; i < ctx__.num_bands(); i++) {
            printf("e[%i] = %20.16f\n", i, result.eval(i, 0));
        }
    }
}

int
test_davidson(cmd_args const& args__)
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
    auto json_conf                              = R"({
      "parameters" : {
        "electronic_structure_method" : "pseudopotential"
      }
    })"_json;
    json_conf["control"]["processing_unit"]     = args__.value<std::string>("device", "CPU");
    json_conf["control"]["mpi_grid_dims"]       = mpi_grid;
    json_conf["control"]["std_evp_solver_name"] = solver;
    json_conf["control"]["gen_evp_solver_name"] = solver;
    json_conf["parameters"]["pw_cutoff"]        = pw_cutoff;
    json_conf["parameters"]["gk_cutoff"]        = gk_cutoff;
    json_conf["parameters"]["gamma_point"]      = false;
    if (num_bands >= 0) {
        json_conf["parameters"]["num_bands"] = num_bands;
    }

    std::vector<r3::vector<double>> coord;
    double p = 1.0 / N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                coord.push_back(r3::vector<double>(i * p, j * p, k * p));
            }
        }
    }

    double a{5};
    auto sctx_ptr = sirius::create_simulation_context(json_conf, {{a * N, 0, 0}, {0, a * N, 0}, {0, 0, a * N}},
                                                      N * N * N, coord, add_vloc, add_dion);

    auto& ctx = *sctx_ptr;
    PROFILE_STOP("test_davidson|setup")

    ////ctx.cfg().iterative_solver().type("exact");

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
#if defined(SIRIUS_USE_FP32)
            diagonalize<float, std::complex<float>>(ctx, vk, pot, res_tol, eval_tol, only_kin, subspace_size,
                                                    estimate_eval, extra_ortho);
#endif
        }
        if (precision_wf == "fp32" && precision_hs == "fp64") {
#if defined(SIRIUS_USE_FP32)
            diagonalize<float, std::complex<double>>(ctx, vk, pot, res_tol, eval_tol, only_kin, subspace_size,
                                                     estimate_eval, extra_ortho);
#endif
        }
        if (precision_wf == "fp64" && precision_hs == "fp64") {
            diagonalize<double, std::complex<double>>(ctx, vk, pot, res_tol, eval_tol, only_kin, subspace_size,
                                                      estimate_eval, extra_ortho);
        }
    }
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"device=", "(string) CPU or GPU"},
                   {"pw_cutoff=", "(double) plane-wave cutoff for density and potential"},
                   {"gk_cutoff=", "(double) plane-wave cutoff for wave-functions"},
                   {"num_bands=", "(int) number of bands"},
                   {"N=", "(int) cell multiplicity"},
                   {"mpi_grid=", "(int[2]) dimensions of the MPI grid for band diagonalization"},
                   {"solver=", "(string) eigen-value solver"},
                   {"res_tol=", "(double) residual L2-norm tolerance"},
                   {"eval_tol=", "(double) eigen-value tolerance"},
                   {"subspace_size=", "(int) size of the diagonalization subspace"},
                   {"use_res_norm", "use residual norm to estimate the convergence"},
                   {"extra_ortho", "use second orthogonalisation"},
                   {"precision_wf=", "{string} precision of wave-functions"},
                   {"precision_hs=", "{string} precision of the Hamiltonian subspace"},
                   {"only_kin", "use kinetic-operator only"}});

    sirius::initialize(1);
    int result = call_test("test_davidson", test_davidson, args);
    sirius::finalize();
    return result;
}
