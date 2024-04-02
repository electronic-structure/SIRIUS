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
#include "multi_cg/multi_cg.hpp"

using namespace sirius;

void
linear_solver_executor(Simulation_context const& sctx, Hamiltonian0<double> const& H0, double const* vkq__,
                       int const* num_gvec_kq_loc__, int const* gvec_kq_loc__, std::complex<double>* dpsi__,
                       std::complex<double>* psi__, double* eigvals__, std::complex<double>* dvpsi__, int const* ld__,
                       int const* num_spin_comp__, double const* alpha_pv__, int const* spin__, int const* nbnd_occ__,
                       double tol)
{
    PROFILE("linear_solver_executor")

    /* works for non-magnetic and collinear cases */
    RTE_ASSERT(*num_spin_comp__ == 1);

    int nbnd_occ = *nbnd_occ__;

    if (nbnd_occ == 0) {
        return;
    }

    wf::spin_range sr(0);
    if (sctx.num_mag_dims() == 1) {
        if (!(*spin__ == 1 || *spin__ == 2)) {
            RTE_THROW("wrong spin channel");
        }
        sr = wf::spin_range(*spin__ - 1);
    }

    std::shared_ptr<fft::Gvec> gvkq_in;
    gvkq_in = std::make_shared<fft::Gvec>(r3::vector<double>(vkq__), sctx.unit_cell().reciprocal_lattice_vectors(),
                                          *num_gvec_kq_loc__, gvec_kq_loc__, sctx.comm_band(), false);

    int num_gvec_kq_loc = *num_gvec_kq_loc__;
    int num_gvec_kq     = num_gvec_kq_loc;
    sctx.comm_band().allreduce(&num_gvec_kq, 1);

    if (num_gvec_kq != gvkq_in->num_gvec()) {
        RTE_THROW("wrong number of G+k vectors for k");
    }

    sirius::K_point<double> kp(const_cast<sirius::Simulation_context&>(sctx), gvkq_in, 1.0);
    kp.initialize();

    auto Hk = H0(kp);

    /* copy eigenvalues (factor 2 for rydberg/hartree) */
    std::vector<double> eigvals_vec(eigvals__, eigvals__ + nbnd_occ);
    for (auto& val : eigvals_vec) {
        val /= 2;
    }

    // Setup dpsi (unknown), psi (part of projector), and dvpsi (right-hand side)
    mdarray<std::complex<double>, 3> psi({*ld__, *num_spin_comp__, nbnd_occ}, psi__);
    mdarray<std::complex<double>, 3> dpsi({*ld__, *num_spin_comp__, nbnd_occ}, dpsi__);
    mdarray<std::complex<double>, 3> dvpsi({*ld__, *num_spin_comp__, nbnd_occ}, dvpsi__);

    auto dpsi_wf = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ), wf::num_mag_dims(0), false);
    auto psi_wf  = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ), wf::num_mag_dims(0), false);
    auto dvpsi_wf =
            sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ), wf::num_mag_dims(0), false);
    auto tmp_wf = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ), wf::num_mag_dims(0), false);

    for (int ispn = 0; ispn < *num_spin_comp__; ispn++) {
        for (int i = 0; i < nbnd_occ; i++) {
            for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                psi_wf->pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i))  = psi(ig, ispn, i);
                dpsi_wf->pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i)) = dpsi(ig, ispn, i);
                // divide by two to account for hartree / rydberg, this is
                // dv * psi and dv should be 2x smaller in sirius.
                dvpsi_wf->pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i)) = dvpsi(ig, ispn, i) / 2.0;
            }
        }
    }

    /* check residuals H|psi> - e * S |psi> */
    if (sctx.cfg().control().verification() >= 1) {
        sirius::K_point<double> kp(const_cast<sirius::Simulation_context&>(sctx), gvkq_in, 1.0);
        kp.initialize();
        auto Hk = H0(kp);
        sirius::check_wave_functions<double, std::complex<double>>(Hk, *psi_wf, sr, wf::band_range(0, nbnd_occ),
                                                                   eigvals_vec.data());
    }

    /* setup auxiliary state vectors for CG */
    auto U = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ), wf::num_mag_dims(0), false);
    auto C = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ), wf::num_mag_dims(0), false);

    auto Hphi_wf = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ), wf::num_mag_dims(0), false);
    auto Sphi_wf = sirius::wave_function_factory<double>(sctx, kp, wf::num_bands(nbnd_occ), wf::num_mag_dims(0), false);

    auto mem = sctx.processing_unit_memory_t();

    std::vector<wf::device_memory_guard> mg;

    mg.emplace_back(psi_wf->memory_guard(mem, wf::copy_to::device));
    mg.emplace_back(dpsi_wf->memory_guard(mem, wf::copy_to::device | wf::copy_to::host));
    mg.emplace_back(dvpsi_wf->memory_guard(mem, wf::copy_to::device));
    mg.emplace_back(tmp_wf->memory_guard(mem, wf::copy_to::device));

    mg.emplace_back(U->memory_guard(mem, wf::copy_to::device));
    mg.emplace_back(C->memory_guard(mem, wf::copy_to::device));
    mg.emplace_back(Hphi_wf->memory_guard(mem, wf::copy_to::device));
    mg.emplace_back(Sphi_wf->memory_guard(mem, wf::copy_to::device));

    sirius::lr::Linear_response_operator linear_operator(const_cast<sirius::Simulation_context&>(sctx), Hk, eigvals_vec,
                                                         Hphi_wf.get(), Sphi_wf.get(), psi_wf.get(), tmp_wf.get(),
                                                         *alpha_pv__ / 2, // rydberg/hartree factor
                                                         wf::band_range(0, nbnd_occ), sr, mem);
    /* CG state vectors */
    auto X_wrap = sirius::lr::Wave_functions_wrap{dpsi_wf.get(), mem};
    auto B_wrap = sirius::lr::Wave_functions_wrap{dvpsi_wf.get(), mem};
    auto U_wrap = sirius::lr::Wave_functions_wrap{U.get(), mem};
    auto C_wrap = sirius::lr::Wave_functions_wrap{C.get(), mem};

    /* set up the diagonal preconditioner */
    auto h_o_diag = Hk.get_h_o_diag_pw<double, 3>(); // already on the GPU if mem=GPU
    mdarray<double, 1> eigvals_mdarray({eigvals_vec.size()});
    eigvals_mdarray = [&](int i) { return eigvals_vec[i]; };
    /* allocate and copy eigvals_mdarray to GPU if running on GPU */
    if (is_device_memory(mem)) {
        eigvals_mdarray.allocate(mem).copy_to(mem);
    }

    sirius::lr::Smoothed_diagonal_preconditioner preconditioner{
            std::move(h_o_diag.first), std::move(h_o_diag.second), std::move(eigvals_mdarray), nbnd_occ, mem, sr};

    // Identity_preconditioner preconditioner{static_cast<size_t>(nbnd_occ)};

    auto result = sirius::cg::multi_cg(linear_operator, preconditioner, X_wrap, B_wrap, U_wrap, C_wrap, // state vectors
                                       20,                                                              // iters
                                       tol                                                              // tol
    );
    mg.clear();

    if (mpi::Communicator::world().rank() == 0) {
        std::cout << "converged in " << result.niter << " iterations" << std::endl;
    }

    /* bring wave functions back in order of QE */
    for (int ispn = 0; ispn < *num_spin_comp__; ispn++) {
        for (int i = 0; i < nbnd_occ; i++) {
            for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                dpsi(ig, ispn, i) = dpsi_wf->pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i));
            }
        }
    }
}

template <typename T>
void
init_wf(K_point<T> const& kp__, wf::Wave_functions<T>& phi__, int num_bands__, int num_mag_dims__)
{
    std::vector<double> tmp(0xFFFF);
    for (int i = 0; i < 0xFFFF; i++) {
        tmp[i] = random<double>();
    }

    phi__.zero(memory_t::host, wf::spin_index(0), wf::band_range(0, num_bands__));

    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_bands__; i++) {
        for (int igk_loc = 0; igk_loc < kp__.num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__.gkvec().offset() + igk_loc;
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
solve_lr(Simulation_context& ctx__, std::array<double, 3> vk__, Potential& pot__, double tol)
{
    /* create k-point */
    K_point<T> kp(ctx__, &vk__[0], 1.0);
    kp.initialize();
    if (mpi::Communicator::world().rank() == 0) {
        std::cout << "num_gkvec: " << kp.num_gkvec() << "\n";
        std::cout << "num_bands: " << ctx__.num_bands() << "\n";
    }
    for (int i = 0; i < ctx__.num_bands(); i++) {
        kp.band_occupancy(i, 0, 2);
    }
    /* create k-point independent Hamiltonian */
    Hamiltonian0<T> H0(pot__, true);
    auto Hk = H0(kp);
    /* initialize kp.spinor_wave_functions */
    sirius::initialize_subspace<T, std::complex<F>>(Hk, kp, ctx__.unit_cell().num_ps_atomic_wf().first);

    /* create auxiliary wave-functions */
    auto dpsi = wave_function_factory(ctx__, kp, wf::num_bands(ctx__.num_bands()), wf::num_mag_dims(0), false);
    init_wf(kp, *dpsi, ctx__.num_bands(), 0);

    auto dvpsi = wave_function_factory(ctx__, kp, wf::num_bands(ctx__.num_bands()), wf::num_mag_dims(0), false);
    init_wf(kp, *dvpsi, ctx__.num_bands(), 0);

    const int num_bands = ctx__.num_bands();

    int num_gvec_kq_loc = kp.num_gkvec_loc();
    int ld              = kp.spinor_wave_functions().ld();
    int num_spin_comp{1};
    double alpha_pv{1.0};
    int spin{0};

    auto& gvec_kq_loc = kp.gkvec().gvec_local();

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) {
        eval[i] = kp.band_energy(i, 0);
    }

    linear_solver_executor(ctx__, H0, &vk__[0], &num_gvec_kq_loc, &gvec_kq_loc(0, 0),
                           dpsi->at(memory_t::host, 0, wf::spin_index(0), wf::band_index(0)),
                           kp.spinor_wave_functions().at(memory_t::host, 0, wf::spin_index(0), wf::band_index(0)),
                           eval.data(), dvpsi->at(memory_t::host, 0, wf::spin_index(0), wf::band_index(0)), &ld,
                           &num_spin_comp, &alpha_pv, &spin, &num_bands, tol);
}

void
test_lr_solver(cmd_args const& args__)
{
    auto pw_cutoff = args__.value<double>("pw_cutoff", 30);
    auto gk_cutoff = args__.value<double>("gk_cutoff", 10);
    auto N         = args__.value<int>("N", 1);
    auto mpi_grid  = args__.value("mpi_grid", std::vector<int>({1, 1}));
    // auto solver        = args__.value<std::string>("solver", "lapack");
    // auto precision_wf  = args__.value<std::string>("precision_wf", "fp64");
    // auto precision_hs  = args__.value<std::string>("precision_hs", "fp64");
    // auto res_tol       = args__.value<double>("res_tol", 1e-5);
    // auto eval_tol      = args__.value<double>("eval_tol", 1e-7);
    auto only_kin = args__.exist("only_kin");
    // auto subspace_size = args__.value<int>("subspace_size", 2);
    // auto estimate_eval = !args__.exist("use_res_norm");
    // auto extra_ortho   = args__.exist("extra_ortho");
    auto num_bands = args__.value<int>("num_bands", -1);
    auto tol       = args__.value<double>("tol", 1e-13);

    bool add_dion{!only_kin};
    bool add_vloc{!only_kin};

    PROFILE_START("test_lr_solver|setup")

    /* create simulation context */
    auto json_conf                          = R"({
      "parameters" : {
        "electronic_structure_method" : "pseudopotential"
      }
    })"_json;
    json_conf["control"]["processing_unit"] = args__.value<std::string>("device", "CPU");
    json_conf["control"]["mpi_grid_dims"]   = mpi_grid;
    // json_conf["control"]["std_evp_solver_name"] = solver;
    // json_conf["control"]["gen_evp_solver_name"] = solver;
    json_conf["parameters"]["pw_cutoff"]   = pw_cutoff;
    json_conf["parameters"]["gk_cutoff"]   = gk_cutoff;
    json_conf["parameters"]["gamma_point"] = false;
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
    PROFILE_STOP("test_lr_solver|setup")

    if (mpi::Communicator::world().rank() == 0) {
        std::cout << "number of atomic orbitals: " << ctx.unit_cell().num_ps_atomic_wf().first << "\n";
    }

    Density rho(ctx);
    rho.initial_density();

    Potential pot(ctx);
    pot.generate(rho, ctx.use_symmetry(), true);

    /* repeat several times for the accurate performance measurment */
    for (int r = 0; r < args__.value<int>("repeat", 1); r++) {
        std::array<double, 3> vk({0.1, 0.1, 0.1});

        solve_lr<double, double>(ctx, vk, pot, tol);
    }
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
                   {"tol=", "(double) CG solver tolerance"},
                   {"only_kin", "use kinetic-operator only"},
                   {"repeat=", "{int} number of repetitions"}});

    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test_lr_solver(args);
    sirius::finalize();
}
