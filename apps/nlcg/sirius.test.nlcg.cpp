/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include "core/profiler.hpp"
#include "core/json.hpp"
#include "nlcglib/adaptor.hpp"

using namespace sirius;
using json = nlohmann::json;

const std::string aiida_output_file = "output_aiida.json";

enum class task_t : int
{
    ground_state_new     = 0,
    ground_state_restart = 1,
    k_point_path         = 2
};

void
json_output_common(json& dict__)
{
    dict__["git_hash"] = sirius::git_hash();
    // dict__["build_date"] = build_date;
    dict__["comm_world_size"]  = Communicator::world().size();
    dict__["threads_per_rank"] = omp_get_max_threads();
}

std::unique_ptr<Simulation_context>
create_sim_ctx(std::string fname__, cmd_args const& args__)
{
    auto ctx_ptr = std::unique_ptr<Simulation_context>(new Simulation_context(fname__, Communicator::world()));
    Simulation_context& ctx = *ctx_ptr;

    auto& inp = ctx.parameters_input();
    if (inp.gamma_point_ && !(inp.ngridk_[0] * inp.ngridk_[1] * inp.ngridk_[2] == 1)) {
        RTE_THROW("this is not a Gamma-point calculation")
    }

    ctx.import(args__);

    return ctx_ptr;
}

double
ground_state(Simulation_context& ctx, task_t task, cmd_args const& args, int write_output)
{
    print_memory_usage(ctx.out(), FILE_LINE);

    auto& inp = ctx.parameters_input();

    std::string ref_file = args.value<std::string>("test_against", "");
    /* don't write output if we compare against the reference calculation */
    bool write_state = (ref_file.size() == 0);

    K_point_set kset(ctx, ctx.parameters_input().ngridk_, ctx.parameters_input().shiftk_, ctx.use_symmetry());
    DFT_ground_state dft(kset);

    print_memory_usage(ctx.out(), FILE_LINE);

    auto& potential = dft.potential();
    auto& density   = dft.density();

    if (task == task_t::ground_state_restart) {
        if (!utils::file_exists(storage_file_name)) {
            RTE_THROW("storage file is not found");
        }
        density.load();
        potential.load();
    } else {
        dft.initial_state();
    }

    /* launch the calculation */
    int num_dft_iter = 1;
    auto result      = dft.find(inp.density_tol_, inp.energy_tol_, ctx.cfg().iterative_solver().energy_tolerance(),
                                num_dft_iter, write_state);

    std::cout << "call my stub solver: "
              << "\n";
    Energy energy(kset, density, potential, nlcglib::smearing_type::FERMI_DIRAC);
    if (is_device_memory(ctx.preferred_memory_t())) {
        // nlcglib::nlcg_mvp2_cuda(energy);
        nlcglib::test_nlcg_mvp2_cuda(energy);
    } else {
        nlcglib::test_nlcg_mvp2(energy);
    }

    if (ctx.control().verification_ >= 1) {
        dft.check_scf_density();
    }

    auto repeat_update = args.value<int>("repeat_update", 0);
    if (repeat_update) {
        for (int i = 0; i < repeat_update; i++) {
            dft.update();
            result = dft.find(inp.density_tol_, inp.energy_tol_, initial_tol, inp.num_dft_iter_, write_state);
        }
    }

    // dft.print_magnetic_moment();

    if (ctx.control().print_stress_ && !ctx.full_potential()) {
        Stress& s       = dft.stress();
        auto stress_tot = s.calc_stress_total();
        s.print_info(dft.ctx().out(), dft.ctx().verbosity());
        result["stress"] = std::vector<std::vector<double>>(3, std::vector<double>(3));
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result["stress"][i][j] = stress_tot(j, i);
            }
        }
    }
    if (ctx.control().print_forces_) {
        Force& f         = dft.forces();
        auto& forces_tot = f.calc_forces_total();
        f.print_info(dft.ctx().out(), dft.ctx().verbosity());
        result["forces"] = std::vector<std::vector<double>>(ctx.unit_cell().num_atoms(), std::vector<double>(3));
        for (int i = 0; i < ctx.unit_cell().num_atoms(); i++) {
            for (int j = 0; j < 3; j++) {
                result["forces"][i][j] = forces_tot(j, i);
            }
        }
    }

    if (ref_file.size() != 0) {
        json dict_ref;
        std::ifstream(ref_file) >> dict_ref;

        double e1 = result["energy"]["total"];
        double e2 = dict_ref["ground_state"]["energy"]["total"];

        if (std::abs(e1 - e2) > 1e-5) {
            std::printf("total energy is different: %18.7f computed vs. %18.7f reference\n", e1, e2);
            ctx.comm().abort(1);
        }
        if (result.count("stress") && dict_ref["ground_state"].count("stress")) {
            double diff{0};
            auto s1 = result["stress"].get<std::vector<std::vector<double>>>();
            auto s2 = dict_ref["ground_state"]["stress"].get<std::vector<std::vector<double>>>();
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    diff += std::abs(s1[i][j] - s2[i][j]);
                }
            }
            if (diff > 1e-5) {
                std::printf("total stress is different!");
                std::cout << "  reference: " << dict_ref["ground_state"]["stress"] << "\n";
                std::cout << "  computed: " << result["stress"] << "\n";
                ctx.comm().abort(2);
            }
        }
        if (result.count("forces") && dict_ref["ground_state"].count("forces")) {
            double diff{0};
            auto s1 = result["forces"].get<std::vector<std::vector<double>>>();
            auto s2 = dict_ref["ground_state"]["forces"].get<std::vector<std::vector<double>>>();
            for (int i = 0; i < ctx.unit_cell().num_atoms(); i++) {
                for (int j = 0; j < 3; j++) {
                    diff += std::abs(s1[i][j] - s2[i][j]);
                }
            }
            if (diff > 1e-6) {
                std::printf("total force is different!");
                std::cout << "  reference: " << dict_ref["ground_state"]["forces"] << "\n";
                std::cout << "  computed: " << result["forces"] << "\n";
                ctx.comm().abort(3);
            }
        }
    }

    if (write_state && write_output) {
        json dict;
        json_output_common(dict);

        dict["task"]                                   = static_cast<int>(task);
        dict["ground_state"]                           = result;
        dict["timers"]                                 = utils::timer::serialize();
        dict["counters"]                               = json::object();
        dict["counters"]["local_operator_num_applied"] = Local_operator::num_applied();
        dict["counters"]["band_evp_work_count"]        = Band::evp_work_count();

        if (ctx.comm().rank() == 0) {
            std::string output_file = args.value<std::string>("output", std::string("output_") + ctx.start_time_tag() +
                                                                                std::string(".json"));
            std::ofstream ofs(output_file, std::ofstream::out | std::ofstream::trunc);
            ofs << dict.dump(4);
        }

        // if (args.exist("aiida_output")) {
        //     json dict;
        //     json_output_common(dict);
        //     dict["task"] = static_cast<int>(task);
        //     if (result >= 0) {
        //         dict["task_status"] = "converged";
        //         dict["num_scf_iterations"] =  result;
        //     } else {
        //         dict["task_status"] = "unconverged";
        //     }
        //     dict["volume"] = ctx.unit_cell().omega() * std::pow(bohr_radius, 3);
        //     dict["volume_units"] = "angstrom^3";
        //     dict["energy"] = dft.total_energy() * ha2ev;
        //     dict["energy_units"] = "eV";
        //     if (ctx.comm().rank() == 0) {
        //         std::ofstream ofs(aiida_output_file, std::ofstream::out | std::ofstream::trunc);
        //         ofs << dict.dump(4);
        //     }
        // }
    }

    /* wait for all */
    ctx.comm().barrier();

    return dft.total_energy();
}

/// Run a task based on a command line input.
void
run_tasks(cmd_args const& args)
{
    /* get the task id */
    task_t task = static_cast<task_t>(args.value<int>("task", 0));
    /* get the input file name */
    std::string fname = args.value<std::string>("input", "sirius.json");
    if (!utils::file_exists(fname)) {
        if (Communicator::world().rank() == 0) {
            std::printf("input file does not exist\n");
        }
        return;
    }

    if (task == task_t::ground_state_new || task == task_t::ground_state_restart) {
        auto ctx = create_sim_ctx(fname, args);
        ctx->initialize();
        // if (ctx->full_potential()) {
        //     ctx->gk_cutoff(ctx->aw_cutoff() / ctx->unit_cell().min_mt_radius());
        // }
        ground_state(*ctx, task, args, 1);
    }

    if (task == task_t::k_point_path) {
        auto ctx = create_sim_ctx(fname, args);
        ctx->iterative_solver().energy_tolerance(1e-12);
        ctx->gamma_point(false);
        ctx->initialize();
        // if (ctx->full_potential()) {
        //     ctx->gk_cutoff(ctx->aw_cutoff() / ctx->unit_cell().min_mt_radius());
        // }

        Potential potential(*ctx);

        Density density(*ctx);

        K_point_set ks(*ctx);

        json inp;
        std::ifstream(fname) >> inp;

        /* list of pairs (label, k-point vector) */
        std::vector<std::pair<std::string, std::vector<double>>> vertex;

        auto labels = inp["kpoints_path"].get<std::vector<std::string>>();
        for (auto e : labels) {
            auto v = inp["kpoints_rel"][e].get<std::vector<double>>();
            vertex.push_back({e, v});
        }

        std::vector<double> x_axis;
        std::vector<std::pair<double, std::string>> x_ticks;

        /* first point */
        x_axis.push_back(0);
        x_ticks.push_back({0, vertex[0].first});
        ks.add_kpoint(&vertex[0].second[0], 1.0);

        double t{0};
        for (size_t i = 0; i < vertex.size() - 1; i++) {
            vector3d<double> v0      = vector3d<double>(vertex[i].second);
            vector3d<double> v1      = vector3d<double>(vertex[i + 1].second);
            vector3d<double> dv      = v1 - v0;
            vector3d<double> dv_cart = ctx->unit_cell().reciprocal_lattice_vectors() * dv;
            int np                   = std::max(10, static_cast<int>(30 * dv_cart.length()));
            for (int j = 1; j <= np; j++) {
                vector3d<double> v = v0 + dv * static_cast<double>(j) / np;
                ks.add_kpoint(&v[0], 1.0);
                t += dv_cart.length() / np;
                x_axis.push_back(t);
            }
            x_ticks.push_back({t, vertex[i + 1].first});
        }

        ks.initialize();

        // density.initial_density();
        density.load();
        potential.generate(density, ctx->use_symmetry(), true);
        Band band(*ctx);
        Hamiltonian0 H0(potential);
        if (!ctx->full_potential()) {
            band.initialize_subspace(ks, H0);
            if (ctx->hubbard_correction()) {
                RTE_THROW("fix me");
                potential.U().hubbard_compute_occupation_numbers(
                        ks); // TODO: this is wrong; U matrix should come form the saved file
                potential.U().calculate_hubbard_potential_and_energy();
            }
        }
        band.solve(ks, H0, true);

        ks.sync_band<sync_band_t::_energy>();
        if (Communicator::world().rank() == 0) {
            json dict;
            dict["header"]                 = {};
            dict["header"]["x_axis"]       = x_axis;
            dict["header"]["x_ticks"]      = std::vector<json>();
            dict["header"]["num_bands"]    = ctx->num_bands();
            dict["header"]["num_mag_dims"] = ctx->num_mag_dims();
            for (auto& e : x_ticks) {
                json j;
                j["x"]     = e.first;
                j["label"] = e.second;
                dict["header"]["x_ticks"].push_back(j);
            }
            dict["bands"] = std::vector<json>();

            for (int ik = 0; ik < ks.num_kpoints(); ik++) {
                json bnd_k;
                bnd_k["kpoint"] = std::vector<double>(3, 0);
                for (int x = 0; x < 3; x++) {
                    bnd_k["kpoint"][x] = ks[ik]->vk()[x];
                }
                std::vector<double> bnd_e;

                for (int ispn = 0; ispn < ctx->num_spin_dims(); ispn++) {
                    for (int j = 0; j < ctx->num_bands(); j++) {
                        bnd_e.push_back(ks[ik]->band_energy(j, ispn));
                    }
                }
                // ks.get_band_energies(ik, bnd_e.data());
                bnd_k["values"] = bnd_e;
                dict["bands"].push_back(bnd_k);
            }
            std::ofstream ofs("bands.json", std::ofstream::out | std::ofstream::trunc);
            ofs << dict.dump(4);
        }
    }
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--input=", "{string} input file name");
    args.register_key("--output=", "{string} output file name");
    args.register_key("--task=", "{int} task id");
    args.register_key("--aiida_output", "write output for AiiDA");
    args.register_key("--test_against=", "{string} json file with reference values");
    args.register_key("--repeat_update=", "{int} number of times to repeat update()");
    args.register_key("--control.processing_unit=", "");
    args.register_key("--control.verbosity=", "");
    args.register_key("--control.verification=", "");
    args.register_key("--control.mpi_grid_dims=", "");
    args.register_key("--control.std_evp_solver_name=", "");
    args.register_key("--control.gen_evp_solver_name=", "");
    args.register_key("--control.fft_mode=", "");
    args.register_key("--control.memory_usage=", "");
    args.register_key("--parameters.ngridk=", "");
    args.register_key("--parameters.gamma_point=", "");
    args.register_key("--parameters.pw_cutoff=", "");
    args.register_key("--iterative_solver.orthogonalize=", "");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        std::printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);

    run_tasks(args);

    // int my_rank = Communicator::world().rank();

    sirius::finalize(1);

    // if (my_rank == 0)  {
    //     const auto timing_result = ::utils::global_rtgraph_timer.process();
    //     std::cout<< timing_result.print();
    //     std::ofstream ofs("timers.json", std::ofstream::out | std::ofstream::trunc);
    //     ofs << timing_result.json();
    // }

    return 0;
}
