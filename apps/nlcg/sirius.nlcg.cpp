/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include "core/json.hpp"
#include "nlcglib/call_nlcg.hpp"

using namespace sirius;
using json = nlohmann::json;

const std::string aiida_output_file = "output_aiida.json";

enum class task_t : int
{
    ground_state_new = 0
};

void
json_output_common(json& dict__)
{
    dict__["git_hash"] = sirius::git_hash();
    // dict__["build_date"] = build_date;
    dict__["comm_world_size"]  = mpi::Communicator::world().size();
    dict__["threads_per_rank"] = omp_get_max_threads();
}

std::unique_ptr<Simulation_context>
create_sim_ctx(std::string fname__, cmd_args const& args__)
{
    auto ctx_ptr = std::unique_ptr<Simulation_context>(new Simulation_context(fname__, mpi::Communicator::world()));
    Simulation_context& ctx = *ctx_ptr;

    auto& inp = ctx.cfg().parameters();
    if (inp.gamma_point() && !(inp.ngridk()[0] * inp.ngridk()[1] * inp.ngridk()[2] == 1)) {
        RTE_THROW("this is not a Gamma-point calculation")
    }

    ctx.import(args__);

    return ctx_ptr;
}

double
ground_state(Simulation_context& ctx, task_t task, cmd_args const& args, int write_output)
{
    print_memory_usage(ctx.out(), FILE_LINE);

    auto& inp = ctx.cfg().parameters();

    std::string ref_file = args.value<std::string>("test_against", "");
    /* don't write output if we compare against the reference calculation */
    bool write_state = (ref_file.size() == 0);

    std::shared_ptr<K_point_set> kset;
    if (ctx.cfg().parameters().vk().size() == 0) {
        kset = std::make_shared<K_point_set>(ctx, ctx.cfg().parameters().ngridk(), ctx.cfg().parameters().shiftk(),
                                             ctx.use_symmetry());
    } else {
        // setting
        kset = std::make_shared<K_point_set>(ctx, ctx.cfg().parameters().vk());
    }
    DFT_ground_state dft(*kset);

    print_memory_usage(ctx.out(), FILE_LINE);

    auto& potential = dft.potential();
    auto& density   = dft.density();

    dft.initial_state();

    /* launch the calculation */
    auto result = dft.find(inp.density_tol(), inp.energy_tol(), ctx.cfg().iterative_solver().energy_tolerance(),
                           inp.num_dft_iter(), write_state);

    std::cout << "nlcg after scf init (freeCudaMem): " << acc::get_free_mem() << "\n";
    std::cout << "mempool (total/free): " << get_memory_pool(memory_t::device).total_size() << " / "
              << get_memory_pool(memory_t::device).free_size() << "\n";
    auto& nlcg_params = ctx.cfg().nlcg();
    if (ctx.cfg().control().verification() >= 1) {
        dft.check_scf_density();
    }

    Energy energy(*kset, density, potential);

    nlcglib::initialize();
    call_nlcg(ctx, nlcg_params, energy, *kset, potential);
    nlcglib::finalize();

    // dft.print_magnetic_moment();

    if (ctx.cfg().control().print_stress() && !ctx.full_potential()) {
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
    if (ctx.cfg().control().print_forces()) {
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

        double e1 = result["energy"]["total"].get<double>();
        double e2 = dict_ref["ground_state"]["energy"]["total"].get<double>();

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

        dict["task"]         = static_cast<int>(task);
        dict["ground_state"] = result;
        // dict["timers"] = utils::timer::serialize();
        dict["counters"]                               = json::object();
        dict["counters"]["local_operator_num_applied"] = ctx.num_loc_op_applied();
        dict["counters"]["band_evp_work_count"]        = ctx.evp_work_count();

        if (ctx.comm().rank() == 0) {
            std::string output_file = args.value<std::string>("output", std::string("output_") + ctx.start_time_tag() +
                                                                                std::string(".json"));
            std::ofstream ofs(output_file, std::ofstream::out | std::ofstream::trunc);
            ofs << dict.dump(4);
        }
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
    if (!file_exists(fname)) {
        if (mpi::Communicator::world().rank() == 0) {
            std::printf("input file does not exist\n");
        }
        return;
    }

    if (task == task_t::ground_state_new) {
        auto ctx = create_sim_ctx(fname, args);
        ctx->initialize();
        ground_state(*ctx, task, args, 1);
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

    sirius::finalize(1);

    return 0;
}
