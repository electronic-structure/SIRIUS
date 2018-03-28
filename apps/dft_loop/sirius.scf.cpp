#include <sirius.h>
#include <json.hpp>

using namespace sirius;
using json = nlohmann::json;

const std::string aiida_output_file = "output_aiida.json";

enum class task_t : int
{
    ground_state_new     = 0,
    ground_state_restart = 1,
    k_point_path         = 2
};

const double au2angs = 0.5291772108;

void json_output_common(json& dict__)
{
    dict__["git_hash"] = git_hash;
    dict__["build_date"] = build_date;
    dict__["comm_world_size"] = mpi_comm_world().size();
    dict__["threads_per_rank"] = omp_get_max_threads();
}

std::unique_ptr<Simulation_context> create_sim_ctx(std::string     fname__,
                                                   cmd_args const& args__)
{
    auto ctx_ptr = std::unique_ptr<Simulation_context>(new Simulation_context(fname__, mpi_comm_world()));
    Simulation_context& ctx = *ctx_ptr;

    auto& inp = ctx.parameters_input();
    if (inp.gamma_point_ && !(inp.ngridk_[0] * inp.ngridk_[1] * inp.ngridk_[2] == 1)) {
        TERMINATE("this is not a Gamma-point calculation")
    }

    auto mpi_grid_dims = args__.value<std::vector<int>>("mpi_grid", ctx.mpi_grid_dims());
    ctx.set_mpi_grid_dims(mpi_grid_dims);

    auto std_evp_solver_name = args__.value<std::string>("std_evp_solver_name", ctx.control().std_evp_solver_name_);
    ctx.set_std_evp_solver_name(std_evp_solver_name);

    auto gen_evp_solver_name = args__.value<std::string>("gen_evp_solver_name", ctx.control().gen_evp_solver_name_);
    ctx.set_gen_evp_solver_name(gen_evp_solver_name);

    auto pu = args__.value<std::string>("processing_unit", ctx.control().processing_unit_);
    if (pu == "") {
        #ifdef __GPU
        pu = "gpu";
        #else
        pu = "cpu";
        #endif
    }
    ctx.set_processing_unit(pu);

    return std::move(ctx_ptr);
}


double ground_state(Simulation_context& ctx,
                    task_t              task,
                    cmd_args const&     args,
                    int                 write_output)
{
    if (ctx.comm().rank() == 0 && ctx.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    //Potential potential(ctx);
    //potential.allocate();

    //Density density(ctx);
    //density.allocate();

    //Hamiltonian H(ctx, potential);

    //if (ctx.comm().rank() == 0 && ctx.control().print_memory_usage_) {
    //    MEMORY_USAGE_INFO();
    //}

    auto& inp = ctx.parameters_input();

    //K_point_set ks(ctx, inp.ngridk_, inp.shiftk_, ctx.use_symmetry());
    //ks.initialize();

    //if (ctx.comm().rank() == 0 && ctx.control().print_memory_usage_) {
    //    MEMORY_USAGE_INFO();
    //}

    std::string ref_file = args.value<std::string>("test_against", "");
    /* don't write output if we compare against the reference calculation */
    bool write_state = (ref_file.size() == 0);

    //DFT_ground_state dft(ctx, H, density, ks);

    DFT_ground_state dft(ctx);

    if (ctx.comm().rank() == 0 && ctx.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    auto& potential = dft.potential();
    auto& density = dft.density();

    if (task == task_t::ground_state_restart) {
        if (!Utils::file_exists(storage_file_name)) {
            TERMINATE("storage file is not found");
        }
        density.load();
        potential.load();
    } else {
        dft.initial_state();
        //density.initial_density();
        //potential.generate(density);
        //if (!ctx.full_potential()) {
        //    dft.band().initialize_subspace(dft.k_point_set(), dft.hamiltonian());
        //}
    }

    /* launch the calculation */
    int result = dft.find(inp.potential_tol_, inp.energy_tol_, inp.num_dft_iter_, write_state);

    dft.print_magnetic_moment();

    if (ref_file.size() != 0) {
        auto dict = dft.serialize();
        json dict_ref;
        std::ifstream(ref_file) >> dict_ref;

        double e1 = dict["energy"]["total"];
        double e2 = dict_ref["ground_state"]["energy"]["total"];

        if (std::abs(e1 - e2) > 1e-6) {
            printf("total energy is different: %18.7f computed vs. %18.7f reference\n", e1, e2);
            sirius::terminate(1);
        }
    }

    if (!ctx.full_potential()) {
        if (ctx.control().print_stress_) {
            Stress s(ctx, dft.k_point_set(), density, potential);
            s.calc_stress_total();
            s.print_info();
        }
        if (ctx.control().print_forces_) {
            Force f(ctx, density, potential, dft.hamiltonian(), dft.k_point_set());
            f.calc_forces_total();
            f.print_info();
        }
    }

    if (write_state && write_output) {
        json dict;
        json_output_common(dict);

        dict["task"] = static_cast<int>(task);
        dict["ground_state"] = dft.serialize();
        dict["timers"] = sddk::timer::serialize_timers();
        dict["counters"] = json::object();
        dict["counters"]["local_operator_num_applied"] = Local_operator::num_applied();
        dict["counters"]["band_evp_work_count"] = Band::evp_work_count();

        if (ctx.comm().rank() == 0) {
            std::ofstream ofs(std::string("output_") + ctx.start_time_tag() + std::string(".json"),
                              std::ofstream::out | std::ofstream::trunc);
            ofs << dict.dump(4);
        }

        if (args.exist("aiida_output")) {
            json dict;
            json_output_common(dict);
            dict["task"] = static_cast<int>(task);
            if (result >= 0) {
                dict["task_status"] = "converged";
                dict["num_scf_iterations"] =  result;
            } else {
                dict["task_status"] = "unconverged";
            }
            dict["volume"] = ctx.unit_cell().omega() * std::pow(au2angs, 3);
            dict["volume_units"] = "angstrom^3";
            dict["energy"] = dft.total_energy() * ha2ev;
            dict["energy_units"] = "eV";
            if (ctx.comm().rank() == 0) {
                std::ofstream ofs(aiida_output_file, std::ofstream::out | std::ofstream::trunc);
                ofs << dict.dump(4);
            }
        }
    }

    /* wait for all */
    ctx.comm().barrier();

    if (ctx.control().print_timers_ && ctx.comm().rank() == 0)  {
        sddk::timer::print();
    }

    return dft.total_energy();
}

/// Run a task based on a command line input.
void run_tasks(cmd_args const& args)
{
    /* get the task id */
    task_t task = static_cast<task_t>(args.value<int>("task", 0));
    /* get the input file name */
    std::string fname = args.value<std::string>("input", "sirius.json");
    if (!Utils::file_exists(fname)) {
        if (mpi_comm_world().rank() == 0) {
            printf("input file does not exist\n");
        }
        return;
    }

    if (task == task_t::ground_state_new || task == task_t::ground_state_restart) {
        auto ctx = create_sim_ctx(fname, args);
        ctx->initialize();
        ground_state(*ctx, task, args, 1);
    }

    if (task == task_t::k_point_path) {
        auto ctx = create_sim_ctx(fname, args);
        ctx->set_iterative_solver_tolerance(1e-12);
        ctx->set_gamma_point(false);
        ctx->initialize();

        Potential potential(*ctx);
        potential.allocate();

        Hamiltonian H(*ctx, potential);

        Density density(*ctx);
        density.allocate();

        K_point_set ks(*ctx);

        json inp;
        std::ifstream(fname) >> inp;

        /* list of pairs (label, k-point vector) */
        std::vector<std::pair<std::string, std::vector<double>>> vertex;

        auto labels = inp["kpoints_path"].get<std::vector<std::string>>();
        for (auto e: labels) {
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
            vector3d<double> v0 = vector3d<double>(vertex[i].second);
            vector3d<double> v1 = vector3d<double>(vertex[i + 1].second);
            vector3d<double> dv = v1 - v0;
            vector3d<double> dv_cart = ctx->unit_cell().reciprocal_lattice_vectors() * dv;
            int np = std::max(10, static_cast<int>(30 * dv_cart.length()));
            for (int j = 1; j <= np; j++) {
                vector3d<double> v = v0 + dv * static_cast<double>(j) / np;
                ks.add_kpoint(&v[0], 1.0);
                t += dv_cart.length() / np;
                x_axis.push_back(t);
            }
            x_ticks.push_back({t, vertex[i + 1].first});
        }

        ks.initialize();

        //density.initial_density();
        density.load();
        potential.generate(density);
        Band band(*ctx);
        if (!ctx->full_potential()) {
            band.initialize_subspace(ks, H);
            if (ctx->hubbard_correction()) {
                TERMINATE("fix me");
                H.U().hubbard_compute_occupation_numbers(ks); // TODO: this is wrong; U matrix should come form the saved file
                H.U().calculate_hubbard_potential_and_energy();
            }
        }
        band.solve(ks, H, true);

        ks.sync_band_energies();
        if (mpi_comm_world().rank() == 0) {
            json dict;
            dict["header"] = {};
            dict["header"]["x_axis"] = x_axis;
            dict["header"]["x_ticks"] = std::vector<json>();
            dict["header"]["num_bands"] = ctx->num_bands();
            dict["header"]["num_mag_dims"] = ctx->num_mag_dims();
            for (auto& e: x_ticks) {
                json j;
                j["x"] = e.first;
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
                //ks.get_band_energies(ik, bnd_e.data());
                bnd_k["values"] = bnd_e;
                dict["bands"].push_back(bnd_k);
            }
            std::ofstream ofs("bands.json", std::ofstream::out | std::ofstream::trunc);
            ofs << dict.dump(4);
        }
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--input=", "{string} input file name");
    args.register_key("--task=", "{int} task id");
    args.register_key("--mpi_grid=", "{vector int} MPI grid dimensions");
    args.register_key("--aiida_output", "write output for AiiDA");
    args.register_key("--test_against=", "{string} json file with reference values");
    args.register_key("--std_evp_solver_name=", "{string} standard eigen-value solver");
    args.register_key("--gen_evp_solver_name=", "{string} generalized eigen-value solver");
    args.register_key("--processing_unit=", "{string} type of the processing unit");

    args.parse_args(argn, argv);

    if (args.exist("help")) {
        printf("Usage: %s [options] \n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);

    run_tasks(args);

    sirius::finalize(1);
    return 0;
}
