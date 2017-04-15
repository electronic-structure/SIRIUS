#include <sirius.h>
#include <json.hpp>

using namespace sirius;
using json = nlohmann::json;

const std::string aiida_output_file = "output_aiida.json";

enum class task_t
{
    ground_state_new = 0,
    ground_state_restart = 1,
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

    std::vector<int> mpi_grid_dims = ctx.mpi_grid_dims();
    mpi_grid_dims = args__.value<std::vector<int>>("mpi_grid", mpi_grid_dims);
    ctx.set_mpi_grid_dims(mpi_grid_dims);

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
    
    Potential potential(ctx);
    potential.allocate();

    Density density(ctx);
    density.allocate();

    if (ctx.comm().rank() == 0 && ctx.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    auto& inp = ctx.parameters_input();

    K_point_set ks(ctx, inp.ngridk_, inp.shiftk_, ctx.use_symmetry());
    ks.initialize();

    if (ctx.comm().rank() == 0 && ctx.control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    std::string ref_file = args.value<std::string>("test_against", "");
    bool write_state = (ref_file.size() == 0);
    
    DFT_ground_state dft(ctx, potential, density, ks, ctx.use_symmetry());

    if (task == task_t::ground_state_restart) {
        if (!Utils::file_exists(storage_file_name)) {
            TERMINATE("storage file is not found");
        }
        density.load();
        potential.load();
    } else {
        density.initial_density();
        potential.generate(density);
        if (!ctx.full_potential()) {
            dft.band().initialize_subspace(ks, potential);
        }
    }
    
    /* launch the calculation */
    int result = dft.find(inp.potential_tol_, inp.energy_tol_, inp.num_dft_iter_, write_state);

    if (ref_file.size() != 0) {
        json dict;
        dict["ground_state"] = dft.serialize();
        json dict_ref;
        std::ifstream(ref_file) >> dict_ref;
        
        double e1 = dict["ground_state"]["energy"]["total"];
        double e2 = dict_ref["ground_state"]["energy"]["total"];

        if (std::abs(e1 - e2) > 1e-7) {
            printf("total energy is different\n");
            exit(1);
        }

        write_output = 0;
    }
    
    if (!ctx.full_potential()) {
        //dft.forces();
        Stress s(ctx, ks, density, potential);
    }

    if (write_output) {
        json dict;
        json_output_common(dict);
        
        dict["task"] = static_cast<int>(task);
        dict["ground_state"] = dft.serialize();
        dict["timers"] = Utils::serialize_timers();
 
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

    sddk::timer::print(0);

    return dft.total_energy();
}

void run_tasks(cmd_args const& args)
{
    /* get the task id */
    task_t task = static_cast<task_t>(args.value<int>("task", 0));
    /* get the input file name */
    std::string fname = args.value<std::string>("input", "sirius.json");
    if (!Utils::file_exists(fname)) {
        TERMINATE("input file does not exist");
    }

    if (task == task_t::ground_state_new || task == task_t::ground_state_restart) {
        auto ctx = create_sim_ctx(fname, args);
        ctx->initialize();
        ground_state(*ctx, task, args, 1);
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

    args.parse_args(argn, argv);

    if (args.exist("help")) {
        printf("Usage: %s [options] \n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);

    run_tasks(args);
    
    sirius::finalize();
    return 0;
}
