#include <sirius.hpp>

using namespace sirius;
using json   = nlohmann::json;
namespace fs = std::filesystem;

inline void
json_output_common(json& dict__)
{
    dict__["git_hash"]         = sirius::git_hash();
    dict__["comm_world_size"]  = mpi::Communicator::world().size();
    dict__["threads_per_rank"] = omp_get_max_threads();
}

inline void
rewrite_relative_paths(json& dict__, fs::path const& working_directory = fs::current_path())
{
    // the json.unit_cell.atom_files[] dict might contain relative paths,
    // which should be relative to the json file. So better make them
    // absolute such that the simulation context does not have to be
    // aware of paths.
    if (!dict__.count("unit_cell")) {
        return;
    }

    auto& section = dict__["unit_cell"];

    if (!section.count("atom_files")) {
        return;
    }

    auto& atom_files = section["atom_files"];

    for (auto& label : atom_files.items()) {
        label.value() = working_directory / std::string(label.value());
    }
}

inline auto
preprocess_json_input(std::string fname__)
{
    if (fname__.find("{") == std::string::npos) {
        // If it's a file, set the working directory to that file.
        auto json = read_json_from_file(fname__);
        rewrite_relative_paths(json, fs::path{fname__}.parent_path());
        return json;
    } else {
        // Raw JSON input
        auto json = read_json_from_string(fname__);
        rewrite_relative_paths(json);
        return json;
    }
}

inline auto
create_sim_ctx(std::string fname__, cmd_args const& args__)
{
    std::string config_string;
    if (isHDF5(fname__)) {
        config_string = fname__;
    } else {
        auto json     = preprocess_json_input(fname__);
        config_string = json.dump();
    }

    auto ctx = std::make_unique<Simulation_context>(config_string);

    auto& inp = ctx->cfg().parameters();
    if (inp.gamma_point() && !(inp.ngridk()[0] * inp.ngridk()[1] * inp.ngridk()[2] == 1)) {
        RTE_THROW("this is not a Gamma-point calculation")
    }

    ctx->import(args__);

    return ctx;
}
