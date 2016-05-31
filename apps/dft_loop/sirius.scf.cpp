#include <sirius.h>

using namespace sirius;

enum class task_t
{
    ground_state_new = 0,
    ground_state_restart = 1,
    relaxation_new = 2,
    relaxation_restart = 3
};

const double au2angs = 0.5291772108;

void write_json_output(Simulation_context& ctx, DFT_ground_state& gs, bool aiida_output, int result)
{
    //double evalsum1 = gs.kset_->valence_eval_sum();
    //double evalsum2 = gs.core_eval_sum();
    //double ekin = gs.energy_kin();
    double evxc = gs.energy_vxc();
    double eexc = gs.energy_exc();
    //double ebxc = gs.energy_bxc();
    double evha = gs.energy_vha();
    double etot = gs.total_energy();
    //double gap = kset_->band_gap() * ha2ev;
    //double ef = kset_->energy_fermi();
    //double core_leak = density_->core_leakage();
    double enuc = gs.energy_enuc();

    auto ts = runtime::Timer::collect_timer_stats();
    if (ctx.comm().rank() == 0) {
        std::string fname = std::string("output_") + ctx.start_time_tag() + std::string(".json");
        JSON_write jw(fname);
        
        jw.single("git_hash", git_hash);
        jw.single("build_date", build_date);
        jw.single("num_ranks", ctx.comm().size());
        jw.single("max_num_threads", omp_get_max_threads());
        //jw.single("cyclic_block_size", p->cyclic_block_size());
        jw.single("mpi_grid", ctx.mpi_grid_dims());
        std::vector<int> fftgrid(3);
        for (int i = 0; i < 3; i++) {
            fftgrid[i] = ctx.fft().grid().size(i);
        }
        jw.single("fft_grid", fftgrid);
        jw.single("chemical_formula", ctx.unit_cell().chemical_formula());
        jw.single("num_atoms", ctx.unit_cell().num_atoms());
        jw.single("num_fv_states", ctx.num_fv_states());
        jw.single("num_bands", ctx.num_bands());
        jw.single("aw_cutoff", ctx.aw_cutoff());
        jw.single("pw_cutoff", ctx.pw_cutoff());
        jw.single("omega", ctx.unit_cell().omega());

        jw.begin_set("energy");
        jw.single("total", etot, 8);
        jw.single("evxc", evxc, 8);
        jw.single("eexc", eexc, 8);
        jw.single("evha", evha, 8);
        jw.single("enuc", enuc, 8);
        jw.end_set();
        
        //** if (num_mag_dims())
        //** {
        //**     std::vector<double> v(3, 0);
        //**     v[2] = rti().total_magnetization[0];
        //**     if (num_mag_dims() == 3)
        //**     {
        //**         v[0] = rti().total_magnetization[1];
        //**         v[1] = rti().total_magnetization[2];
        //**     }
        //**     jw.single("total_moment", v);
        //**     jw.single("total_moment_len", Utils::vector_length(&v[0]));
        //** }
        
        //** jw.single("total_energy", total_energy());
        //** jw.single("kinetic_energy", kinetic_energy());
        //** jw.single("energy_veff", rti_.energy_veff);
        //** jw.single("energy_vha", rti_.energy_vha);
        //** jw.single("energy_vxc", rti_.energy_vxc);
        //** jw.single("energy_bxc", rti_.energy_bxc);
        //** jw.single("energy_exc", rti_.energy_exc);
        //** jw.single("energy_enuc", rti_.energy_enuc);
        //** jw.single("core_eval_sum", rti_.core_eval_sum);
        //** jw.single("valence_eval_sum", rti_.valence_eval_sum);
        //** jw.single("band_gap", rti_.band_gap);
        //** jw.single("energy_fermi", rti_.energy_fermi);
        
        jw.single("timers", ts);
    }

    if (ctx.comm().rank() == 0) {
        std::string fname = std::string("output_aiida.json");
        JSON_write jw(fname);
        if (result >= 0) {
            jw.single("status", "converged");
            jw.single("num_scf_iterations", result);
        } else {
            jw.single("status", "unconverged");
        }

        jw.single("volume", ctx.unit_cell().omega() * std::pow(au2angs, 3));
        jw.single("volume_units", "angstrom^3");
        jw.single("energy", etot * ha2ev);
        jw.single("energy_units", "eV");
    }
}

void ground_state(cmd_args args)
{
    task_t task = static_cast<task_t>(args.value<int>("task", 0));

    std::string fname = args.value<std::string>("input", "sirius.json");
    
    Simulation_context ctx(fname, mpi_comm_world());

    std::vector<int> mpi_grid_dims = ctx.mpi_grid_dims();
    mpi_grid_dims = args.value< std::vector<int> >("mpi_grid", mpi_grid_dims);
    ctx.set_mpi_grid_dims(mpi_grid_dims);

    JSON_tree parser(fname);
    Parameters_input_section inp;
    inp.read(parser);

    ctx.set_esm_type(inp.esm_);
    ctx.set_num_fv_states(inp.num_fv_states_);
    ctx.set_smearing_width(inp.smearing_width_);
    for (auto& s: inp.xc_functionals_) {
        ctx.add_xc_functional(s);
    }
    ctx.set_pw_cutoff(inp.pw_cutoff_);
    ctx.set_aw_cutoff(inp.aw_cutoff_);
    ctx.set_gk_cutoff(inp.gk_cutoff_);
    ctx.set_lmax_apw(inp.lmax_apw_);
    ctx.set_lmax_pot(inp.lmax_pot_);
    ctx.set_lmax_rho(inp.lmax_rho_);
    ctx.set_num_mag_dims(inp.num_mag_dims_);
    ctx.set_auto_rmt(inp.auto_rmt_);
    ctx.set_core_relativity(inp.core_relativity_);
    ctx.set_valence_relativity(inp.valence_relativity_);

    auto ngridk = parser["parameters"]["ngridk"].get(std::vector<int>(3, 1));
    auto shiftk = parser["parameters"]["shiftk"].get(std::vector<int>(3, 0));

    if (inp.gamma_point_ && !(ngridk[0] * ngridk[1] * ngridk[2] == 1)) {
        TERMINATE("this is not a Gamma-point calculation")
    }
    ctx.set_gamma_point(inp.gamma_point_);

    ctx.initialize();
    
    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif
    
    Potential* potential = new Potential(ctx);
    potential->allocate();

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    K_set ks(ctx, ctx.mpi_grid().communicator(1 << _mpi_dim_k_), vector3d<int>(ngridk[0], ngridk[1], ngridk[2]),
             vector3d<int>(shiftk[0], shiftk[1], shiftk[2]), inp.use_symmetry_);

    ks.initialize();
    
    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif
    
    Density* density = new Density(ctx);
    density->allocate();
    
    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif
    
    DFT_ground_state dft(ctx, potential, density, &ks, inp.use_symmetry_);

    if (task == task_t::ground_state_restart) {
        if (!Utils::file_exists(storage_file_name)) {
            TERMINATE("storage file is not found");
        }
        density->load();
        potential->load();
    } else {
        density->initial_density();
        dft.generate_effective_potential();
        if (!ctx.full_potential()) {
            dft.initialize_subspace();
        }
    }
    
    double potential_tol = parser["parameters"]["potential_tol"].get(1e-4);
    double energy_tol = parser["parameters"]["energy_tol"].get(1e-4);

    //if (task_name == "test_init")
    //{
    //    potential->update_atomic_potential();
    //    ctx.unit_cell().generate_radial_functions();
    //    dft.print_info();
    //    ctx.create_storage_file();
    //    density->save();
    //}
    int result{0};
    if (task == task_t::ground_state_new || task == task_t::ground_state_restart) {
        result = dft.find(potential_tol, energy_tol, parser["parameters"]["num_dft_iter"].get(100));
    }
    //if (task_name == "gs_relax")
    //{
    //    dft.relax_atom_positions();
    //}

    write_json_output(ctx, dft, args.exist("aiida_output"), result);

    delete density;
    delete potential;

    runtime::Timer::print();
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--input=", "{string} input file name");
    args.register_key("--task=", "{int} task id");
    args.register_key("--mpi_grid=", "{vector int} MPI grid dimensions");
    args.register_key("--aiida_output", "write output for AiiDA");

    args.parse_args(argn, argv);

    if (args.exist("help")) {
        printf("Usage: %s [options] \n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);

    ground_state(args);
    
    sirius::finalize();
}
