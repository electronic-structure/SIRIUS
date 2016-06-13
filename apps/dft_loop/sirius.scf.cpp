#include <sirius.h>

using namespace sirius;

enum class task_t
{
    ground_state_new = 0,
    ground_state_restart = 1,
    relaxation_new = 2,
    relaxation_restart = 3,
    lattice_relaxation_new = 4
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

    if (ctx.comm().rank() == 0 && aiida_output) {
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

Simulation_context* create_sim_ctx(std::string                     fname__,
                                   cmd_args const&                 args__,
                                   Parameters_input_section const& inp__)
{
    Simulation_context* ctx_ptr = new Simulation_context(fname__, mpi_comm_world());
    Simulation_context& ctx = *ctx_ptr;

    std::vector<int> mpi_grid_dims = ctx.mpi_grid_dims();
    mpi_grid_dims = args__.value< std::vector<int> >("mpi_grid", mpi_grid_dims);
    ctx.set_mpi_grid_dims(mpi_grid_dims);

    ctx.set_esm_type(inp__.esm_);
    ctx.set_num_fv_states(inp__.num_fv_states_);
    ctx.set_smearing_width(inp__.smearing_width_);
    for (auto& s: inp__.xc_functionals_) {
        ctx.add_xc_functional(s);
    }
    ctx.set_pw_cutoff(inp__.pw_cutoff_);
    ctx.set_aw_cutoff(inp__.aw_cutoff_);
    ctx.set_gk_cutoff(inp__.gk_cutoff_);
    if (ctx.esm_type() == full_potential_lapwlo) {
        ctx.set_lmax_apw(inp__.lmax_apw_);
        ctx.set_lmax_pot(inp__.lmax_pot_);
        ctx.set_lmax_rho(inp__.lmax_rho_);
    }
    ctx.set_num_mag_dims(inp__.num_mag_dims_);
    ctx.set_auto_rmt(inp__.auto_rmt_);
    ctx.set_core_relativity(inp__.core_relativity_);
    ctx.set_valence_relativity(inp__.valence_relativity_);
    ctx.set_gamma_point(inp__.gamma_point_);

    return ctx_ptr;
}

double ground_state(Simulation_context&       ctx,
                    task_t                    task,
                    cmd_args const&           args,
                    Parameters_input_section& inp,
                    int                       write_output)
{
    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif
    
    Potential potential(ctx);
    potential.allocate();

    Density density(ctx);
    density.allocate();

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    K_set ks(ctx, ctx.mpi_grid().communicator(1 << _mpi_dim_k_), inp.ngridk_, inp.shiftk_, inp.use_symmetry_);
    ks.initialize();
    
    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif
    
    DFT_ground_state dft(ctx, potential, density, ks, inp.use_symmetry_);

    if (task == task_t::ground_state_restart) {
        if (!Utils::file_exists(storage_file_name)) {
            TERMINATE("storage file is not found");
        }
        density.load();
        potential.load();
    } else {
        density.initial_density();
        dft.generate_effective_potential();
        if (!ctx.full_potential()) {
            dft.initialize_subspace();
        }
    }
    
    int result = dft.find(inp.potential_tol_, inp.energy_tol_, inp.num_dft_iter_);
    
    if (write_output) {
        write_json_output(ctx, dft, args.exist("aiida_output"), result);
    }

    /* wait for all */
    ctx.comm().barrier();

    runtime::Timer::print();

    return dft.total_energy();
}

double Etot_fake(vector3d<double> a0__, vector3d<double> a1__, vector3d<double> a2__)
{
    vector3d<double> a0{0, 5, 5};
    vector3d<double> a1{5, 0, 5};
    vector3d<double> a2{5, 5, 0};

    return std::pow((a0 - a0__).length(), 2) + std::pow((a1 - a1__).length(), 2) + std::pow((a2 - a2__).length(), 2);
}

void lattice_relaxation(task_t task, cmd_args args, Parameters_input_section& inp)
{
    /* get the input file name */
    std::string fname = args.value<std::string>("input", "sirius.json");

    std::vector< matrix3d<double> > strain;

    strain.push_back({{1, 0, 0},
                      {0, 0, 0},
                      {0, 0, 0}});

    strain.push_back({{0, 0, 0},
                      {0, 1, 0},
                      {0, 0, 0}});

    strain.push_back({{0, 0, 0},
                      {0, 0, 0},
                      {0, 0, 1}});
    
    strain.push_back({{0, 1, 0},
                      {1, 0, 0},
                      {0, 0, 0}});
    
    strain.push_back({{0, 0, 0},
                      {0, 0, 1},
                      {0, 1, 0}});
    
    strain.push_back({{0, 0, 1},
                      {0, 0, 0},
                      {1, 0, 0}});

    Simulation_context* ctx0 = create_sim_ctx(fname, args, inp);
    auto a0 = ctx0->unit_cell().lattice_vector(0);
    auto a1 = ctx0->unit_cell().lattice_vector(1);
    auto a2 = ctx0->unit_cell().lattice_vector(2);

    ///double step = 1e-4;
    std::vector<double> lat_step(6, 1e-1);
    std::vector<double> stress_prev(6, 0);
    
    matrix3d<double> mt, mtprev;

    for (int iter = 0; iter < 100; iter++) {
        matrix3d<double> latv;
        for (int x: {0, 1, 2}) {
            latv(x, 0) = a0[x];
            latv(x, 1) = a1[x];
            latv(x, 2) = a2[x];
        }
        std::vector< matrix3d<double> > strain_cart;
        for (int i = 0; i < 6; i++) {
            strain_cart.push_back(latv * strain[i]);
        }

        double e0 = Etot_fake(a0, a1, a2);
        std::vector<double> stress(6);
        for (int i = 0; i < 6; i++) {
            double step = lat_step[i] / 8;
            vector3d<double> c0 = a0;
            vector3d<double> c1 = a1;
            vector3d<double> c2 = a2;

            for (int j = 0; j < 3; j++) {
                c0[j] += step * strain[i](0, j);
                c1[j] += step * strain[i](1, j);
                c2[j] += step * strain[i](2, j);
            }

            double e1 = Etot_fake(c0, c1, c2);

            stress[i] = -(e1 - e0) / step;
        }

        for (int i = 0; i < 6; i++) {
            if (stress[i] * stress_prev[i] >= 0) {
                lat_step[i] *= 1.25;
            }
            else {
                lat_step[i] *= 0.5;
            }
        }
        stress_prev = stress;
        
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 3; j++) {
                a0[j] += stress[i] * lat_step[i] * strain[i](0, j);
                a1[j] += stress[i] * lat_step[i] * strain[i](1, j);
                a2[j] += stress[i] * lat_step[i] * strain[i](2, j);
            }
        }

        double d{0};
        for (int i = 0; i < 6; i++) {
            d += std::abs(lat_step[i]);
        }
        std::cout << "diff in step=" << d << std::endl;
        if (d < 1e-7) {
            printf("Done in %i iterations!\n", iter);
            break;
        }

        for (int i = 0; i < 6; i++) {
            std::cout << i << " " << stress[i] << " " << lat_step[i] << std::endl;
        }

        mtprev = mt;

        mt(0, 0) = a0 * a0;
        mt(0, 1) = a0 * a1;
        mt(0, 2) = a0 * a2;

        mt(1, 0) = a1 * a0;
        mt(1, 1) = a1 * a1;
        mt(1, 2) = a1 * a2;
        
        mt(2, 0) = a2 * a0;
        mt(2, 1) = a2 * a1;
        mt(2, 2) = a2 * a2;

        d = 0;
        for (int i: {0, 1, 2}) {
            for (int j: {0, 1, 2}) {
                d += std::abs(mt(i, j) - mtprev(i, j));
            }
        }
        std::cout << "diff in mt=" << d << std::endl;
        if (d < 1e-4) {
            printf("Metric tensor converged in %i iter!\n", iter);
            break;
        }


        std::cout << "new vectors: " << std::endl;
        std::cout << a0 << std::endl;
        std::cout << a1 << std::endl;
        std::cout << a2 << std::endl;
    }
}

void run_tasks(cmd_args const& args)
{
    /* get the task id */
    task_t task = static_cast<task_t>(args.value<int>("task", 0));
    /* get the input file name */
    std::string fname = args.value<std::string>("input", "sirius.json");
    /* read input section */
    JSON_tree parser(fname);
    Parameters_input_section inp;
    inp.read(parser);

    if (inp.gamma_point_ && !(inp.ngridk_[0] * inp.ngridk_[1] * inp.ngridk_[2] == 1)) {
        TERMINATE("this is not a Gamma-point calculation")
    }

    if (task == task_t::ground_state_new || task == task_t::ground_state_restart) {
        Simulation_context* ctx = create_sim_ctx(fname, args, inp);
        ctx->initialize();
        ground_state(*ctx, task, args, inp, 1);
        delete ctx;
    }

    if (task == task_t::lattice_relaxation_new) {
        lattice_relaxation(task, args, inp);
    }

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

    run_tasks(args);
    
    sirius::finalize();
}
